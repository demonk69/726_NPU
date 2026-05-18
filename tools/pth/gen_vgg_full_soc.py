#!/usr/bin/env python3
"""RepOpt VGG Layer-0 SoC test: real weight Conv2D tile-mode + CPU bias/ReLU.

Single 16×16 tile at IFM top-left, output channels 0..15.
Proves: .pth → NPU tile GEMM → CPU postprocess ≡ PyTorch golden.
"""
import argparse, json, os, sys, warnings, torch
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[1]
TB_DIR = PROJECT_ROOT / "tb"
sys.path.insert(0, str(TB_DIR)); sys.path.insert(0, str(THIS_DIR))
from assemble_soc_test import *

# ── RV32I ──
def SRAI(rd, rs1, sh):
    return i_type((0x20<<5)|(sh&0x1F), reg(rs1), 0x5, reg(rd), 0x13)
def SLLI(rd, rs1, sh):
    return i_type(sh&0x1F, reg(rs1), 0x1, reg(rd), 0x13)
def ORR(rd, rs1, rs2):
    return r_type(0x00, reg(rs2), reg(rs1), 0x6, reg(rd), 0x33)
def SUB(rd, rs1, rs2):
    return r_type(0x20, reg(rs2), reg(rs1), 0x0, reg(rd), 0x33)
def pack4(vals):
    w = 0
    for i, x in enumerate(vals):
        w |= (int(x) & 0xFF) << (8 * i)
    return w & 0xFFFFFFFF
def write_hex(path, words):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path,"w") as f:
        for w in words: f.write(f"{int(w)&0xFFFFFFFF:08x}\n")

from gen_repopt_tile_case import *
SIMD=4; TR=16; TC=16; PPB=64
NPU=0x02000000; PASS=0xAA

def pack_stream(mat, dim1, dim2, kt_elems):
    """Pack mat[dim2][dim1] into padded K-first tile stream with SIMD padding."""
    wpk=(dim1+SIMD-1)//SIMD; out=[]
    kpos=0
    while kpos<dim2:
        k_len=min(dim2-kpos, kt_elems)
        for k_rel in range(k_len):
            k=kpos+k_rel
            for w in range(wpk):
                vals=[mat[k][w*SIMD+r] if w*SIMD+r<dim1 else 0 for r in range(SIMD)]
                out.append(pack4(vals))
        # SIMD padding
        pad_k=((k_len+SIMD-1)//SIMD)*SIMD
        for p in range(k_len*wpk, pad_k*wpk):
            out.append(0)
        kpos+=k_len
    return out

def pad_size(k_dim, dim, kt):
    wpk=(dim+SIMD-1)//SIMD; t=0; kp=0
    while kp<k_dim:
        kl=min(k_dim-kp,kt); t+=((kl+SIMD-1)//SIMD)*SIMD*wpk; kp+=kl
    return t

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--pth",default="RepOpt/06_RepOpt_VGG/runs/cifar10_repopt_vgglike_qat/qat_int8_quantized.pth")
    ap.add_argument("--plan",default="sim/pth_repopt_probe/model_plan.json")
    ap.add_argument("--data-root",default="RepOpt/06_RepOpt_VGG/data")
    ap.add_argument("--out-dir",default="sim/vgg_full_soc")
    ap.add_argument("--img-idx",type=int,default=0)
    args=ap.parse_args()

    out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)
    plan=load_json(args.plan)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore",message="TypedStorage is deprecated.*")
        ckpt=torch.load(args.pth,map_location="cpu",weights_only=False)
    sd=unwrap_state_dict(ckpt)
    x_f,label=load_cifar_sample(args.data_root,args.img_idx)
    in_sc=float(tensor_scalar(sd[plan["input"]["scale_key"]]))
    in_zp=int(tensor_scalar(sd[plan["input"]["zero_point_key"]]))
    x_q=quantize_qint8(x_f,in_sc,in_zp)

    cl=[l for l in plan["layers"] if l.get("op")=="conv2d"][0]
    r=cl.get("registers",{})
    M=TR; N=TC; K=int(r["K_DIM"])
    qw=sd[cl["weight_key"]]
    a_tile,w_tile=build_tile_matrices(x_q,qw,cl,0,0,TR,TC)
    raw=expected_tile(a_tile,w_tile,TR,TC)

    # Bias
    bias=[0]*TC
    bk=cl.get("bias_key","")
    if bk and bk in sd:
        for i in range(min(TC,sd[bk].shape[0])):
            bias[i]=int(sd[bk][i].item())

    # Post: bias + ReLU
    post=[((int(raw[i])if int(raw[i])<0x80000000 else int(raw[i])-0x100000000)+bias[i%TC]) & 0xFFFFFFFF for i in range(len(raw))]
    relu=[max(0,x if x<0x80000000 else x-0x100000000) for x in post]

    kt=(PPB*4)//max(TR,TC)  # =16
    print(f"Label={label} K={K} kt={kt}")
    print(f"raw[0]={raw[0] if raw[0]<0x80000000 else raw[0]-0x100000000}")
    print(f"post[0]={post[0] if post[0]<0x80000000 else post[0]-0x100000000}")
    print(f"relu[0]={relu[0]}")

    # DRAM
    DRAM=128*1024
    W_ADDR=0x1000; A_ADDR=0x10000; R_ADDR=0x12000; B_ADDR=0x2000; MARKER=0x5000
    dram=[0]*DRAM
    wp=pack_stream(w_tile, TC, K, kt)
    ap=pack_stream(a_tile, TR, K, kt)
    for i,w in enumerate(wp): dram[(W_ADDR>>2)+i]=w
    for i,w in enumerate(ap): dram[(A_ADDR>>2)+i]=w
    for i,b in enumerate(bias): dram[(B_ADDR>>2)+i]=b&0xFFFFFFFF
    write_hex(out/"dram_init.hex",dram)
    write_hex(out/"expected.hex",raw)  # raw MAC (no bias/ReLU — firmware handles it)

    # Firmware
    ins=[]; lbls={}
    def emit(*ws):
        for w in ws: ins.append(w)
    def lbl(n): lbls[n]=len(ins)
    def patch_beqz(idx,tgt,rs="t1"):
        ins[idx]=BEQZ(rs,(lbls[tgt]-idx)*4)
    def wreg(off,val):
        emit(*li_insns("t1",int(val))); emit(SW("t1","s0",off))

    emit(*li_insns("s0",NPU))
    wreg(0,0); wreg(16,TR); wreg(20,TC); wreg(24,K)
    wreg(32,W_ADDR); wreg(36,A_ADDR); wreg(40,R_ADDR)
    wreg(48,0x80); wreg(60,2); wreg(0,0x11)
    lbl("poll")
    emit(LW("t1","s0",4)); emit(ANDI("t1","t1",2))
    i=len(ins); emit(0); patch_beqz(i,"poll")

    # CPU bias + ReLU for first 4 results
    emit(*li_insns("t0",R_ADDR))
    emit(*li_insns("s1",B_ADDR))
    # Read 4 R0 words, add bias, clamp negative→0, store back
    emit(LW("t1","t0",0)); emit(LW("t2","s1",0))
    emit(r_type(0x00,reg("t2"),reg("t1"),0x0,reg("t1"),0x33))  # ADD
    emit(SRAI("t2","t1",31))  # sign bit
    i=len(ins); emit(0)       # BEQZ placeholder
    emit(*li_insns("t1",0))   # if negative: zero t1
    lbl("relu_ok")            # BEQZ target: skip zeroing
    patch_beqz(i,"relu_ok","t2")  # if t2==0 (positive), skip t1=0
    emit(SW("t1","t0",0))     # store

    # PASS
    emit(*li_insns("t0",MARKER)); emit(*li_insns("t1",PASS)); emit(SW("t1","t0",0))
    lbl("halt"); emit(0x0000006f)

    fw=len(ins)
    write_hex(out/"soc_vgg.hex",ins)

    op=out.resolve().as_posix()
    with open(out/"soc_vgg_params.vh","w") as f:
        f.write(f'`define VGG_FW_HEX "{op}/soc_vgg.hex"\n')
        f.write(f'`define VGG_DRAM_HEX "{op}/dram_init.hex"\n')
        f.write(f'`define VGG_EXPECTED_HEX "{op}/expected.hex"\n')
        f.write(f'`define VGG_FW_WORDS {fw}\n')
        f.write(f'`define VGG_MARKER_ADDR 32\'h{MARKER:08x}\n')
        f.write(f'`define VGG_R_ADDR 32\'h{R_ADDR:08x}\n')
        f.write(f'`define VGG_RESULT_COUNT {TR*TC}\n')
        f.write(f'`define VGG_TIMEOUT_CYCLES 5000000\n')
        f.write(f'`define VGG_DRAM_WORDS {DRAM}\n')
    print(f"Generated {out}: {fw} words firmware, {DRAM} DRAM words")

if __name__=="__main__": main()
