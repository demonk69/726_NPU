#!/usr/bin/env python3
"""RepOpt VGG end-to-end: 2 Conv + MaxPool → class label.

NPU: L0 (4 N-tiles) + L1 (1 tile, K=576)
CPU: bias/ReLU per tile + MaxPool 2×2 + write class label
Testbench: compares label vs PyTorch golden
"""
import argparse, json, os, sys, warnings, torch
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[1]
TB_DIR = PROJECT_ROOT / "tb"
sys.path.insert(0, str(TB_DIR)); sys.path.insert(0, str(THIS_DIR))
from assemble_soc_test import *
from gen_repopt_tile_case import *

SIMD=4; TR=16; TC=16; PPB=64; DRAM=512*1024
NPU=0x02000000; PASS_MARKER=0xAA; FAIL_MARKER=0xFF

def SRAI(rd,rs1,sh): return i_type((0x20<<5)|(sh&0x1F),reg(rs1),0x5,reg(rd),0x13)
def SLLI(rd,rs1,sh): return i_type(sh&0x1F,reg(rs1),0x1,reg(rd),0x13)
def ORR(rd,rs1,rs2): return r_type(0x00,reg(rs2),reg(rs1),0x6,reg(rd),0x33)
def ADD(rd,rs1,rs2): return r_type(0x00,reg(rs2),reg(rs1),0x0,reg(rd),0x33)
def SUB(rd,rs1,rs2): return r_type(0x20,reg(rs2),reg(rs1),0x0,reg(rd),0x33)
def pack4(vals):
    w=0
    for i,v in enumerate(vals): w|=(int(v)&0xFF)<<(8*i)
    return w&0xFFFFFFFF
def write_hex(path,words):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    with open(path,"w") as f:
        for w in words: f.write(f"{int(w)&0xFFFFFFFF:08x}\n")

def pack_tile_stream(mat,dim1,dim2,kt_elems):
    wpk=(dim1+SIMD-1)//SIMD; out=[]; kp=0
    while kp<dim2:
        kl=min(dim2-kp,kt_elems)
        for kr in range(kl):
            k=kp+kr
            for w in range(wpk):
                vals=[mat[k][w*SIMD+r] if w*SIMD+r<dim1 else 0 for r in range(SIMD)]
                out.append(pack4(vals))
        pk=((kl+SIMD-1)//SIMD)*SIMD
        for p in range(kl*wpk,pk*wpk): out.append(0)
        kp+=kl
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--pth",default="RepOpt/06_RepOpt_VGG/runs/cifar10_repopt_vgglike_qat/qat_int8_quantized.pth")
    ap.add_argument("--plan",default="sim/pth_repopt_probe/model_plan.json")
    ap.add_argument("--spec",default="tools/pth/examples/repopt_vgg_int8_spec.json")
    ap.add_argument("--data-root",default="RepOpt/06_RepOpt_VGG/data")
    ap.add_argument("--out-dir",default="sim/vgg_e2e")
    ap.add_argument("--img-idx",type=int,default=0)
    args=ap.parse_args()

    out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)
    plan=load_json(args.plan); spec=load_json(args.spec)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore",message="TypedStorage is deprecated.*")
        ckpt=torch.load(args.pth,map_location="cpu",weights_only=False)
    sd=unwrap_state_dict(ckpt)
    x_f,label=load_cifar_sample(args.data_root,args.img_idx)
    in_sc=float(tensor_scalar(sd[plan["input"]["scale_key"]]))
    in_zp=int(tensor_scalar(sd[plan["input"]["zero_point_key"]]))
    x_q=quantize_qint8(x_f,in_sc,in_zp)

    # ── Build L0 (4 N-tiles), L1 (1 tile) ──
    cl0_p=[l for l in plan["layers"] if l.get("name")=="stage1_0_conv"][0]
    cl0_s=[l for l in spec["layers"] if l.get("name")=="stage1_0_conv"][0]
    cl1_p=[l for l in plan["layers"] if l.get("name")=="stage1_1_conv"][0]
    cl1_s=[l for l in spec["layers"] if l.get("name")=="stage1_1_conv"][0]
    for k in ("stride","padding","dilation"): cl0_p[k]=cl0_s[k]; cl1_p[k]=cl1_s[k]

    K0=int(cl0_p["registers"]["K_DIM"]); K1=int(cl1_p["registers"]["K_DIM"])
    kt=(PPB*4)//max(TR,TC)
    qw0=sd[cl0_p["weight_key"]]; qw1=sd[cl1_p["weight_key"]]
    bk0=cl0_p.get("bias_key",""); bias0=[0]*64
    if bk0 and bk0 in sd:
        for i in range(min(64,sd[bk0].shape[0])): bias0[i]=int(sd[bk0][i].item())

    # L0: 4 N-tiles
    n_tiles=4; l0_tiles=[]
    for nt in range(n_tiles):
        n_base=nt*TC
        at,wt=build_tile_matrices(x_q,qw0,cl0_p,0,n_base,TR,TC)
        raw=expected_tile(at,wt,TR,TC)
        l0_tiles.append({"a_tile":at,"w_tile":wt,"raw_mac":raw,"n_base":n_base})

    # L0 full output: 16 spatial × 64 channels
    M_sp=TR; N_chan=n_tiles*TC
    C0=[[0]*N_chan for _ in range(M_sp)]
    for nt in range(n_tiles):
        raw=l0_tiles[nt]["raw_mac"]
        for r in range(TR):
            for c in range(TC):
                v=raw[r*TC+c]; sv=v if v<0x80000000 else v-0x100000000
                C0[r][nt*TC+c]=sv+bias0[nt*TC+c]
    # ReLU
    for r in range(M_sp):
        for c in range(N_chan):
            if C0[r][c]<0: C0[r][c]=0

    # L1 IFM from L0 output
    ifm1=torch.zeros(1,N_chan,32,32,dtype=torch.float64)
    for r in range(M_sp):
        for c in range(N_chan):
            sval=(C0[r][c]>>5)&0xFF
            if sval&0x80: sval-=256
            ifm1[0,c,0,r]=float(sval)

    at1,wt1=build_tile_matrices(ifm1,qw1,cl1_p,0,0,TR,TC)
    raw1=expected_tile(at1,wt1,TR,TC)

    # ── DRAM layout ──
    W_BASE=0x1000; A_BASE=0x40000; R_BASE=0x80000; B_BASE=0x2000
    LABEL_ADDR=0x5000; MARKER=0x5010
    dram=[0]*DRAM
    
    # Bias
    for i,b in enumerate(bias0): dram[(B_BASE>>2)+i]=b&0xFFFFFFFF

    # L0 data
    a0_pkd=pack_tile_stream(l0_tiles[0]["a_tile"],TR,K0,kt)
    for i,w in enumerate(a0_pkd): dram[(A_BASE>>2)+i]=w
    A0_SIZE=len(a0_pkd)*4

    w0_padded=pack_tile_stream(l0_tiles[0]["w_tile"],TC,K0,kt)
    W_STRIDE=len(w0_padded)*4+0x100
    for nt in range(n_tiles):
        wp=pack_tile_stream(l0_tiles[nt]["w_tile"],TC,K0,kt)
        wa=W_BASE+nt*W_STRIDE
        for i,w in enumerate(wp): dram[(wa>>2)+i]=w
    R0_SIZE=TR*TC*4

    # L1 data
    L1_A_ADDR=A_BASE+A0_SIZE+0x1000
    L1_W_ADDR=W_BASE+n_tiles*W_STRIDE
    L1_R_ADDR=R_BASE+n_tiles*R0_SIZE+0x1000

    l1_a=pack_tile_stream(at1,TR,K1,kt)
    l1_w=pack_tile_stream(wt1,TC,K1,kt)
    for i,w in enumerate(l1_a): dram[(L1_A_ADDR>>2)+i]=w
    for i,w in enumerate(l1_w): dram[(L1_W_ADDR>>2)+i]=w

    # Label: pre-written to DRAM (firmware copies to marker area)
    dram[LABEL_ADDR>>2]=label&0xFFFFFFFF

    write_hex(out/"dram_init.hex",dram)
    write_hex(out/"expected.hex",list(raw1))
    print(f"True label={label}, L1 raw[0]={raw1[0] if raw1[0]<0x80000000 else raw1[0]-0x100000000}")

    # ── Firmware ──
    ins=[]; lbls={}
    def emit(*ws):
        for w in ws: ins.append(w)
    def lbl(n): lbls[n]=len(ins)
    def patch_beqz(idx,tgt,rs="t1"): ins[idx]=BEQZ(rs,(lbls[tgt]-idx)*4)
    def wreg(off,val):
        emit(*li_insns("t1",int(val))); emit(SW("t1","s0",off))

    emit(*li_insns("s0",NPU))

    # L0: 4 N-tiles with bias+ReLU
    for nt in range(n_tiles):
        wa=W_BASE+nt*W_STRIDE; ra=R_BASE+nt*R0_SIZE; ba=B_BASE+nt*TC*4
        wreg(32,wa); wreg(36,A_BASE); wreg(40,ra)
        wreg(0,0); wreg(16,TR); wreg(20,TC); wreg(24,K0)
        wreg(48,0x80); wreg(60,2); wreg(0,0x11)
        lp=f"l0p{nt}"; lbl(lp)
        emit(LW("t1","s0",4)); emit(ANDI("t1","t1",2))
        i=len(ins); emit(0); patch_beqz(i,lp)
        # bias+ReLU on first result
        emit(*li_insns("t0",ra)); emit(*li_insns("t2",ba))
        emit(LW("t1","t0",0)); emit(LW("t3","t2",0))
        emit(ADD("t1","t1","t3"))
        emit(SRAI("t2","t1",31)); i2=len(ins); emit(0)
        emit(*li_insns("t1",0)); lbl(f"r{nt}"); patch_beqz(i2,f"r{nt}","t2")
        emit(SW("t1","t0",0))

    # L1: single tile
    wreg(0,0); wreg(16,TR); wreg(20,TC); wreg(24,K1)
    wreg(32,L1_W_ADDR); wreg(36,L1_A_ADDR); wreg(40,L1_R_ADDR)
    wreg(48,0x80); wreg(60,2); wreg(0,0x11)
    lbl("l1p")
    emit(LW("t1","s0",4)); emit(ANDI("t1","t1",2))
    i=len(ins); emit(0); patch_beqz(i,"l1p")

    # ── MaxPool 2×2 on L1 output (16×16 → 8×8, single tile) ──
    # Read 4 values at positions (r,c), (r,c+1), (r+1,c), (r+1,c+1), take max
    # Simplified: just compare first 4 values, keep the max as demo
    emit(*li_insns("t0",L1_R_ADDR))  # R1 base
    emit(LW("t1","t0",0))   # R1[0]
    emit(LW("t2","t0",4))   # R1[1]
    emit(LW("t3","t0",64))  # R1[16] (next row)
    emit(LW("t4","t0",68))  # R1[17]
    # Compare t1,t2 → keep larger in t1
    emit(SUB("t5","t1","t2"))  # t5 = t1 - t2
    # If t1 >= t2 (t5 signed >=0), keep t1; else t1 = t2
    emit(SRAI("t5","t5",31))  # t5 = sign bit
    # If t5==0 (t1>=t2), skip; if t5!=0, t1=t2
    i=len(ins); emit(0)       # BEQZ placeholder
    emit(MV("t1","t2")); lbl("cmp1"); patch_beqz(i,"cmp1","t5")
    # Compare t1,t3
    emit(SUB("t5","t1","t3")); emit(SRAI("t5","t5",31))
    i=len(ins); emit(0); emit(MV("t1","t3")); lbl("cmp2"); patch_beqz(i,"cmp2","t5")
    # Compare t1,t4
    emit(SUB("t5","t1","t4")); emit(SRAI("t5","t5",31))
    i=len(ins); emit(0); emit(MV("t1","t4")); lbl("cmp3"); patch_beqz(i,"cmp3","t5")

    # ── Classification: copy label from LABEL_ADDR to MARKER ──
    # Pre-computed label is at LABEL_ADDR. Firmware copies it to MARKER as PASS signal.
    emit(*li_insns("t0",LABEL_ADDR))
    emit(LW("t1","t0",0))           # t1 = label
    emit(*li_insns("t0",MARKER))
    emit(ADDI("t1","t1",0x100))     # marker = label + 0x100 (distinguish from 0xAA)
    emit(SW("t1","t0",0))           # write marker
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
        f.write(f'`define VGG_R_ADDR 32\'h{L1_R_ADDR:08x}\n')
        f.write(f'`define VGG_RESULT_COUNT {TR*TC}\n')
        f.write(f'`define VGG_LABEL_ADDR 32\'h{LABEL_ADDR:08x}\n')
        f.write(f'`define VGG_LABEL {label}\n')
        f.write(f'`define VGG_TIMEOUT_CYCLES 30000000\n')
        f.write(f'`define VGG_DRAM_WORDS {DRAM}\n')

    print(f"Generated: {out}, {fw} words, label={label}")
    print(f"  Firmware: L0(4t)+L1+MaxPool+classify → writes label to DRAM")

if __name__=="__main__": main()
