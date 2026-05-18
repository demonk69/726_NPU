#!/usr/bin/env python3
"""RepOpt VGG 2-layer chain: L0 (4 N-tiles) → repack → L1.
L0 covers all 64 output channels (4 N-tiles × 16 col/tile).
L0 golden output repacked as L1 IFM in Python.
Firmware: L0 N-tile loop + L1 single tile.
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
NPU=0x02000000; PASS=0xAA

def SRAI(rd,rs1,sh): return i_type((0x20<<5)|(sh&0x1F),reg(rs1),0x5,reg(rd),0x13)
def SLLI(rd,rs1,sh): return i_type(sh&0x1F,reg(rs1),0x1,reg(rd),0x13)
def ORR(rd,rs1,rs2): return r_type(0x00,reg(rs2),reg(rs1),0x6,reg(rd),0x33)
def ADD(rd,rs1,rs2): return r_type(0x00,reg(rs2),reg(rs1),0x0,reg(rd),0x33)
def SUB(rd,rs1,rs2): return r_type(0x20,reg(rs2),reg(rs1),0x0,reg(rd),0x33)
def MUL(rd,rs1,rs2): return r_type(0x01,reg(rs2),reg(rs1),0x0,reg(rd),0x33)
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
    ap.add_argument("--out-dir",default="sim/vgg_chain")
    args=ap.parse_args()

    out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)
    plan=load_json(args.plan); spec=load_json(args.spec)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore",message="TypedStorage is deprecated.*")
        ckpt=torch.load(args.pth,map_location="cpu",weights_only=False)
    sd=unwrap_state_dict(ckpt)
    x_f,label=load_cifar_sample(args.data_root,0)
    in_sc=float(tensor_scalar(sd[plan["input"]["scale_key"]]))
    in_zp=int(tensor_scalar(sd[plan["input"]["zero_point_key"]]))
    x_q=quantize_qint8(x_f,in_sc,in_zp)

    # ── Layer 0: 4 N-tiles (channels 0..15, 16..31, 32..47, 48..63) ──
    cl0_plan=[l for l in plan["layers"] if l.get("name")=="stage1_0_conv"][0]
    cl0_spec=[l for l in spec["layers"] if l.get("name")=="stage1_0_conv"][0]
    for k in ("stride","padding","dilation"): cl0_plan[k]=cl0_spec[k]
    r0=cl0_plan.get("registers",{})
    K0=int(r0["K_DIM"]); kt=(PPB*4)//max(TR,TC)
    qw0=sd[cl0_plan["weight_key"]]
    bk0=cl0_plan.get("bias_key",""); bias0=[0]*64
    if bk0 and bk0 in sd:
        for i in range(min(64,sd[bk0].shape[0])): bias0[i]=int(sd[bk0][i].item())

    n_tiles=4
    l0_tiles=[]
    for nt in range(n_tiles):
        n_base=nt*TC
        at,wt=build_tile_matrices(x_q,qw0,cl0_plan,0,n_base,TR,TC)
        raw=expected_tile(at,wt,TR,TC)
        l0_tiles.append({"a_tile":at,"w_tile":wt,"raw_mac":raw,"n_base":n_base})

    # L0 full output (16 spatial rows × 64 channels)
    M_spatial=TR  # 16 spatial positions
    N_chan=n_tiles*TC  # 64 channels
    C0=[[0]*N_chan for _ in range(M_spatial)]
    for nt in range(n_tiles):
        raw=l0_tiles[nt]["raw_mac"]
        for r in range(TR):
            for c in range(TC):
                v=raw[r*TC+c]; sv=v if v<0x80000000 else v-0x100000000
                C0[r][nt*TC+c]=sv+bias0[nt*TC+c]

    print(f"L0: {n_tiles} N-tiles, output {M_spatial}×{N_chan}")
    print(f"  C0[0][0]={C0[0][0]}, C0[0][16]={C0[0][16]}, C0[0][63]={C0[0][63]}")

    # ── Layer 1: use repacked C0 as IFM ──
    cl1_plan=[l for l in plan["layers"] if l.get("name")=="stage1_1_conv"][0]
    cl1_spec=[l for l in spec["layers"] if l.get("name")=="stage1_1_conv"][0]
    for k in ("stride","padding","dilation"): cl1_plan[k]=cl1_spec[k]
    r1=cl1_plan.get("registers",{})
    K1=int(r1["K_DIM"])
    qw1=sd[cl1_plan["weight_key"]]

    # Build L1 IFM from C0: [1, 64, 32, 32] int8 tensor
    ifm1=torch.zeros(1,N_chan,32,32,dtype=torch.float64)
    for r in range(M_spatial):
        for c in range(N_chan):
            # C0 is int32, clamp to int8
            val=C0[r][c]; sval=val if val<0x80000000 else val-0x100000000
            sval=(sval>>5)&0xFF
            if sval&0x80: sval-=256
            ifm1[0,c,0,r]=float(sval)

    # Build L1 A tile (single spatial tile, channels 0..15)
    at1,wt1=build_tile_matrices(ifm1,qw1,cl1_plan,0,0,TR,TC)
    raw1=expected_tile(at1,wt1,TR,TC)
    r1v=raw1[0] if raw1[0]<0x80000000 else raw1[0]-0x100000000
    print(f"L1: K={K1}, raw[0]={r1v}")

    # ── DRAM ──
    W_BASE=0x1000; A_BASE=0x40000; R_BASE=0x80000; MARKER=0x5000
    dram=[0]*DRAM

    # L0 W: 4 N-tiles, each K0*TC_words words
    w0_padded=pack_tile_stream(l0_tiles[0]["w_tile"],TC,K0,kt)
    W_STRIDE_BYTES=len(w0_padded)*4+0x100

    # L0 A: one copy (same A for all N-tiles)
    a0_pkd=pack_tile_stream(l0_tiles[0]["a_tile"],TR,K0,kt)
    for i,w in enumerate(a0_pkd): dram[(A_BASE>>2)+i]=w
    A_SIZE=len(a0_pkd)*4

    # Pack L0 W for each N-tile
    for nt in range(n_tiles):
        wp=pack_tile_stream(l0_tiles[nt]["w_tile"],TC,K0,kt)
        wa=W_BASE+nt*W_STRIDE_BYTES
        for i,w in enumerate(wp): dram[(wa>>2)+i]=w

    # L0 R: n_tiles * 256 int32 values
    R0_SIZE=TR*TC*4

    # L1 A (repacked IFM for single spatial tile, channels 0..15)
    l1_a=pack_tile_stream(at1,TR,K1,kt)
    L1_A_ADDR=A_BASE+A_SIZE+0x1000

    # L1 W (single N-tile, channels 0..15)
    l1_w=pack_tile_stream(wt1,TC,K1,kt)
    L1_W_ADDR=W_BASE+n_tiles*W_STRIDE_BYTES

    # L1 R
    L1_R_ADDR=R_BASE+n_tiles*R0_SIZE+0x1000

    for i,w in enumerate(l1_a): dram[(L1_A_ADDR>>2)+i]=w
    for i,w in enumerate(l1_w): dram[(L1_W_ADDR>>2)+i]=w

    write_hex(out/"dram_init.hex",dram)

    # Expected: L1 raw MAC
    e_raw=list(raw1)
    write_hex(out/"expected.hex",e_raw)

    # ── Firmware ──
    ins=[]; lbls={}
    def emit(*ws):
        for w in ws: ins.append(w)
    def lbl(n): lbls[n]=len(ins)
    def patch_beqz(idx,tgt,rs="t1"): ins[idx]=BEQZ(rs,(lbls[tgt]-idx)*4)
    def wreg(off,val):
        emit(*li_insns("t1",int(val))); emit(SW("t1","s0",off))

    emit(*li_insns("s0",NPU))
    # L0 N-tile loop (unrolled 4 iterations — avoids RV32I M extension)
    for nt in range(n_tiles):
        wa=W_BASE+nt*W_STRIDE_BYTES
        ra=R_BASE+nt*R0_SIZE
        wreg(32,wa); wreg(36,A_BASE); wreg(40,ra)
        wreg(0,0); wreg(16,TR); wreg(20,TC); wreg(24,K0)
        wreg(48,0x80); wreg(60,2); wreg(0,0x11)
        lp=f"l0p{nt}"
        lbl(lp)
        emit(LW("t1","s0",4)); emit(ANDI("t1","t1",2))
        i=len(ins); emit(0); patch_beqz(i,lp)

    # L1: single tile
    wreg(0,0); wreg(16,TR); wreg(20,TC); wreg(24,K1)
    wreg(32,L1_W_ADDR); wreg(36,L1_A_ADDR); wreg(40,L1_R_ADDR)
    wreg(48,0x80); wreg(60,2); wreg(0,0x11)
    lbl("l1p")
    emit(LW("t1","s0",4)); emit(ANDI("t1","t1",2))
    i=len(ins); emit(0); patch_beqz(i,"l1p")

    # PASS
    emit(*li_insns("t0",MARKER)); emit(*li_insns("t1",0xAA)); emit(SW("t1","t0",0))
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
        f.write(f'`define VGG_TIMEOUT_CYCLES 20000000\n')
        f.write(f'`define VGG_DRAM_WORDS {DRAM}\n')

    print(f"\nGenerated: {out}, {fw} words")
    print(f"  L1 golden raw[0] = {r1v}")

if __name__=="__main__": main()
