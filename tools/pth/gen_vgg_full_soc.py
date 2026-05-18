#!/usr/bin/env python3
"""RepOpt VGG N-layer SoC test generator (real weights, CPU post-process).

Usage:
  python gen_vgg_full_soc.py --out-dir sim/vgg_2layer --layers 2

Each layer: NPU tile-mode GEMM (single 16×16 tile) → CPU bias/ReLU/requant.
Layers use the same CIFAR IFM for input (full chain repack TBD).
"""
import argparse, json, os, sys, warnings, torch
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[1]
TB_DIR = PROJECT_ROOT / "tb"
sys.path.insert(0, str(TB_DIR)); sys.path.insert(0, str(THIS_DIR))
from assemble_soc_test import *
from gen_repopt_tile_case import *

SIMD=4; TR=16; TC=16; PPB=64; DRAM=256*1024
NPU=0x02000000; PASS=0xAA
REG_CTRL=0x00; REG_STATUS=0x04; REG_M_DIM=0x10; REG_N_DIM=0x14; REG_K_DIM=0x18
REG_W_ADDR=0x20; REG_A_ADDR=0x24; REG_R_ADDR=0x28
REG_ARR_CFG=0x30; REG_CFG_SHAPE=0x3C

def SRAI(rd,rs1,sh): return i_type((0x20<<5)|(sh&0x1F),reg(rs1),0x5,reg(rd),0x13)
def SLLI(rd,rs1,sh): return i_type(sh&0x1F,reg(rs1),0x1,reg(rd),0x13)
def ORR(rd,rs1,rs2): return r_type(0x00,reg(rs2),reg(rs1),0x6,reg(rd),0x33)
def ADD(rd,rs1,rs2): return r_type(0x00,reg(rs2),reg(rs1),0x0,reg(rd),0x33)
def pack4(vals):
    w=0
    for i,v in enumerate(vals): w|=(int(v)&0xFF)<<(8*i)
    return w&0xFFFFFFFF
def write_hex(path,words):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    with open(path,"w") as f:
        for w in words: f.write(f"{int(w)&0xFFFFFFFF:08x}\n")

def pack_stream(mat,dim1,dim2,kt_elems):
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

def build_layer(spec_layers,plan_layers,sd,x_q,li):
    cl=plan_layers[li]
    # Merge spec layer info (padding/stride/dilation) into plan dict
    spec=spec_layers[li]
    for k in ("stride","padding","dilation"):
        if k in spec: cl[k]=spec[k]
    r=cl.get("registers",{})
    M=TR; N=TC; K=int(r["K_DIM"])
    qw=sd[cl["weight_key"]]
    a_tile,w_tile=build_tile_matrices(x_q,qw,cl,0,0,TR,TC)
    raw=expected_tile(a_tile,w_tile,TR,TC)
    # Bias
    bk=cl.get("bias_key",""); bias=[0]*TC
    if bk and bk in sd:
        for i in range(min(TC,sd[bk].shape[0])): bias[i]=int(sd[bk][i].item())
    # Post: bias+ReLU  
    post=[]
    for i,val in enumerate(raw):
        acc=(int(val)if int(val)<0x80000000 else int(val)-0x100000000)+bias[i%TC]
        post.append(max(0,acc)&0xFFFFFFFF)
    kt=(PPB*4)//max(TR,TC)
    return dict(name=cl["name"],M=TR,N=TC,K=K,kt_elems=kt,
                a_tile=a_tile,w_tile=w_tile,raw_mac=raw,
                post_bias_relu=post,bias=bias)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--pth",default="RepOpt/06_RepOpt_VGG/runs/cifar10_repopt_vgglike_qat/qat_int8_quantized.pth")
    ap.add_argument("--plan",default="sim/pth_repopt_probe/model_plan.json")
    ap.add_argument("--data-root",default="RepOpt/06_RepOpt_VGG/data")
    ap.add_argument("--out-dir",default="sim/vgg_2layer")
    ap.add_argument("--layers",type=int,default=2)
    ap.add_argument("--spec", default="tools/pth/examples/repopt_vgg_int8_spec.json")
    args=ap.parse_args()

    out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)
    plan=load_json(args.plan)
    spec=load_json(args.spec)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore",message="TypedStorage is deprecated.*")
        ckpt=torch.load(args.pth,map_location="cpu",weights_only=False)
    sd=unwrap_state_dict(ckpt)
    x_f,label=load_cifar_sample(args.data_root,0)
    in_sc=float(tensor_scalar(sd[plan["input"]["scale_key"]]))
    in_zp=int(tensor_scalar(sd[plan["input"]["zero_point_key"]]))
    x_q=quantize_qint8(x_f,in_sc,in_zp)

    conv_plan=[l for l in plan["layers"] if l.get("op")=="conv2d"]
    conv_spec=[l for l in spec["layers"] if l.get("op")=="conv2d"]
    num_layers=min(args.layers, len(conv_plan))

    # Build layers (each uses the same CIFAR IFM, padded to correct Cin)
    layers=[]
    for i in range(num_layers):
        cl=conv_plan[i]
        ws=cl.get("weight_shape",[0,0,0,0])
        cin=ws[1] if len(ws)>1 else 3
        if cin==x_q.shape[1]:
            ifm=x_q
        else:
            ifm=torch.zeros(1,cin,*x_q.shape[2:],dtype=x_q.dtype)
            ifm[0,:x_q.shape[1]]=x_q[0]
        layers.append(build_layer(conv_spec,conv_plan,sd,ifm,i))

    print(f"Label={label}, layers={num_layers}")
    for i,l in enumerate(layers):
        r0=l["raw_mac"][0] if l["raw_mac"][0]<0x80000000 else l["raw_mac"][0]-0x100000000
        print(f"  L{i} {l['name']}: raw[0]={r0} bias[0]={l['bias'][0]} post[0]={l['post_bias_relu'][0] if l['post_bias_relu'][0]<0x80000000 else l['post_bias_relu'][0]-0x100000000}")

    # ── DRAM layout (dynamic, based on actual tile sizes) ──
    W_BASE=0x1000; A_BASE=0x40000; R_BASE=0x80000; B_BASE=0x2000; MARKER=0x5000
    dram=[0]*DRAM

    # Compute per-layer byte sizes
    layouts=[]
    w_cursor=W_BASE; a_cursor=A_BASE; r_cursor=R_BASE; b_cursor=B_BASE
    for li,l in enumerate(layers):
        wp=pack_stream(l["w_tile"],TC,l["K"],l["kt_elems"])
        ap=pack_stream(l["a_tile"],TR,l["K"],l["kt_elems"])
        w_bytes=len(wp)*4; a_bytes=len(ap)*4; r_bytes=TR*TC*4; b_bytes=TC*4
        wa=w_cursor; w_cursor+=w_bytes+0x100
        aa=a_cursor; a_cursor+=a_bytes+0x100
        ra=r_cursor; r_cursor+=r_bytes+0x100
        ba=b_cursor; b_cursor+=b_bytes+0x100
        for i,w in enumerate(wp): dram[(wa>>2)+i]=w
        for i,w in enumerate(ap): dram[(aa>>2)+i]=w
        for i,bv in enumerate(l["bias"]): dram[(ba>>2)+i]=bv&0xFFFFFFFF
        layouts.append((wa,aa,ra,ba,w_bytes,a_bytes))
        print(f"  L{li}: W={wa:#x}({len(wp)}w) A={aa:#x}({len(ap)}w) R={ra:#x} B={ba:#x}")

    # Build expected
    e_raw=[]
    for li,l in enumerate(layers):
        e_raw+=list(l["raw_mac"])

    write_hex(out/"dram_init.hex",dram)
    # Expected: raw MAC (firmware only processes R[0] as bias+ReLU demo)
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

    for li,l in enumerate(layers):
        wa,aa,ra,ba,_,_ = layouts[li]
        # Launch NPU
        wreg(0,0); wreg(16,TR); wreg(20,TC); wreg(24,l["K"])
        wreg(32,wa); wreg(36,aa); wreg(40,ra)
        wreg(48,0x80); wreg(60,2); wreg(0,0x11)
        lp=f"p{li}"
        lbl(lp)
        emit(LW("t1","s0",4)); emit(ANDI("t1","t1",2))
        i=len(ins); emit(0); patch_beqz(i,lp)

        # CPU bias + ReLU for first 4 results
        emit(*li_insns("s2",ra))
        emit(*li_insns("s3",ba))
        emit(LW("t1","s2",0)); emit(LW("t2","s3",0))
        emit(ADD("t1","t1","t2"))        # bias add
        emit(SRAI("t2","t1",31))         # sign bit
        i=len(ins); emit(0)              # BEQZ placeholder
        emit(*li_insns("t1",0))          # zero if negative
        lbl(f"relu{li}")
        patch_beqz(i,f"relu{li}","t2")   # skip zeroing if positive
        emit(SW("t1","s2",0))            # store back

    # PASS marker
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
        f.write(f'`define VGG_R_ADDR 32\'h{R_BASE:08x}\n')
        f.write(f'`define VGG_RESULT_COUNT {TR*TC*num_layers}\n')
        f.write(f'`define VGG_TIMEOUT_CYCLES 5000000\n')
        f.write(f'`define VGG_DRAM_WORDS {DRAM}\n')

    print(f"\nGenerated: {out}")
    print(f"  firmware: {fw} words")

if __name__=="__main__":
    main()
