#!/usr/bin/env python3
"""RepOpt VGG e2e + real classifier: Conv chain + Linear 512→10 in firmware.

NPU: L0 (4 N-tiles) + L1 (1 tile, K=576)
CPU: bias/ReLU per tile + MaxPool 2×2
CPU receives 512 real features from full VGG golden torch run (stages 2-4 in Python).
CPU Classifier: 512 features × 10 classes dot product (MUL+ADD loop) → argmax
Testbench: compares predicted class vs PyTorch golden.

Input: real CIFAR-10 image. Features: real VGG layer output (not synthetic).
"""
import argparse, json, os, sys, warnings, torch
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[1]
TB_DIR = PROJECT_ROOT / "tb"
sys.path.insert(0, str(TB_DIR)); sys.path.insert(0, str(THIS_DIR))
from assemble_soc_test import *
from gen_repopt_tile_case import *
from run_repopt_vgg_host import (
    load_json, tensor_scalar, unwrap_state_dict,
    load_int32_hex, quantize_qint8, requant_qint8,
    conv2d_acc_npu, maxpool2d_cpu, adaptive_avgpool2d_cpu,
    load_cifar_sample, load_image_input,
)

SIMD=4; TR=16; TC=16; PPB=64; DRAM=512*1024
NPU=0x02000000; PASS_MARKER=0xAA

def SRAI(rd,rs1,sh): return i_type((0x20<<5)|(sh&0x1F),reg(rs1),0x5,reg(rd),0x13)
def SLLI(rd,rs1,sh): return i_type(sh&0x1F,reg(rs1),0x1,reg(rd),0x13)
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

def extract_real_features(plan,sd,x_q,out_dir):
    """Run full VGG forward pass (all conv+pool layers) and extract 512 features after avgpool+flatten."""
    current=x_q
    assets_dir = Path(out_dir)
    for layer in plan["layers"]:
        if layer["op"]=="conv2d":
            bias_acc=load_int32_hex(assets_dir/layer["assets"]["bias_int32_hex"])
            qweight=sd[layer["weight_key"]]
            acc=conv2d_acc_npu(current,qweight,bias_acc,layer)
            req=layer["cpu_requant_after_npu"]
            current=requant_qint8(acc,req["multipliers"],req["output_zero_point"])
        elif layer["op"]=="maxpool2d":
            current=maxpool2d_cpu(current,layer)
        elif layer["op"]=="adaptive_avgpool2d":
            current=adaptive_avgpool2d_cpu(current,layer)
        elif layer["op"]=="flatten":
            current=current.reshape(current.shape[0],-1)
            break
        elif layer["op"]=="linear":
            break
    return current.reshape(-1).to(torch.float64)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--pth",default="RepOpt/06_RepOpt_VGG/runs/cifar10_repopt_vgglike_qat/qat_int8_quantized.pth")
    ap.add_argument("--plan",default="sim/pth_repopt_probe/model_plan.json")
    ap.add_argument("--spec",default="tools/pth/examples/repopt_vgg_int8_spec.json")
    ap.add_argument("--data-root",default="RepOpt/06_RepOpt_VGG/data")
    ap.add_argument("--out-dir",default="sim/vgg_e2e")
    ap.add_argument("--img-idx",type=int,default=0)
    ap.add_argument("--image",default="",help="Arbitrary RGB image path (overrides --img-idx)")
    ap.add_argument("--image-size",type=int,default=32,help="Image resize size (default 32)")
    args=ap.parse_args()

    out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)
    plan=load_json(args.plan); spec=load_json(args.spec)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore",message="TypedStorage is deprecated.*")
        ckpt=torch.load(args.pth,map_location="cpu",weights_only=False)
    sd=unwrap_state_dict(ckpt)
    if args.image:
        x_f,label=load_image_input(args.image,args.image_size),None
    else:
        x_f,label=load_cifar_sample(args.data_root,args.img_idx)
    in_sc=float(tensor_scalar(sd[plan["input"]["scale_key"]]))
    in_zp=int(tensor_scalar(sd[plan["input"]["zero_point_key"]]))
    x_q=quantize_qint8(x_f,in_sc,in_zp)

    # ── Extract classifier weights ──
    cp=sd['model.classifier._packed_params._packed_params']
    cls_w=cp[0].int_repr()  # [10, 512] qint8 → int8
    cls_b=cp[1]  # [10] float32 bias
    cls_bias=[int(cls_b[i].item()) for i in range(10)]

    N_FEAT=512  # full 512 features from real VGG forward pass
    plan_dir=Path(args.plan).resolve().parent
    features=extract_real_features(plan,sd,x_q,plan_dir)
    features=torch.clamp(features,-128,127)

    # ── Compute expected scores and class ──
    scores=[cls_bias[c] for c in range(10)]
    for c in range(10):
        for f in range(N_FEAT):
            scores[c]+=int(features[f].item())*int(cls_w[c,f].item())
    pred_class=scores.index(max(scores))
    print(f"Classifier: features[0]={int(features[0].item())}, true_label={label}, pred={pred_class}, scores={[scores[c] for c in range(10)]}")

    # ── NPU pipeline (same as before) ──
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

    n_tiles=4; l0_tiles=[]
    for nt in range(n_tiles):
        n_base=nt*TC
        at,wt=build_tile_matrices(x_q,qw0,cl0_p,0,n_base,TR,TC)
        raw=expected_tile(at,wt,TR,TC)
        l0_tiles.append({"a_tile":at,"w_tile":wt,"raw_mac":raw,"n_base":n_base})

    M_sp=TR; N_chan=n_tiles*TC
    C0=[[0]*N_chan for _ in range(M_sp)]
    for nt in range(n_tiles):
        raw=l0_tiles[nt]["raw_mac"]
        for r in range(TR):
            for c in range(TC):
                v=raw[r*TC+c]; sv=v if v<0x80000000 else v-0x100000000
                C0[r][nt*TC+c]=sv+bias0[nt*TC+c]
    for r in range(M_sp):
        for c in range(N_chan):
            if C0[r][c]<0: C0[r][c]=0

    ifm1=torch.zeros(1,N_chan,32,32,dtype=torch.float64)
    for r in range(M_sp):
        for c in range(N_chan):
            sval=(C0[r][c]>>5)&0xFF
            if sval&0x80: sval-=256
            ifm1[0,c,0,r]=float(sval)

    at1,wt1=build_tile_matrices(ifm1,qw1,cl1_p,0,0,TR,TC)
    raw1=expected_tile(at1,wt1,TR,TC)

    # ── Compute full L1 output (all 4 N-tiles) as input to L2 ──
    bk1=cl1_p.get("bias_key",""); bias1=[0]*N_chan
    if bk1 and bk1 in sd:
        for i in range(min(N_chan,sd[bk1].shape[0])): bias1[i]=int(sd[bk1][i].item())
    n_tiles_l1=(N_chan+TC-1)//TC
    C1=[[0]*N_chan for _ in range(M_sp)]
    for nt in range(n_tiles_l1):
        n_base=nt*TC
        at1n,wt1n=build_tile_matrices(ifm1,qw1,cl1_p,0,n_base,TR,TC)
        raw1n=expected_tile(at1n,wt1n,TR,TC)
        for r in range(TR):
            for c in range(min(TC,N_chan-n_base)):
                v=raw1n[r*TC+c]; sv=v if v<0x80000000 else v-0x100000000
                C1[r][n_base+c]=sv+bias1[n_base+c]
    for r in range(M_sp):
        for c in range(N_chan):
            if C1[r][c]<0: C1[r][c]=0

    ifm2=torch.zeros(1,N_chan,32,32,dtype=torch.float64)
    for r in range(M_sp):
        for c in range(N_chan):
            sval=(C1[r][c]>>5)&0xFF
            if sval&0x80: sval-=256
            ifm2[0,c,0,r]=float(sval)

    # ── L2: stage2_0_conv (64→128, K=576, 1 N-tile) ──
    cl2_p=[l for l in plan["layers"] if l.get("name")=="stage2_0_conv"][0]
    cl2_s=[l for l in spec["layers"] if l.get("name")=="stage2_0_conv"][0]
    for k in ("stride","padding","dilation"): cl2_p[k]=cl2_s[k]
    K2=int(cl2_p["registers"]["K_DIM"])
    qw2=sd[cl2_p["weight_key"]]
    bk2=cl2_p.get("bias_key",""); bias2=[0]*128
    if bk2 and bk2 in sd:
        for i in range(min(128,sd[bk2].shape[0])): bias2[i]=int(sd[bk2][i].item())

    at2,wt2=build_tile_matrices(ifm2,qw2,cl2_p,0,0,TR,TC)
    raw2=expected_tile(at2,wt2,TR,TC)

    # ── DRAM layout ──
    W_BASE=0x1000; A_BASE=0x40000; R_BASE=0x80000; B_BASE=0x2000
    FEAT_BASE=0x6000; CLS_W_BASE=0x7000; CLS_B_BASE=0xC000
    SCORE_BASE=0xC100; MARKER=0xC200; LABEL_ADDR=0xC300
    dram=[0]*DRAM

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

    L1_A_ADDR=A_BASE+A0_SIZE+0x1000
    L1_W_ADDR=W_BASE+n_tiles*W_STRIDE
    L1_R_ADDR=R_BASE+n_tiles*R0_SIZE+0x1000

    l1_a=pack_tile_stream(at1,TR,K1,kt)
    l1_w=pack_tile_stream(wt1,TC,K1,kt)
    for i,w in enumerate(l1_a): dram[(L1_A_ADDR>>2)+i]=w
    for i,w in enumerate(l1_w): dram[(L1_W_ADDR>>2)+i]=w

    # L2 data
    L2_A_ADDR=L1_A_ADDR+len(l1_a)*4+0x1000
    L2_W_ADDR=L1_W_ADDR+len(l1_w)*4+0x100
    L2_R_ADDR=L1_R_ADDR+R0_SIZE+0x1000
    L2_B_ADDR=B_BASE+len(bias0)*4+0x100

    for i,b in enumerate(bias2): dram[(L2_B_ADDR>>2)+i]=b&0xFFFFFFFF
    l2_a=pack_tile_stream(at2,TR,K2,kt)
    l2_w=pack_tile_stream(wt2,TC,K2,kt)
    for i,w in enumerate(l2_a): dram[(L2_A_ADDR>>2)+i]=w
    for i,w in enumerate(l2_w): dram[(L2_W_ADDR>>2)+i]=w

    # Classifier data: features (512 int8, one per word), weights (10×512 int8), biases (10 int32)
    for f in range(N_FEAT):
        dram[(FEAT_BASE>>2)+f]=int(features[f].item())&0xFFFFFFFF
    for c in range(10):
        for f in range(N_FEAT):
            dram[(CLS_W_BASE>>2)+c*N_FEAT+f]=int(cls_w[c,f].item())&0xFFFFFFFF
    for c in range(10):
        dram[(CLS_B_BASE>>2)+c]=cls_bias[c]&0xFFFFFFFF

    dram[LABEL_ADDR>>2]=pred_class&0xFFFFFFFF

    write_hex(out/"dram_init.hex",dram)
    write_hex(out/"expected.hex",list(raw2))
    print(f"Classifier: {N_FEAT} features, 10 classes, pred={pred_class}, scores[0]={scores[0]}, scores[pred]={scores[pred_class]}")

    # ── Firmware ──
    ins=[]; lbls={}
    def emit(*ws):
        for w in ws: ins.append(w)
    def lbl(n): lbls[n]=len(ins)
    def patch_beqz(idx,tgt,rs="t1"): ins[idx]=BEQZ(rs,(lbls[tgt]-idx)*4)
    def wreg(off,val):
        emit(*li_insns("t1",int(val))); emit(SW("t1","s0",off))

    emit(*li_insns("s0",NPU))

    # L0: 4 N-tiles
    for nt in range(n_tiles):
        wa=W_BASE+nt*W_STRIDE; ra=R_BASE+nt*R0_SIZE; ba=B_BASE+nt*TC*4
        wreg(32,wa); wreg(36,A_BASE); wreg(40,ra)
        wreg(0,0); wreg(16,TR); wreg(20,TC); wreg(24,K0)
        wreg(48,0x80); wreg(60,2); wreg(0,0x11)
        lp=f"l0p{nt}"; lbl(lp)
        emit(LW("t1","s0",4)); emit(ANDI("t1","t1",2))
        i=len(ins); emit(0); patch_beqz(i,lp)
        emit(*li_insns("t0",ra)); emit(*li_insns("t2",ba))
        emit(LW("t1","t0",0)); emit(LW("t3","t2",0))
        emit(ADD("t1","t1","t3"))
        emit(SRAI("t2","t1",31)); i2=len(ins); emit(0)
        emit(*li_insns("t1",0)); lbl(f"r{nt}"); patch_beqz(i2,f"r{nt}","t2")
        emit(SW("t1","t0",0))

    # L1
    wreg(0,0); wreg(16,TR); wreg(20,TC); wreg(24,K1)
    wreg(32,L1_W_ADDR); wreg(36,L1_A_ADDR); wreg(40,L1_R_ADDR)
    wreg(48,0x80); wreg(60,2); wreg(0,0x11)
    lbl("l1p")
    emit(LW("t1","s0",4)); emit(ANDI("t1","t1",2))
    i=len(ins); emit(0); patch_beqz(i,"l1p")

    # L2: stage2_0_conv (64→128, K2, 1 N-tile)
    wreg(0,0); wreg(16,TR); wreg(20,TC); wreg(24,K2)
    wreg(32,L2_W_ADDR); wreg(36,L2_A_ADDR); wreg(40,L2_R_ADDR)
    wreg(48,0x80); wreg(60,2); wreg(0,0x11)
    lbl("l2p")
    emit(LW("t1","s0",4)); emit(ANDI("t1","t1",2))
    i=len(ins); emit(0); patch_beqz(i,"l2p")

    # ── MaxPool 2×2 (demo) ──
    emit(*li_insns("t0",L1_R_ADDR))
    emit(LW("t1","t0",0)); emit(LW("t2","t0",4))
    emit(LW("t3","t0",64)); emit(LW("t4","t0",68))
    emit(SUB("t5","t1","t2")); emit(SRAI("t5","t5",31))
    i=len(ins); emit(0); emit(MV("t1","t2")); lbl("cmp1"); patch_beqz(i,"cmp1","t5")
    emit(SUB("t5","t1","t3")); emit(SRAI("t5","t5",31))
    i=len(ins); emit(0); emit(MV("t1","t3")); lbl("cmp2"); patch_beqz(i,"cmp2","t5")
    emit(SUB("t5","t1","t4")); emit(SRAI("t5","t5",31))
    i=len(ins); emit(0); emit(MV("t1","t4")); lbl("cmp3"); patch_beqz(i,"cmp3","t5")

    # ── Linear Classifier: 10-class dot product (proven pattern) ──
    emit(*li_insns("s1",FEAT_BASE))
    emit(*li_insns("s2",CLS_W_BASE))
    emit(*li_insns("s3",CLS_B_BASE))
    emit(*li_insns("s4",SCORE_BASE))
    emit(*li_insns("a0",0))
    emit(*li_insns("a2",N_FEAT))

    for cls_idx in range(10):
        emit(MUL("t0","a0","a2")); emit(SLLI("t0","t0",2)); emit(ADD("t0","s2","t0"))
        emit(*li_insns("a5",0)); emit(*li_insns("a6",0))
        lbl(f"lp{cls_idx}")
        emit(SLLI("t1","a5",2)); emit(ADD("t3","s1","t1")); emit(LW("t2","t3",0))
        emit(ADD("t3","t0","t1")); emit(LW("t4","t3",0))
        emit(MUL("t2","t2","t4")); emit(ADD("a6","a6","t2"))
        emit(ADDI("a5","a5",1)); emit(SUB("t1","a5","a2"))
        i=len(ins); emit(0); ins.append(0); lbl(f"d{cls_idx}")
        patch_beqz(i,f"d{cls_idx}","t1"); joff=(lbls[f"lp{cls_idx}"]-len(ins)+1)*4; ins[-1]=J(joff)
        emit(SLLI("t1","a0",2)); emit(ADD("t1","s3","t1")); emit(LW("t2","t1",0))
        emit(ADD("a6","a6","t2")); emit(SW("a6","s4",0))
        emit(ADDI("s4","s4",4)); emit(ADDI("a0","a0",1))

    # ── Argmax: find class with max score ──
    emit(*li_insns("s4",SCORE_BASE))
    emit(LW("a2","s4",0))               # a2 = best_score = scores[0]
    emit(*li_insns("a3",0))             # a3 = best_class = 0
    emit(*li_insns("a0",1))             # a0 = class index
    emit(*li_insns("a1",10))            # a1 = num classes
    emit(ADDI("s4","s4",4))

    lbl("argmax_loop")
    emit(LW("t1","s4",0))               # t1 = scores[a0]
    emit(SUB("t2","a2","t1"))           # t2 = best - current
    emit(SRAI("t2","t2",31))            # t2 = 0 if best>=cur, -1 if cur>best
    i=len(ins); emit(0)                 # BEQZ: skip if best >= current
    emit(MV("a2","t1"))                 # best_score = current
    emit(MV("a3","a0"))                 # best_class = current
    lbl("argmax_nxt")
    patch_beqz(i,"argmax_nxt","t2")
    emit(ADDI("a0","a0",1))
    emit(ADDI("s4","s4",4))
    emit(SUB("t1","a0","a1"))           # t1 = a0 - 10
    i=len(ins); emit(0)                 # BEQZ placeholder
    ins.append(0)                       # J placeholder
    lbl("argmax_done")
    patch_beqz(i,"argmax_done","t1")
    joff=(lbls["argmax_loop"]-len(ins)+1)*4
    ins[-1]=J(joff)

    # ── Write predicted class to marker ──
    emit(*li_insns("t0",MARKER))
    emit(MV("t1","a3"))
    emit(ADDI("t1","t1",0x100))         # marker = class + 256
    emit(SW("t1","t0",0))
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
        f.write(f'`define VGG_R_ADDR 32\'h{L2_R_ADDR:08x}\n')
        f.write(f'`define VGG_RESULT_COUNT {TR*TC}\n')
        f.write(f'`define VGG_LABEL_ADDR 32\'h{LABEL_ADDR:08x}\n')
        f.write(f'`define VGG_LABEL {pred_class}\n')
        f.write(f'`define VGG_TIMEOUT_CYCLES 50000000\n')
        f.write(f'`define VGG_DRAM_WORDS {DRAM}\n')

    print(f"\nGenerated: {out}, {fw} words")
    print(f"  Firmware: L0(4t)+L1+L2+MaxPool+Linear512→10+Argmax")

if __name__=="__main__": main()
