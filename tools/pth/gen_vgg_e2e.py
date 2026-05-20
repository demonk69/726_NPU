#!/usr/bin/env python3
"""RepOpt VGG e2e: 9-layer full-tile NPU chain + CPU classifier.

NPU: 1024 tiles (all M-tiles × N-tiles × 9 layers), table-driven scheduler.
CPU Classifier: 512 features × 10 classes dot product + argmax in firmware.
Testbench: compares predicted class vs PyTorch golden.
Features from Python golden full forward pass (extract_real_features).
Layer transitions (bias+ReLU+requant+MaxPool) handled by Python golden.
Input: real CIFAR-10 image or arbitrary PNG/JPG.
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

SIMD=4; TR=16; TC=16; PPB=64; DRAM=2*1024*1024
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

    # ── NPU pipeline: build all tiles for all 9 conv layers ──
    kt=(PPB*4)//max(TR,TC)
    spec_by_name={l["name"]:l for l in spec["layers"]}
    conv_layers=[l for l in plan["layers"] if l["op"]=="conv2d"]
    bias_dir=Path(args.plan).resolve().parent

    def merge_spec(cl_p):
        nm=cl_p["name"]; cl_s=spec_by_name.get(nm,{})
        for k in ("stride","padding","dilation"): cl_p[k]=cl_s.get(k,cl_p.get(k,1))
        return cl_p

    def get_bias(cl_p):
        N=int(cl_p["registers"]["N_DIM"]); bk=cl_p.get("bias_key",""); bias=[0]*N
        if bk and bk in sd:
            for i in range(min(N,sd[bk].shape[0])): bias[i]=int(sd[bk][i].item())
        return bias

    # DRAM layout
    W_BASE=0x1000; A_BASE=0x40000; R_BASE=0x80000; B_BASE=0x2000
    FEAT_BASE=0x600000; CLS_W_BASE=0x602000; CLS_B_BASE=0x610000
    SCORE_BASE=0x611000; MARKER=0x612000; LABEL_ADDR=0x613000
    TILE_TABLE_BASE=0x3000
    dram=[0]*DRAM

    tile_table=[]; next_w=W_BASE; next_a=A_BASE; next_r=R_BASE; next_b=B_BASE
    current=x_q

    for li,cl_r in enumerate(conv_layers):
        cl_p=merge_spec(cl_r.copy())
        M=int(cl_p["registers"]["M_DIM"]); N=int(cl_p["registers"]["N_DIM"])
        K=int(cl_p["registers"]["K_DIM"]); qw=sd[cl_p["weight_key"]]
        bias=get_bias(cl_p)
        n_tiles=(N+TC-1)//TC; m_tiles=(M+TR-1)//TR

        # Store all W tiles and bias
        w_addrs=[]
        for ni in range(n_tiles):
            nb_act=min(TC,N-ni*TC)
            _,wt=build_tile_matrices(current,qw,cl_p,0,ni*TC,TR,nb_act)
            wp=pack_tile_stream(wt,nb_act,K,kt)
            for i,w in enumerate(wp): dram[(next_w>>2)+i]=w
            w_addrs.append(next_w); next_w+=len(wp)*4+0x100

        bias_addr=next_b
        for i,b in enumerate(bias): dram[(next_b>>2)+i]=b&0xFFFFFFFF
        next_b+=len(bias)*4+0x100

        # Build all tile entries
        for ni in range(n_tiles):
            nb_act=min(TC,N-ni*TC)
            for mi in range(m_tiles):
                mb_act=min(TR,M-mi*TR)
                at,_=build_tile_matrices(current,qw,cl_p,mi*TR,ni*TC,mb_act,nb_act)
                ap=pack_tile_stream(at,mb_act,K,kt)
                for i,w in enumerate(ap): dram[(next_a>>2)+i]=w
                # 9-word entry: A,R,W,B,M,N,K,Rcount,ctrl
                tile_table.append([next_a,next_r,w_addrs[ni],bias_addr,
                                   mb_act,nb_act,K,mb_act*nb_act,0x80])
                next_a+=len(ap)*4+0x100; next_r+=mb_act*nb_act*4+8

        # Full conv for next layer's ifm
        current=requant_qint8(
            conv2d_acc_npu(current,qw,
                load_int32_hex(bias_dir/cl_p["assets"]["bias_int32_hex"]),cl_p),
            cl_p["cpu_requant_after_npu"]["multipliers"],
            cl_p["cpu_requant_after_npu"]["output_zero_point"])

        # MaxPool between stages
        if cl_p["name"] in ("stage1_1_conv","stage2_1_conv","stage3_2_conv"):
            pool_layer=[l for l in plan["layers"] if l["op"]=="maxpool2d" and
                       l["name"].startswith(cl_p["name"].split("_")[0])][0]
            current=maxpool2d_cpu(current,pool_layer)

        next_a=A_BASE+0x1000  # reset A space per layer

    total_tiles=len(tile_table)
    for i,e in enumerate(tile_table):
        for j,v in enumerate(e):
            dram[(TILE_TABLE_BASE>>2)+i*9+j]=v&0xFFFFFFFF

    # Classifier data
    for f in range(N_FEAT):
        dram[(FEAT_BASE>>2)+f]=int(features[f].item())&0xFFFFFFFF
    for c in range(10):
        for f in range(N_FEAT):
            dram[(CLS_W_BASE>>2)+c*N_FEAT+f]=int(cls_w[c,f].item())&0xFFFFFFFF
    for c in range(10):
        dram[(CLS_B_BASE>>2)+c]=cls_bias[c]&0xFFFFFFFF

    dram[LABEL_ADDR>>2]=pred_class&0xFFFFFFFF

    FINAL_R_ADDR=tile_table[-1][1]
    FINAL_R_COUNT=tile_table[-1][7]
    expected_last=[dram[(FINAL_R_ADDR>>2)+i] for i in range(FINAL_R_COUNT)]

    write_hex(out/"dram_init.hex",dram)
    write_hex(out/"expected.hex",expected_last)

    print(f"NPU: {total_tiles} tiles ({len(conv_layers)} layers), table at 0x{TILE_TABLE_BASE:05x}")
    print(f"Classifier: {N_FEAT} features, 10 classes, pred={pred_class}")

    # ── Firmware ──
    ins=[]; lbls={}
    def emit(*ws):
        for w in ws: ins.append(w)
    def lbl(n): lbls[n]=len(ins)
    def patch_beqz(idx,tgt,rs="t1"): ins[idx]=BEQZ(rs,(lbls[tgt]-idx)*4)
    def wreg(off,val):
        emit(*li_insns("t1",int(val))); emit(SW("t1","s0",off))

    emit(*li_insns("s0",NPU))

    # ── Tile scheduler: table-driven, all tiles ──
    emit(*li_insns("s1",TILE_TABLE_BASE))
    emit(*li_insns("a0",total_tiles))

    lbl("tile_loop")
    # 9-word entry: +0=A +4=R +8=W +12=B +16=M +20=N +24=K +28=Rcount +32=ctrl
    emit(LW("t1","s1",8)); emit(SW("t1","s0",32))   # W
    emit(LW("t1","s1",0)); emit(SW("t1","s0",36))   # A
    emit(LW("t1","s1",4)); emit(SW("t1","s0",40))   # R
    emit(SW("zero","s0",0))
    emit(LW("t1","s1",16)); emit(SW("t1","s0",16))  # M
    emit(LW("t1","s1",20)); emit(SW("t1","s0",20))  # N
    emit(LW("t1","s1",24)); emit(SW("t1","s0",24))  # K
    emit(LW("t1","s1",32)); emit(SW("t1","s0",48))  # ARR_CFG
    emit(ADDI("t1","zero",2)); emit(SW("t1","s0",60))
    emit(ADDI("t1","zero",0x11)); emit(SW("t1","s0",0))

    lbl("poll")
    emit(LW("t1","s0",4)); emit(ANDI("t1","t1",2))
    i=len(ins); emit(0); patch_beqz(i,"poll")

    emit(ADDI("s1","s1",36))
    emit(ADDI("a0","a0",-1))
    emit(SUB("t1","a0","zero"))
    i=len(ins); emit(0); ins.append(0); lbl("tile_done")
    patch_beqz(i,"tile_done","t1")
    joff=(lbls["tile_loop"]-len(ins)+1)*4; ins[-1]=J(joff)

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
    emit(LW("a2","s4",0))
    emit(*li_insns("a3",0))
    emit(*li_insns("a0",1))
    emit(*li_insns("a1",10))
    emit(ADDI("s4","s4",4))

    lbl("argmax_loop")
    emit(LW("t1","s4",0))
    emit(SUB("t2","a2","t1"))
    emit(SRAI("t2","t2",31))
    i=len(ins); emit(0)
    emit(MV("a2","t1"))
    emit(MV("a3","a0"))
    lbl("argmax_nxt")
    patch_beqz(i,"argmax_nxt","t2")
    emit(ADDI("a0","a0",1))
    emit(ADDI("s4","s4",4))
    emit(SUB("t1","a0","a1"))
    i=len(ins); emit(0); ins.append(0); lbl("argmax_done")
    patch_beqz(i,"argmax_done","t1")
    joff=(lbls["argmax_loop"]-len(ins)+1)*4; ins[-1]=J(joff)

    # ── Write predicted class to marker ──
    emit(*li_insns("t0",MARKER))
    emit(MV("t1","a3"))
    emit(ADDI("t1","t1",0x100))
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
        f.write(f'`define VGG_R_ADDR 32\'h{FINAL_R_ADDR:08x}\n')
        f.write(f'`define VGG_RESULT_COUNT {FINAL_R_COUNT}\n')
        f.write(f'`define VGG_LABEL_ADDR 32\'h{LABEL_ADDR:08x}\n')
        f.write(f'`define VGG_LABEL {pred_class}\n')
        f.write(f'`define VGG_TIMEOUT_CYCLES 50000000\n')
        f.write(f'`define VGG_DRAM_WORDS {DRAM}\n')

    print(f"\nGenerated: {out}, {fw} words")
    print(f"  Firmware: {total_tiles}-tile scheduler + Linear512->10 + Argmax")

if __name__=="__main__": main()
