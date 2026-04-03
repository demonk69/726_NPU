#!/usr/bin/env python3
"""
gen_classifier_data.py
======================
生成 3 层全连接分类网络（Tiny-FC-Net）的参数和测试数据，
用于 NPU Ping-Pong Buffer 系统仿真。

网络结构（INT8 量化）：
  Input(16)  → FC1(16→8)  → ReLU → FC2(8→4)  → ReLU → FC3(4→4) → Softmax(仅验证)

所有计算用 INT8 权重，激活用 INT8，累加为 INT32。
ReLU 截断：max(0, acc)，结果 >> 8 作为下一层的 INT8 激活（模拟量化）。

输出文件：
  tb/classifier_dram.hex   — DRAM 初始化 hex（每行一个 32-bit 字，小端）
  tb/classifier_golden.txt — 每层期望输出（供 testbench 比较）
  tb/classifier_layout.txt — DRAM 布局说明（各矩阵起始地址）
"""

import numpy as np
import os, struct, textwrap

np.random.seed(42)

# ===================================================================
# 网络配置
# ===================================================================
IN_DIM   = 16   # 输入特征维度
H1_DIM   = 8    # FC1 输出维度
H2_DIM   = 4    # FC2 输出维度
OUT_DIM  = 4    # FC3 输出维度（4 类分类）

# 量化参数：权重 INT8, 激活 INT8, 累加 INT32
# 反量化 shift：累加结果右移 8 bit 作为下一层激活
SHIFT = 8

# ===================================================================
# 生成权重（INT8，使用较大值域保证累加值不被 shift 截断至 0）
# ===================================================================
def rand_int8_matrix(rows, cols, low=-64, high=64):
    """生成 INT8 权重矩阵，限制在 [-64, 64] 确保 INT32 累加不溢出"""
    w = np.random.randint(low, high+1, size=(rows, cols), dtype=np.int32)
    w = np.clip(w, -128, 127).astype(np.int8)
    return w

W1 = rand_int8_matrix(H1_DIM,  IN_DIM)    # (8, 16)
W2 = rand_int8_matrix(H2_DIM,  H1_DIM)    # (4, 8)
W3 = rand_int8_matrix(OUT_DIM, H2_DIM)    # (4, 4)

# ===================================================================
# 生成输入激活（INT8，代表一帧特征）
# ===================================================================
X = np.array([10, 20, 30, 40, 50, 60, 70, 80,
              90,100, 80, 70, 60, 50, 40, 30], dtype=np.int8).reshape(1, IN_DIM)

# ===================================================================
# 前向推理（INT8 → INT32 → ReLU + shift → INT8）
# ===================================================================
# shift = 8：INT8 × INT8 最大累加 = 128 × 128 × K ≈ 2M，
# 右移 8 ≈ 除以 256，K=16 时期望非零结果约 64~255 范围
def fc_relu(x_int8, w_int8, shift=8):
    """
    全连接层 + ReLU：
      acc = x @ W^T   (INT32)
      relu = max(0, acc)
      out = clip(relu >> shift, -128, 127) as INT8  (量化到下一层 INT8 输入)
    """
    acc = x_int8.astype(np.int32) @ w_int8.T.astype(np.int32)   # (1, out_dim)
    relu = np.maximum(0, acc)
    out = (relu >> shift)
    out_clipped = np.clip(out, -128, 127).astype(np.int8)
    return acc, relu, out_clipped

acc1, relu1, A1 = fc_relu(X,  W1, SHIFT)   # A1: (1, 8)
acc2, relu2, A2 = fc_relu(A1, W2, SHIFT)   # A2: (1, 4)
acc3, relu3, A3 = fc_relu(A2, W3, SHIFT)   # A3: (1, 4) — final logits

# 确保各层 activation 非全零（否则网络退化）
print(f"[check] acc1 range: {acc1.min()} ~ {acc1.max()}")
print(f"[check] A1   range: {A1.min()} ~ {A1.max()}")
print(f"[check] acc2 range: {acc2.min()} ~ {acc2.max()}")
print(f"[check] A2   range: {A2.min()} ~ {A2.max()}")
print(f"[check] acc3 range: {acc3.min()} ~ {acc3.max()}")

# Predicted class
pred_class = int(np.argmax(A3))

# ===================================================================
# DRAM 布局
# ===================================================================
# DRAM 基地址 = 0x0000_1000（紧接 4KB SRAM 之后）
# 每个矩阵按 32-bit 字对齐存放
# INT8 元素每 4 个打包进一个 32-bit 字（小端序）

# 地址规划（所有地址相对于 DRAM 起始，单位字节）
# 以 256 字节为区块边界（便于 DMA 地址计算）
DRAM_BASE    = 0x00001000

OFF_W1       = 0x000        # W1 (8×16 INT8 = 128 B)
OFF_W2       = 0x100        # W2 (4×8  INT8 = 32  B)
OFF_W3       = 0x140        # W3 (4×4  INT8 = 16  B)
OFF_INPUT    = 0x200        # X  (1×16 INT8 = 16  B)
OFF_A1       = 0x300        # A1 output (1×8  INT8 = 8 B) — NPU 计算结果
OFF_A2       = 0x380        # A2 output (1×4  INT8 = 4 B)
OFF_A3       = 0x3C0        # A3 final  (1×4  INT32 = 16 B，存 INT32 累加值)

# DRAM 大小：4KB（1024 words × 4）
DRAM_WORDS   = 1024
dram = np.zeros(DRAM_WORDS, dtype=np.uint32)

def write_int8_matrix(dram_arr, byte_off, mat):
    """将 INT8 矩阵写入 dram，每 4 个 INT8 打包成 1 个 uint32（小端）"""
    data = mat.flatten().astype(np.int8)
    # Pad to multiple of 4
    pad_len = (4 - len(data) % 4) % 4
    data = np.concatenate([data, np.zeros(pad_len, dtype=np.int8)])
    words = data.view(np.uint32)
    word_off = byte_off // 4
    dram_arr[word_off:word_off+len(words)] = words
    return len(data)   # bytes written

def write_int32_row(dram_arr, byte_off, vec):
    """将 INT32 向量写入 dram"""
    data = vec.flatten().astype(np.int32)
    word_off = byte_off // 4
    dram_arr[word_off:word_off+len(data)] = data.view(np.uint32)

# 写入权重
write_int8_matrix(dram, OFF_W1, W1)
write_int8_matrix(dram, OFF_W2, W2)
write_int8_matrix(dram, OFF_W3, W3)

# 写入输入激活
write_int8_matrix(dram, OFF_INPUT, X)

# ===================================================================
# NPU 寄存器配置（每层的 DMA 参数）
# ===================================================================
# FC1: W(8×16 INT8), A(1×16 INT8) → R(1×8 INT32 acc)
FC1_M, FC1_N, FC1_K = 1, H1_DIM, IN_DIM
# FC2: W(4×8 INT8), A(1×8 INT8)   → R(1×4 INT32 acc)
FC2_M, FC2_N, FC2_K = 1, H2_DIM, H1_DIM
# FC3: W(4×4 INT8), A(1×4 INT8)   → R(1×4 INT32 acc)
FC3_M, FC3_N, FC3_K = 1, OUT_DIM, H2_DIM

# ===================================================================
# 输出文件路径
# ===================================================================
script_dir = os.path.dirname(os.path.abspath(__file__))
prj_root   = os.path.dirname(script_dir)
tb_dir     = os.path.join(prj_root, "tb")
os.makedirs(tb_dir, exist_ok=True)

# --- 1. DRAM hex 文件 ---
hex_path = os.path.join(tb_dir, "classifier_dram.hex")
with open(hex_path, "w") as f:
    for w in dram:
        f.write(f"{w:08x}\n")
print(f"[OK] DRAM hex -> {hex_path}  ({DRAM_WORDS} words)")

# --- 2. Golden 期望输出 ---
golden_path = os.path.join(tb_dir, "classifier_golden.txt")
with open(golden_path, "w") as f:
    f.write("# Tiny-FC-Net Golden Reference\n")
    f.write("# ======================================\n")
    f.write(f"# Input X (INT8, 1x{IN_DIM}):\n")
    f.write(f"# {X.flatten().tolist()}\n\n")

    f.write("# --- Layer 1: FC(16->8) ---\n")
    f.write(f"# W1 ({H1_DIM}x{IN_DIM}):\n")
    for row in W1:
        f.write(f"#   {row.tolist()}\n")
    f.write(f"# ACC1 (INT32, 1x{H1_DIM}):\n")
    f.write(f"#   {acc1.flatten().tolist()}\n")
    f.write(f"# A1 after ReLU+shift (INT8, 1x{H1_DIM}):\n")
    f.write(f"#   {A1.flatten().tolist()}\n\n")

    f.write("# --- Layer 2: FC(8->4) ---\n")
    f.write(f"# W2 ({H2_DIM}x{H1_DIM}):\n")
    for row in W2:
        f.write(f"#   {row.tolist()}\n")
    f.write(f"# ACC2 (INT32, 1x{H2_DIM}):\n")
    f.write(f"#   {acc2.flatten().tolist()}\n")
    f.write(f"# A2 after ReLU+shift (INT8, 1x{H2_DIM}):\n")
    f.write(f"#   {A2.flatten().tolist()}\n\n")

    f.write("# --- Layer 3: FC(4->4) ---\n")
    f.write(f"# W3 ({OUT_DIM}x{H2_DIM}):\n")
    for row in W3:
        f.write(f"#   {row.tolist()}\n")
    f.write(f"# ACC3 final logits (INT32, 1x{OUT_DIM}):\n")
    f.write(f"#   {acc3.flatten().tolist()}\n")
    f.write(f"# A3 after ReLU+shift (INT8, 1x{OUT_DIM}):\n")
    f.write(f"#   {A3.flatten().tolist()}\n\n")

    f.write(f"# PREDICTED CLASS: {pred_class}\n")

    f.write("\n# ======================================\n")
    f.write("# Verilog expected values (hex, INT32 per accumulator output):\n")
    f.write("# These are the raw INT32 accumulators before shift/clip.\n")
    f.write("# Testbench compares NPU DMA writeback against these.\n\n")
    f.write("# [Layer 1 - expected ACC1 at DRAM result addr]\n")
    for i, v in enumerate(acc1.flatten()):
        uv = int(np.int32(v)) & 0xFFFFFFFF
        f.write(f"ACC1[{i}] = {uv:08x}  # decimal: {int(np.int32(v))}\n")
    f.write("\n# [Layer 2 - expected ACC2]\n")
    for i, v in enumerate(acc2.flatten()):
        uv = int(np.int32(v)) & 0xFFFFFFFF
        f.write(f"ACC2[{i}] = {uv:08x}  # decimal: {int(np.int32(v))}\n")
    f.write("\n# [Layer 3 - expected ACC3]\n")
    for i, v in enumerate(acc3.flatten()):
        uv = int(np.int32(v)) & 0xFFFFFFFF
        f.write(f"ACC3[{i}] = {uv:08x}  # decimal: {int(np.int32(v))}\n")
    f.write(f"\nPRED_CLASS = {pred_class}\n")

print(f"[OK] Golden reference -> {golden_path}")

# --- 3. Layout 说明 ---
layout_path = os.path.join(tb_dir, "classifier_layout.txt")
with open(layout_path, "w") as f:
    f.write("# Tiny-FC-Net DRAM Layout\n")
    f.write("# ===================================================\n")
    f.write(f"# DRAM base address: 0x{DRAM_BASE:08X}\n\n")
    f.write("# --- Weights ---\n")
    f.write(f"W1  ({H1_DIM}x{IN_DIM} INT8):   addr=0x{DRAM_BASE+OFF_W1:08X}  (W_ADDR reg)\n")
    f.write(f"W2  ({H2_DIM}x{H1_DIM}  INT8):    addr=0x{DRAM_BASE+OFF_W2:08X}  (W_ADDR reg)\n")
    f.write(f"W3  ({OUT_DIM}x{H2_DIM}   INT8):    addr=0x{DRAM_BASE+OFF_W3:08X}  (W_ADDR reg)\n")
    f.write("# --- Activations ---\n")
    f.write(f"X   (1x{IN_DIM}  INT8):  addr=0x{DRAM_BASE+OFF_INPUT:08X}  (A_ADDR reg)\n")
    f.write(f"A1  (1x{H1_DIM}   INT8):   addr=0x{DRAM_BASE+OFF_A1:08X}  (A_ADDR reg for L2)\n")
    f.write(f"A2  (1x{H2_DIM}    INT8):   addr=0x{DRAM_BASE+OFF_A2:08X}  (A_ADDR reg for L3)\n")
    f.write("# --- Results ---\n")
    f.write(f"R1  (1x{H1_DIM} INT32): addr=0x{DRAM_BASE+OFF_A1:08X}  (R_ADDR reg, overwritten by ReLU+shift)\n")
    f.write(f"R2  (1x{H2_DIM}  INT32): addr=0x{DRAM_BASE+OFF_A2:08X}\n")
    f.write(f"R3  (1x{OUT_DIM}  INT32): addr=0x{DRAM_BASE+OFF_A3:08X}  (final logits)\n")
    f.write("\n# --- NPU Register Config per Layer ---\n")
    f.write(f"FC1: M={FC1_M} N={FC1_N} K={FC1_K}  W_ADDR=0x{DRAM_BASE+OFF_W1:08X}  A_ADDR=0x{DRAM_BASE+OFF_INPUT:08X}  R_ADDR=0x{DRAM_BASE+OFF_A1:08X}\n")
    f.write(f"FC2: M={FC2_M} N={FC2_N} K={FC2_K}   W_ADDR=0x{DRAM_BASE+OFF_W2:08X}  A_ADDR=0x{DRAM_BASE+OFF_A1:08X}  R_ADDR=0x{DRAM_BASE+OFF_A2:08X}\n")
    f.write(f"FC3: M={FC3_M} N={FC3_N} K={FC3_K}   W_ADDR=0x{DRAM_BASE+OFF_W3:08X}  A_ADDR=0x{DRAM_BASE+OFF_A2:08X}  R_ADDR=0x{DRAM_BASE+OFF_A3:08X}\n")

print(f"[OK] DRAM layout   -> {layout_path}")

# --- 4. 给 Verilog testbench 用的 include 文件（硬编码期望值） ---
vh_path = os.path.join(tb_dir, "classifier_expected.vh")
with open(vh_path, "w") as f:
    f.write("// AUTO-GENERATED by gen_classifier_data.py — DO NOT EDIT\n")
    f.write("// Expected INT32 accumulator outputs for Tiny-FC-Net\n\n")

    # DRAM offsets
    f.write(f"localparam DRAM_BASE   = 32'h{DRAM_BASE:08X};\n")
    f.write(f"localparam OFF_W1      = 12'h{OFF_W1:03X};\n")
    f.write(f"localparam OFF_W2      = 12'h{OFF_W2:03X};\n")
    f.write(f"localparam OFF_W3      = 12'h{OFF_W3:03X};\n")
    f.write(f"localparam OFF_INPUT   = 12'h{OFF_INPUT:03X};\n")
    f.write(f"localparam OFF_A1      = 12'h{OFF_A1:03X};\n")
    f.write(f"localparam OFF_A2      = 12'h{OFF_A2:03X};\n")
    f.write(f"localparam OFF_A3      = 12'h{OFF_A3:03X};\n\n")

    # Matrix dims
    f.write(f"localparam FC1_M={FC1_M}, FC1_N={FC1_N}, FC1_K={FC1_K};\n")
    f.write(f"localparam FC2_M={FC2_M}, FC2_N={FC2_N}, FC2_K={FC2_K};\n")
    f.write(f"localparam FC3_M={FC3_M}, FC3_N={FC3_N}, FC3_K={FC3_K};\n\n")

    # Layer 1 expected INT32 accumulator
    f.write("// Layer 1 expected accumulators (INT32)\n")
    for i, v in enumerate(acc1.flatten()):
        uv = int(np.int32(v)) & 0xFFFFFFFF
        f.write(f"localparam EXP_ACC1_{i} = 32'h{uv:08X};  // {int(np.int32(v))}\n")

    f.write("\n// Layer 1 expected A1 (INT8 after ReLU+shift, packed to 32-bit)\n")
    a1_flat = A1.flatten().astype(np.int8)
    a1_pad = np.zeros(4, dtype=np.int8)
    a1_all = np.concatenate([a1_flat, a1_pad[:((4-len(a1_flat)%4)%4)]])
    a1_words = a1_all.view(np.uint32)
    for i, w in enumerate(a1_words):
        f.write(f"localparam EXP_A1_W{i} = 32'h{w:08X};\n")

    # Layer 2 expected INT32 accumulator
    f.write("\n// Layer 2 expected accumulators (INT32)\n")
    for i, v in enumerate(acc2.flatten()):
        uv = int(np.int32(v)) & 0xFFFFFFFF
        f.write(f"localparam EXP_ACC2_{i} = 32'h{uv:08X};  // {int(np.int32(v))}\n")

    f.write("\n// Layer 2 expected A2 (INT8 after ReLU+shift)\n")
    a2_flat = A2.flatten().astype(np.int8)
    a2_words = a2_flat.view(np.uint32)
    for i, w in enumerate(a2_words):
        f.write(f"localparam EXP_A2_W{i} = 32'h{w:08X};\n")

    # Layer 3 expected INT32 accumulator (final logits)
    f.write("\n// Layer 3 expected final logits (INT32)\n")
    for i, v in enumerate(acc3.flatten()):
        uv = int(np.int32(v)) & 0xFFFFFFFF
        f.write(f"localparam EXP_ACC3_{i} = 32'h{uv:08X};  // {int(np.int32(v))}\n")

    f.write(f"\n// Predicted class\n")
    f.write(f"localparam PRED_CLASS = {pred_class};\n")

print(f"[OK] Verilog params -> {vh_path}")

# ===================================================================
# Summary
# ===================================================================
print("\n" + "="*55)
print("  Tiny-FC-Net Forward Pass Summary")
print("="*55)
print(f"  Input X:  {X.flatten().tolist()}")
print(f"  A1 (L1):  {A1.flatten().tolist()}")
print(f"  A2 (L2):  {A2.flatten().tolist()}")
print(f"  A3 (L3):  {A3.flatten().tolist()}")
print(f"  Predicted class: {pred_class}")
print("="*55)
print(f"\nGenerated files:\n  {hex_path}\n  {golden_path}\n  {layout_path}\n  {vh_path}")
