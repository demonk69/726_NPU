// Auto-generated: conv2d_relu_im2col_int8_os_smoke (int8 OS)
// T6.1 Conv2D uses DRAM pre-expanded im2col.
// Conv2D: B=1 IFM=5x5 Cin=2 KHxKW=3x3 Cout=3 OHxOW=5x5
// GEMM: A_im2col[25x18] x W_col[18x3]
`define NUM_RESULTS 75
`define M_DIM 25
`define N_DIM 3
`define K_DIM 18
`define W_ADDR 32'h00010000
`define A_ADDR 32'h0001013c
`define R_ADDR 32'h00010430
`define BIAS_ADDR 32'h0001065c
`define BIAS_EN 1
`define ACT_MODE 1
`define CTRL   32'h611
`define DRAM_SIZE 16794
`define IS_FP16 0
`define IS_OS   1
`define CONV_BATCH 1
`define CONV_IH 5
`define CONV_IW 5
`define CONV_CIN 2
`define CONV_COUT 3
`define CONV_KH 3
`define CONV_KW 3
`define CONV_OH 5
`define CONV_OW 5
`define CONV_IM2COL 0
`define CONV_IFM_SHAPE 32'h00050005
`define CONV_CHANNELS 32'h00010002
`define CONV_KERNEL 32'h00030003
`define CONV_OUT_SHAPE 32'h00050005
`define CONV_STRIDE_PAD 32'h01010101
`define CONV_DILATION 32'h00000101
