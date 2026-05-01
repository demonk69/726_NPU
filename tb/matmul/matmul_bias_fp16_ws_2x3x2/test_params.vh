// Auto-generated: matmul_bias_fp16_ws_2x3x2 (fp16 WS)
// C = A[2x3] x B[3x2] = C[2x2]
`define NUM_RESULTS 4
`define M_DIM 2
`define N_DIM 2
`define K_DIM 3
`define W_ADDR 32'h00010000
`define A_ADDR 32'h00010110
`define R_ADDR 32'h00010220
`define BIAS_ADDR 32'h00010330
`define BIAS_EN 1
`define ACT_MODE 0
`define QUANT_EN 0
`define QUANT_CFG 32'h00010000
`define QUANT_SCALE 1
`define QUANT_SHIFT 0
`define QUANT_ROUND 0
`define CTRL   32'h209
`define DRAM_SIZE 16590
`define IS_FP16 1
`define IS_OS   0
