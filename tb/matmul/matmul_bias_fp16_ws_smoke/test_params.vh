// Auto-generated: matmul_bias_fp16_ws_smoke (fp16 WS)
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
`define CTRL   32'h209
`define DRAM_SIZE 16590
`define IS_FP16 1
`define IS_OS   0
