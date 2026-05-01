// Auto-generated: matmul_relu6_int8_ws_3x5x4 (int8 WS)
// C = A[3x5] x B[5x4] = C[3x4]
`define NUM_RESULTS 12
`define M_DIM 3
`define N_DIM 4
`define K_DIM 5
`define W_ADDR 32'h00010000
`define A_ADDR 32'h00010120
`define R_ADDR 32'h00010238
`define BIAS_ADDR 32'h00010368
`define BIAS_EN 1
`define ACT_MODE 2
`define QUANT_EN 0
`define QUANT_CFG 32'h00010000
`define QUANT_SCALE 1
`define QUANT_SHIFT 0
`define QUANT_ROUND 0
`define CTRL   32'ha01
`define DRAM_SIZE 16606
`define IS_FP16 0
`define IS_OS   0
