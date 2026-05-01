// Auto-generated: matmul_relu_int8_os_nobias_smoke (int8 OS)
// C = A[2x3] x B[3x2] = C[2x2]
`define NUM_RESULTS 4
`define M_DIM 2
`define N_DIM 2
`define K_DIM 3
`define W_ADDR 32'h00010000
`define A_ADDR 32'h00010108
`define R_ADDR 32'h00010210
`define BIAS_ADDR 32'h00000000
`define BIAS_EN 0
`define ACT_MODE 1
`define CTRL   32'h411
`define DRAM_SIZE 16520
`define IS_FP16 0
`define IS_OS   1
