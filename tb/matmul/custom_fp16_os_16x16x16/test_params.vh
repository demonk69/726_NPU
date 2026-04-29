// Auto-generated: custom_fp16_os_16x16x16 (fp16 OS)
// C = A[16x16] x B[16x16] = C[16x16]
`define NUM_RESULTS 256
`define M_DIM 16
`define N_DIM 16
`define K_DIM 16
`define W_ADDR 32'h00010000
`define A_ADDR 32'h00010300
`define R_ADDR 32'h00010600
`define CTRL   32'h19
`define DRAM_SIZE 17024
`define IS_FP16 1
`define IS_OS   1
