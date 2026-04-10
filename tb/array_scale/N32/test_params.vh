// Auto-generated for N=32 (COLS=1, K=32), 4 tests
`define NUM_TESTS 4
`define DRAM_SIZE 17121

// Test 0: int8_WS_K32
`define TEST_0
`define T0_W_ADDR   32'h00010000
`define T0_A_ADDR   32'h00010100
`define T0_R_ADDR   32'h00010200
`define T0_M_DIM    1
`define T0_N_DIM    1
`define T0_K_DIM    32
`define T0_CTRL     32'h01
`define T0_EXPECTED 32'hfffff2e5
`define T0_IS_FP16  0
`define T0_IS_OS    0

// Test 1: int8_OS_K32
`define TEST_1
`define T1_W_ADDR   32'h00010300
`define T1_A_ADDR   32'h00010400
`define T1_R_ADDR   32'h00010500
`define T1_M_DIM    1
`define T1_N_DIM    1
`define T1_K_DIM    32
`define T1_CTRL     32'h11
`define T1_EXPECTED 32'h00001808
`define T1_IS_FP16  0
`define T1_IS_OS    1

// Test 2: fp16_WS_K32
`define TEST_2
`define T2_W_ADDR   32'h00010600
`define T2_A_ADDR   32'h00010700
`define T2_R_ADDR   32'h00010800
`define T2_M_DIM    1
`define T2_N_DIM    1
`define T2_K_DIM    32
`define T2_CTRL     32'h09
`define T2_EXPECTED 32'hc0ca0000
`define T2_IS_FP16  1
`define T2_IS_OS    0

// Test 3: fp16_OS_K32
`define TEST_3
`define T3_W_ADDR   32'h00010900
`define T3_A_ADDR   32'h00010a00
`define T3_R_ADDR   32'h00010b00
`define T3_M_DIM    1
`define T3_N_DIM    1
`define T3_K_DIM    32
`define T3_CTRL     32'h19
`define T3_EXPECTED 32'hc0c40000
`define T3_IS_FP16  1
`define T3_IS_OS    1
