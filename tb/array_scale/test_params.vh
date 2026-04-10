`define NUM_TESTS 28

// Test 0: int8_WS_N4_K4
`define T0_ID       "int8_WS_N4_K4"
`define T0_N        4
`define T0_IS_FP16  0
`define T0_IS_OS    0
`define T0_M_DIM    1
`define T0_N_DIM    1
`define T0_K_DIM    4
`define T0_W_ADDR   32'h00010000
`define T0_A_ADDR   32'h00010100
`define T0_R_ADDR   32'h00010200
`define T0_CTRL     32'h00000001
`define T0_EXPECTED 32'h000006fe

// Test 1: int8_OS_N4_K4
`define T1_ID       "int8_OS_N4_K4"
`define T1_N        4
`define T1_IS_FP16  0
`define T1_IS_OS    1
`define T1_M_DIM    1
`define T1_N_DIM    1
`define T1_K_DIM    4
`define T1_W_ADDR   32'h00010300
`define T1_A_ADDR   32'h00010400
`define T1_R_ADDR   32'h00010500
`define T1_CTRL     32'h00000011
`define T1_EXPECTED 32'hffffffc0

// Test 2: fp16_WS_N4_K4
`define T2_ID       "fp16_WS_N4_K4"
`define T2_N        4
`define T2_IS_FP16  1
`define T2_IS_OS    0
`define T2_M_DIM    1
`define T2_N_DIM    1
`define T2_K_DIM    4
`define T2_W_ADDR   32'h00010600
`define T2_A_ADDR   32'h00010700
`define T2_R_ADDR   32'h00010800
`define T2_CTRL     32'h00000005
`define T2_EXPECTED 32'hbfecd400

// Test 3: fp16_OS_N4_K4
`define T3_ID       "fp16_OS_N4_K4"
`define T3_N        4
`define T3_IS_FP16  1
`define T3_IS_OS    1
`define T3_M_DIM    1
`define T3_N_DIM    1
`define T3_K_DIM    4
`define T3_W_ADDR   32'h00010900
`define T3_A_ADDR   32'h00010a00
`define T3_R_ADDR   32'h00010b00
`define T3_CTRL     32'h00000015
`define T3_EXPECTED 32'h4019a000

// Test 4: int8_WS_N8_K8
`define T4_ID       "int8_WS_N8_K8"
`define T4_N        8
`define T4_IS_FP16  0
`define T4_IS_OS    0
`define T4_M_DIM    1
`define T4_N_DIM    1
`define T4_K_DIM    8
`define T4_W_ADDR   32'h00010c00
`define T4_A_ADDR   32'h00010d00
`define T4_R_ADDR   32'h00010e00
`define T4_CTRL     32'h00000001
`define T4_EXPECTED 32'hffffff03

// Test 5: int8_WS_N8_K4
`define T5_ID       "int8_WS_N8_K4"
`define T5_N        8
`define T5_IS_FP16  0
`define T5_IS_OS    0
`define T5_M_DIM    1
`define T5_N_DIM    1
`define T5_K_DIM    4
`define T5_W_ADDR   32'h00010f00
`define T5_A_ADDR   32'h00011000
`define T5_R_ADDR   32'h00011100
`define T5_CTRL     32'h00000001
`define T5_EXPECTED 32'h00000f92

// Test 6: int8_OS_N8_K8
`define T6_ID       "int8_OS_N8_K8"
`define T6_N        8
`define T6_IS_FP16  0
`define T6_IS_OS    1
`define T6_M_DIM    1
`define T6_N_DIM    1
`define T6_K_DIM    8
`define T6_W_ADDR   32'h00011200
`define T6_A_ADDR   32'h00011300
`define T6_R_ADDR   32'h00011400
`define T6_CTRL     32'h00000011
`define T6_EXPECTED 32'h00001d1f

// Test 7: int8_OS_N8_K4
`define T7_ID       "int8_OS_N8_K4"
`define T7_N        8
`define T7_IS_FP16  0
`define T7_IS_OS    1
`define T7_M_DIM    1
`define T7_N_DIM    1
`define T7_K_DIM    4
`define T7_W_ADDR   32'h00011500
`define T7_A_ADDR   32'h00011600
`define T7_R_ADDR   32'h00011700
`define T7_CTRL     32'h00000011
`define T7_EXPECTED 32'hffffee7d

// Test 8: fp16_WS_N8_K8
`define T8_ID       "fp16_WS_N8_K8"
`define T8_N        8
`define T8_IS_FP16  1
`define T8_IS_OS    0
`define T8_M_DIM    1
`define T8_N_DIM    1
`define T8_K_DIM    8
`define T8_W_ADDR   32'h00011800
`define T8_A_ADDR   32'h00011900
`define T8_R_ADDR   32'h00011a00
`define T8_CTRL     32'h00000005
`define T8_EXPECTED 32'h3e198000

// Test 9: fp16_WS_N8_K4
`define T9_ID       "fp16_WS_N8_K4"
`define T9_N        8
`define T9_IS_FP16  1
`define T9_IS_OS    0
`define T9_M_DIM    1
`define T9_N_DIM    1
`define T9_K_DIM    4
`define T9_W_ADDR   32'h00011b00
`define T9_A_ADDR   32'h00011c00
`define T9_R_ADDR   32'h00011d00
`define T9_CTRL     32'h00000005
`define T9_EXPECTED 32'hbee66800

// Test 10: fp16_OS_N8_K8
`define T10_ID       "fp16_OS_N8_K8"
`define T10_N        8
`define T10_IS_FP16  1
`define T10_IS_OS    1
`define T10_M_DIM    1
`define T10_N_DIM    1
`define T10_K_DIM    8
`define T10_W_ADDR   32'h00011e00
`define T10_A_ADDR   32'h00011f00
`define T10_R_ADDR   32'h00012000
`define T10_CTRL     32'h00000015
`define T10_EXPECTED 32'hc0499c00

// Test 11: fp16_OS_N8_K4
`define T11_ID       "fp16_OS_N8_K4"
`define T11_N        8
`define T11_IS_FP16  1
`define T11_IS_OS    1
`define T11_M_DIM    1
`define T11_N_DIM    1
`define T11_K_DIM    4
`define T11_W_ADDR   32'h00012100
`define T11_A_ADDR   32'h00012200
`define T11_R_ADDR   32'h00012300
`define T11_CTRL     32'h00000015
`define T11_EXPECTED 32'h400cc000

// Test 12: int8_WS_N16_K16
`define T12_ID       "int8_WS_N16_K16"
`define T12_N        16
`define T12_IS_FP16  0
`define T12_IS_OS    0
`define T12_M_DIM    1
`define T12_N_DIM    1
`define T12_K_DIM    16
`define T12_W_ADDR   32'h00012400
`define T12_A_ADDR   32'h00012500
`define T12_R_ADDR   32'h00012600
`define T12_CTRL     32'h00000001
`define T12_EXPECTED 32'hffffe642

// Test 13: int8_WS_N16_K4
`define T13_ID       "int8_WS_N16_K4"
`define T13_N        16
`define T13_IS_FP16  0
`define T13_IS_OS    0
`define T13_M_DIM    1
`define T13_N_DIM    1
`define T13_K_DIM    4
`define T13_W_ADDR   32'h00012700
`define T13_A_ADDR   32'h00012800
`define T13_R_ADDR   32'h00012900
`define T13_CTRL     32'h00000001
`define T13_EXPECTED 32'h00000486

// Test 14: int8_OS_N16_K16
`define T14_ID       "int8_OS_N16_K16"
`define T14_N        16
`define T14_IS_FP16  0
`define T14_IS_OS    1
`define T14_M_DIM    1
`define T14_N_DIM    1
`define T14_K_DIM    16
`define T14_W_ADDR   32'h00012a00
`define T14_A_ADDR   32'h00012b00
`define T14_R_ADDR   32'h00012c00
`define T14_CTRL     32'h00000011
`define T14_EXPECTED 32'h00001e5d

// Test 15: int8_OS_N16_K4
`define T15_ID       "int8_OS_N16_K4"
`define T15_N        16
`define T15_IS_FP16  0
`define T15_IS_OS    1
`define T15_M_DIM    1
`define T15_N_DIM    1
`define T15_K_DIM    4
`define T15_W_ADDR   32'h00012d00
`define T15_A_ADDR   32'h00012e00
`define T15_R_ADDR   32'h00012f00
`define T15_CTRL     32'h00000011
`define T15_EXPECTED 32'hffffff5f

// Test 16: fp16_WS_N16_K16
`define T16_ID       "fp16_WS_N16_K16"
`define T16_N        16
`define T16_IS_FP16  1
`define T16_IS_OS    0
`define T16_M_DIM    1
`define T16_N_DIM    1
`define T16_K_DIM    16
`define T16_W_ADDR   32'h00013000
`define T16_A_ADDR   32'h00013100
`define T16_R_ADDR   32'h00013200
`define T16_CTRL     32'h00000005
`define T16_EXPECTED 32'h406c34a4

// Test 17: fp16_WS_N16_K4
`define T17_ID       "fp16_WS_N16_K4"
`define T17_N        16
`define T17_IS_FP16  1
`define T17_IS_OS    0
`define T17_M_DIM    1
`define T17_N_DIM    1
`define T17_K_DIM    4
`define T17_W_ADDR   32'h00013300
`define T17_A_ADDR   32'h00013400
`define T17_R_ADDR   32'h00013500
`define T17_CTRL     32'h00000005
`define T17_EXPECTED 32'hbfdea748

// Test 18: fp16_OS_N16_K16
`define T18_ID       "fp16_OS_N16_K16"
`define T18_N        16
`define T18_IS_FP16  1
`define T18_IS_OS    1
`define T18_M_DIM    1
`define T18_N_DIM    1
`define T18_K_DIM    16
`define T18_W_ADDR   32'h00013600
`define T18_A_ADDR   32'h00013700
`define T18_R_ADDR   32'h00013800
`define T18_CTRL     32'h00000015
`define T18_EXPECTED 32'hc090a15c

// Test 19: fp16_OS_N16_K4
`define T19_ID       "fp16_OS_N16_K4"
`define T19_N        16
`define T19_IS_FP16  1
`define T19_IS_OS    1
`define T19_M_DIM    1
`define T19_N_DIM    1
`define T19_K_DIM    4
`define T19_W_ADDR   32'h00013900
`define T19_A_ADDR   32'h00013a00
`define T19_R_ADDR   32'h00013b00
`define T19_CTRL     32'h00000015
`define T19_EXPECTED 32'hbe194000

// Test 20: int8_WS_N32_K32
`define T20_ID       "int8_WS_N32_K32"
`define T20_N        32
`define T20_IS_FP16  0
`define T20_IS_OS    0
`define T20_M_DIM    1
`define T20_N_DIM    1
`define T20_K_DIM    32
`define T20_W_ADDR   32'h00013c00
`define T20_A_ADDR   32'h00013d00
`define T20_R_ADDR   32'h00013e00
`define T20_CTRL     32'h00000001
`define T20_EXPECTED 32'h000012e4

// Test 21: int8_WS_N32_K4
`define T21_ID       "int8_WS_N32_K4"
`define T21_N        32
`define T21_IS_FP16  0
`define T21_IS_OS    0
`define T21_M_DIM    1
`define T21_N_DIM    1
`define T21_K_DIM    4
`define T21_W_ADDR   32'h00013f00
`define T21_A_ADDR   32'h00014000
`define T21_R_ADDR   32'h00014100
`define T21_CTRL     32'h00000001
`define T21_EXPECTED 32'h000004b7

// Test 22: int8_OS_N32_K32
`define T22_ID       "int8_OS_N32_K32"
`define T22_N        32
`define T22_IS_FP16  0
`define T22_IS_OS    1
`define T22_M_DIM    1
`define T22_N_DIM    1
`define T22_K_DIM    32
`define T22_W_ADDR   32'h00014200
`define T22_A_ADDR   32'h00014300
`define T22_R_ADDR   32'h00014400
`define T22_CTRL     32'h00000011
`define T22_EXPECTED 32'h00003156

// Test 23: int8_OS_N32_K4
`define T23_ID       "int8_OS_N32_K4"
`define T23_N        32
`define T23_IS_FP16  0
`define T23_IS_OS    1
`define T23_M_DIM    1
`define T23_N_DIM    1
`define T23_K_DIM    4
`define T23_W_ADDR   32'h00014500
`define T23_A_ADDR   32'h00014600
`define T23_R_ADDR   32'h00014700
`define T23_CTRL     32'h00000011
`define T23_EXPECTED 32'h00000af0

// Test 24: fp16_WS_N32_K32
`define T24_ID       "fp16_WS_N32_K32"
`define T24_N        32
`define T24_IS_FP16  1
`define T24_IS_OS    0
`define T24_M_DIM    1
`define T24_N_DIM    1
`define T24_K_DIM    32
`define T24_W_ADDR   32'h00014800
`define T24_A_ADDR   32'h00014900
`define T24_R_ADDR   32'h00014a00
`define T24_CTRL     32'h00000005
`define T24_EXPECTED 32'h4049083d

// Test 25: fp16_WS_N32_K4
`define T25_ID       "fp16_WS_N32_K4"
`define T25_N        32
`define T25_IS_FP16  1
`define T25_IS_OS    0
`define T25_M_DIM    1
`define T25_N_DIM    1
`define T25_K_DIM    4
`define T25_W_ADDR   32'h00014b00
`define T25_A_ADDR   32'h00014c00
`define T25_R_ADDR   32'h00014d00
`define T25_CTRL     32'h00000005
`define T25_EXPECTED 32'hbeccc000

// Test 26: fp16_OS_N32_K32
`define T26_ID       "fp16_OS_N32_K32"
`define T26_N        32
`define T26_IS_FP16  1
`define T26_IS_OS    1
`define T26_M_DIM    1
`define T26_N_DIM    1
`define T26_K_DIM    32
`define T26_W_ADDR   32'h00014e00
`define T26_A_ADDR   32'h00014f00
`define T26_R_ADDR   32'h00015000
`define T26_CTRL     32'h00000015
`define T26_EXPECTED 32'h40690370

// Test 27: fp16_OS_N32_K4
`define T27_ID       "fp16_OS_N32_K4"
`define T27_N        32
`define T27_IS_FP16  1
`define T27_IS_OS    1
`define T27_M_DIM    1
`define T27_N_DIM    1
`define T27_K_DIM    4
`define T27_W_ADDR   32'h00015100
`define T27_A_ADDR   32'h00015200
`define T27_R_ADDR   32'h00015300
`define T27_CTRL     32'h00000015
`define T27_EXPECTED 32'hbee66000

