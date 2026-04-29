// =============================================================================
// tb_matmul_ws.v - Explicit WS-named entry for generated matmul tests.
//
// The common direct-mode matmul testbench is parameterized by test_params.vh.
// This wrapper keeps run scripts from implying that WS cases are OS-only.
// =============================================================================

`include "tb/matmul/tb_matmul_os.v"
