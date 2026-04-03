// =============================================================================
// Module  : tb_soc
// Project : NPU_prj
// Desc    : SoC system-level testbench.
//           Tests PicoRV32 CPU driving NPU to perform a 4x4 INT8 matrix multiply.
//
//   Flow:
//     1. Load firmware (soc_test.hex) into SRAM via $readmemh
//     2. Load test matrices into DRAM
//     3. Run simulation, CPU firmware configures NPU registers
//     4. NPU performs matrix multiply with Ping-Pong buffer
//     5. CPU verifies result, writes PASS/FAIL marker
//     6. Testbench checks PASS/FAIL marker and reports
//
//   Test vectors (INT8 4×4 matmul C = A × B):
//     A = [[1, 2, 3, 4],
//          [5, 6, 7, 8],
//          [9,10,11,12],
//          [13,14,15,16]]
//
//     B = [[1, 0, 0, 0],
//          [0, 1, 0, 0],
//          [0, 0, 1, 0],
//          [0, 0, 0, 1]]
//
//     C = A (identity multiply)
//
//     C[0][0] = 1*1 + 2*0 + 3*0 + 4*0 = 1
//     C[0][1] = 1*0 + 2*1 + 3*0 + 4*0 = 2
//     ...
// =============================================================================

`timescale 1ns/1ps

module tb_soc;

// ---------------------------------------------------------------------------
// Parameters
// ---------------------------------------------------------------------------
localparam CLK_PERIOD   = 10;       // 100MHz
localparam TIMEOUT_CYCLES = 200000; // 2ms timeout
localparam MEM_WORDS    = 1024;     // 4KB SRAM
localparam DRAM_WORDS   = 15360;    // ~60KB DRAM
localparam NPU_BASE     = 32'h0200_0000;
localparam DRAM_BASE    = 32'h0000_1000;
localparam PASS_MARKER  = 32'h0000_0F00;
localparam FAIL_MARKER  = 32'h0000_0F04;

// ---------------------------------------------------------------------------
// Signals
// ---------------------------------------------------------------------------
reg  clk, rst_n;
integer cycle_count;

// ---------------------------------------------------------------------------
// DUT
// ---------------------------------------------------------------------------
soc_top #(
    .MEM_WORDS     (MEM_WORDS),
    .DRAM_WORDS    (DRAM_WORDS),
    .NPU_ROWS      (4),
    .NPU_COLS      (4),
    .NPU_DATA_W    (16),
    .NPU_ACC_W     (32),
    .NPU_PPB_DEPTH (32),
    .NPU_PPB_THRESH(16)
) u_soc (
    .clk    (clk),
    .rst_n  (rst_n)
);

// ---------------------------------------------------------------------------
// Clock generation
// ---------------------------------------------------------------------------
initial clk = 0;
always #(CLK_PERIOD/2) clk = ~clk;

// ---------------------------------------------------------------------------
// Reset
// ---------------------------------------------------------------------------
initial begin
    rst_n = 0;
    #100;
    rst_n = 1;
end

// ---------------------------------------------------------------------------
// Load firmware into SRAM (direct memory initialization)
// ---------------------------------------------------------------------------
initial begin
    // Load test firmware
    $readmemh("soc_test.hex", u_soc.u_sram.mem);
end

// ---------------------------------------------------------------------------
// Load test matrices into DRAM
// ---------------------------------------------------------------------------
initial begin : init_dram
    integer i;
    // Weight matrix W (4×4 INT8, stored row-major, 2 bytes per element)
    // W = identity matrix
    // W[0]=1, W[1]=0, W[2]=0, W[3]=0, W[4]=0, W[5]=1, ...
    // Stored as 32-bit words (each word holds 2 INT8 values)
    for (i = 0; i < 8; i = i + 1) begin
        case (i)
            0: u_soc.u_dram.mem[i] = 32'h0001_0000; // W[0][0]=0, W[0][1]=1
            1: u_soc.u_dram.mem[i] = 32'h0001_0000; // W[0][2]=0, W[0][3]=1
            2: u_soc.u_dram.mem[i] = 32'h0001_0000; // W[1][0]=0, W[1][1]=1
            3: u_soc.u_dram.mem[i] = 32'h0001_0000; // W[1][2]=0, W[1][3]=1
            4: u_soc.u_dram.mem[i] = 32'h0001_0000; // W[2][0]=0, W[2][1]=1
            5: u_soc.u_dram.mem[i] = 32'h0001_0000; // W[2][2]=0, W[2][3]=1
            6: u_soc.u_dram.mem[i] = 32'h0001_0000; // W[3][0]=0, W[3][1]=1
            7: u_soc.u_dram.mem[i] = 32'h0001_0000; // W[3][2]=0, W[3][3]=1
        endcase
    end

    // Activation matrix A (4×4 INT8)
    // A = [[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]]
    // Starting at DRAM word offset 32 (= byte addr 0x1200)
    // INT8 values packed: low halfword = [n], high halfword = [n+1]
    for (i = 32; i < 40; i = i + 1) begin
        case (i - 32)
            0: u_soc.u_dram.mem[i] = {8'd3, 8'd4, 8'd1, 8'd2};   // row 0: 1,2,3,4
            1: u_soc.u_dram.mem[i] = {8'd7, 8'd8, 8'd5, 8'd6};   // row 1: 5,6,7,8
            2: u_soc.u_dram.mem[i] = {8'd11,8'd12,8'd9, 8'd10};  // row 2: 9,10,11,12
            3: u_soc.u_dram.mem[i] = {8'd15,8'd16,8'd13,8'd14};  // row 3: 13,14,15,16
            default: u_soc.u_dram.mem[i] = 0;
        endcase
    end
end

// ---------------------------------------------------------------------------
// Timeout counter
// ---------------------------------------------------------------------------
always @(posedge clk) begin
    if (rst_n)
        cycle_count <= cycle_count + 1;
end

// ---------------------------------------------------------------------------
// Pass/Fail check
// ---------------------------------------------------------------------------
reg pass_seen, fail_seen;

initial begin
    pass_seen = 0;
    fail_seen = 0;
end

// Monitor DRAM for pass/fail marker (CPU writes it there after verification)
// The firmware writes the pass/fail magic value to DRAM address 0x0000_0F00
always @(posedge clk) begin
    if (rst_n) begin
        // Check DRAM word at pass_marker address
        // DRAM word index = 0x0F00 / 4 = 960
        if (u_soc.u_dram.mem[960] == 32'h0000_00AA)
            pass_seen <= 1;
        if (u_soc.u_dram.mem[960] == 32'h0000_00FF)
            fail_seen <= 1;
    end
end

// ---------------------------------------------------------------------------
// Timeout & result reporting
// ---------------------------------------------------------------------------
initial begin
    wait (rst_n);

    // Wait for pass or fail or timeout
    fork
        begin : wait_pass
            wait (pass_seen);
            $display("");
            $display("========================================");
            $display("  [PASS] SoC integration test PASSED!");
            $display("  Cycles: %0d", cycle_count);
            $display("========================================");
            $finish;
        end
        begin : wait_fail
            wait (fail_seen);
            $display("");
            $display("========================================");
            $display("  [FAIL] SoC integration test FAILED!");
            $display("  Cycles: %0d", cycle_count);
            $display("========================================");
            $finish;
        end
        begin : wait_timeout
            #(TIMEOUT_CYCLES * CLK_PERIOD);
            $display("");
            $display("========================================");
            $display("  [TIMEOUT] Simulation exceeded %0d cycles", TIMEOUT_CYCLES);
            $display("  pass_seen=%0b, fail_seen=%0b", pass_seen, fail_seen);
            $display("========================================");
            $finish;
        end
    join_any
    disable fork;
end

// ---------------------------------------------------------------------------
// Waveform dump
// ---------------------------------------------------------------------------
initial begin
    $dumpfile("soc_sim.vcd");
    $dumpvars(0, tb_soc);
end

endmodule
