// =============================================================================
// Module  : tb_soc
// Project : NPU_prj
// Desc    : SoC system-level testbench.
//           Tests PicoRV32 CPU driving NPU to perform a 2x2 INT8 matrix multiply.
//
//   Flow:
//     1. Load firmware (soc_test.hex) into SRAM via $readmemh
//     2. Load test matrices into DRAM
//     3. Run simulation, CPU firmware configures NPU registers
//     4. NPU performs tile-loop matrix multiply (OS mode, M=2,N=2,K=2)
//     5. CPU verifies result, writes PASS/FAIL marker
//     6. Testbench checks PASS/FAIL marker and reports
//
//   Test vectors (INT8 2x2 matmul C = A * B):
//     A = [[1, 2],      B = [[5, 6],
//          [3, 4]]           [7, 8]]
//
//     C = A * B = [[1*5+2*7, 1*6+2*8],     = [[19, 22],
//                  [3*5+4*7, 3*6+4*8]]         [43, 50]]
//
//   DRAM layout:
//     W_ADDR 0x1000: B column-major (as NPU tile-loop expects)
//       B[:,0] = [5, 7] -> word 0x1000: 0x00000705
//       B[:,1] = [6, 8] -> word 0x1010: 0x00000806
//     A_ADDR 0x1010: A row-major
//       A[0,:] = [1, 2] -> word 0x1010: 0x00000201
//       A[1,:] = [3, 4] -> word 0x1020: 0x00000403
//     R_ADDR 0x1020: Result (4 x 32-bit words)
//       C[0][0]=19, C[0][1]=22, C[1][0]=43, C[1][1]=50
//     MARKER 0x2000: 0xAA (PASS) or 0xFF (FAIL)  -- DRAM space (>= 0x1000)
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
localparam PASS_MARKER  = 32'h0000_00AA;
localparam FAIL_MARKER  = 32'h0000_00FF;

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
    // Load test firmware (machine code hex)
    // Note: vvp runs from the sim/ directory, so use ../tb/ prefix.
    // Alternatively, copy soc_test.hex into sim/ before running.
    $readmemh("../tb/soc_test.hex", u_soc.u_sram.mem);
end

// ---------------------------------------------------------------------------
// Load test matrices into DRAM
// ---------------------------------------------------------------------------
initial begin : init_dram
    integer i;
    for (i = 0; i < DRAM_WORDS; i = i + 1)
        u_soc.u_dram.mem[i] = 32'h0;

    // ===================================================================
    // Weight data: B matrix column-major at W_ADDR=0x1000
    // B = [[5, 6],
    //      [7, 8]]
    //
    // K=2 INT8: k_dma_bytes = 4, stride between columns = 4 bytes
    // B[:,0] = [5, 7]: packed INT8 LE = 0x00000705  at 0x1000 (word 0x400)
    // B[:,1] = [6, 8]: packed INT8 LE = 0x00000806  at 0x1004 (word 0x401)
    // ===================================================================
    u_soc.u_dram.mem[32'h1000 >> 2]     = 32'h00000705;   // B[:,0] = [5, 7]
    u_soc.u_dram.mem[(32'h1000+4) >> 2] = 32'h00000806;   // B[:,1] = [6, 8]

    // ===================================================================
    // Activation data: A matrix row-major at A_ADDR=0x1010
    // A = [[1, 2],
    //      [3, 4]]
    //
    // A[0,:] = [1, 2]: packed INT8 LE = 0x00000201  at 0x1010 (word 0x404)
    // A[1,:] = [3, 4]: packed INT8 LE = 0x00000403  at 0x1014 (word 0x405)
    // ===================================================================
    u_soc.u_dram.mem[32'h1010 >> 2]     = 32'h00000201;   // A[0,:] = [1, 2]
    u_soc.u_dram.mem[(32'h1010+4) >> 2] = 32'h00000403;   // A[1,:] = [3, 4]

    // Result area: 4 x 32-bit words at R_ADDR=0x1020 (written by NPU DMA)
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
// The firmware writes the pass/fail magic value to DRAM address 0x2000
// DRAM word index = 0x2000 / 4 = 2048
always @(posedge clk) begin
    if (rst_n) begin
        if (u_soc.u_dram.mem[2048] == 32'h0000_00AA)
            pass_seen <= 1;
        if (u_soc.u_dram.mem[2048] == 32'h0000_00FF)
            fail_seen <= 1;
    end
end

// ---------------------------------------------------------------------------
// Timeout & result reporting
// ---------------------------------------------------------------------------

initial begin
    wait (rst_n);

    fork
        begin : wait_pass
            #100; // small delay to let simulation settle
            forever begin
                if (pass_seen) begin
                    $display("");
                    $display("========================================");
                    $display("  [PASS] SoC integration test PASSED!");
                    $display("  Cycles: %0d", cycle_count);
                    $display("========================================");
                    $finish;
                end
                if (fail_seen) begin
                    $display("");
                    $display("========================================");
                    $display("  [FAIL] SoC integration test FAILED!");
                    $display("  Cycles: %0d", cycle_count);
                    $display("  DRAM result area (0x1020):");
                    $display("    C[0][0] = %0d", $signed(u_soc.u_dram.mem[32'h1020 >> 2]));
                    $display("    C[0][1] = %0d", $signed(u_soc.u_dram.mem[(32'h1020+4) >> 2]));
                    $display("    C[1][0] = %0d", $signed(u_soc.u_dram.mem[(32'h1020+8) >> 2]));
                    $display("    C[1][1] = %0d", $signed(u_soc.u_dram.mem[(32'h1020+12) >> 2]));
                    $display("  Expected: C[0][0]=19, C[0][1]=22, C[1][0]=43, C[1][1]=50");
                    $display("========================================");
                    $finish;
                end
                #(CLK_PERIOD);
            end
        end
        begin : wait_timeout
            #(TIMEOUT_CYCLES * CLK_PERIOD);
            $display("");
            $display("========================================");
            $display("  [TIMEOUT] Simulation exceeded %0d cycles", TIMEOUT_CYCLES);
            $display("  pass_seen=%0b fail_seen=%0b", pass_seen, fail_seen);
            $display("  R_ADDR results: C00=%0d C01=%0d C10=%0d C11=%0d",
                u_soc.u_dram.mem[32'h1020>>2],
                u_soc.u_dram.mem[32'h1024>>2],
                u_soc.u_dram.mem[32'h1028>>2],
                u_soc.u_dram.mem[32'h102c>>2]);
            $display("  marker[0x2000>>2]=0x%08h", u_soc.u_dram.mem[32'h2000>>2]);
            $display("========================================");
            $finish;
        end
    join
end

// ---------------------------------------------------------------------------
// Waveform dump
// ---------------------------------------------------------------------------
initial begin
    $dumpfile("soc_sim.vcd");
    $dumpvars(0, tb_soc);
end

// ---------------------------------------------------------------------------
// Debug: trace CPU activity
// ---------------------------------------------------------------------------
integer dbg_cnt;
initial begin : dbg_trace
    wait (rst_n);
    dbg_cnt = 0;
    @(posedge clk);
    forever begin
        @(posedge clk);
        if (u_soc.mem_valid) begin
            if (dbg_cnt < 500) begin
                $display("[DBG cy=%0d] mem_valid=1 addr=0x%08h instr=%b wstrb=%04b wdata=0x%08h rdata=0x%08h",
                    cycle_count, u_soc.mem_addr, u_soc.mem_instr,
                    u_soc.mem_wstrb, u_soc.mem_wdata, u_soc.mem_rdata);
            end
            dbg_cnt = dbg_cnt + 1;
        end
    end
end

// ---------------------------------------------------------------------------
// Periodic NPU status monitor
// ---------------------------------------------------------------------------
initial begin : npu_monitor
    wait (rst_n);
    // wait until CTRL is written (NPU start)
    @(posedge clk);
    // Poll NPU status every 1000 cycles
    forever begin
        repeat (1000) @(posedge clk);
        $display("[NPU_MON cy=%0d] ctrl=0x%08h status_busy=%b status_done=%b",
            cycle_count,
            u_soc.u_npu.ctrl_reg,
            u_soc.u_npu.status_busy,
            u_soc.u_npu.status_done);
    end
end

endmodule
