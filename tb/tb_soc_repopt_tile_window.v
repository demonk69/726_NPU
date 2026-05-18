// =============================================================================
// Module  : tb_soc_repopt_tile_window
// Project : NPU_prj
// Desc    : SoC smoke where the reference CPU schedules multiple RepOpt
//           4x4 tile-mode GEMMs through NPU MMIO, then runs postprocess.
// =============================================================================

`timescale 1ns/1ps

`include "soc_repopt_tile_window_params.vh"

module tb_soc_repopt_tile_window;

localparam CLK_PERIOD     = 10;
localparam TIMEOUT_CYCLES = `REP_TILE_SOC_TIMEOUT_CYCLES;
localparam MEM_WORDS      = 2048;
localparam DRAM_WORDS     = `REP_TILE_SOC_DRAM_WORDS;
localparam PASS_MARKER    = 32'h0000_00AA;
localparam FAIL_MARKER    = 32'h0000_00FF;
localparam FW_LAST_WORD   = `REP_TILE_SOC_FW_WORDS - 1;

reg clk;
reg rst_n;
integer cycle_count;
integer raw_i;
integer q_i;

reg [31:0] expected_raw [0:`REP_TILE_SOC_RAW_COUNT-1];
reg [31:0] expected_raw_addr [0:`REP_TILE_SOC_RAW_COUNT-1];
reg [31:0] expected_q [0:`REP_TILE_SOC_Q_COUNT-1];
`ifdef REP_TILE_SOC_Q1_BASE
reg [31:0] expected_q1 [0:`REP_TILE_SOC_L1_Q_COUNT-1];
`endif

soc_top #(
    .MEM_WORDS      (MEM_WORDS),
    .DRAM_WORDS     (DRAM_WORDS),
    .NPU_ROWS       (4),
    .NPU_COLS       (4),
    .NPU_DATA_W     (32),
    .NPU_ACC_W      (32),
    .NPU_PPB_DEPTH  (64),
    .NPU_PPB_THRESH (16)
) u_soc (
    .clk   (clk),
    .rst_n (rst_n)
);

initial clk = 0;
always #(CLK_PERIOD/2) clk = ~clk;

initial begin
    rst_n = 0;
    cycle_count = 0;
    #100;
    rst_n = 1;
end

initial begin
    $readmemh(`REP_TILE_SOC_FW_HEX, u_soc.u_sram.mem, 0, FW_LAST_WORD);
end

initial begin
    $readmemh(`REP_TILE_SOC_RAW_ADDR_HEX, expected_raw_addr);
    $readmemh(`REP_TILE_SOC_EXPECTED_RAW_HEX, expected_raw);
    $readmemh(`REP_TILE_SOC_EXPECTED_Q_HEX, expected_q);
`ifdef REP_TILE_SOC_Q1_BASE
    $readmemh(`REP_TILE_SOC_EXPECTED_Q1_HEX, expected_q1);
`endif
end

initial begin : init_dram
    integer i;
    for (i = 0; i < DRAM_WORDS; i = i + 1)
        u_soc.u_dram.mem[i] = 32'h0;
    $readmemh(`REP_TILE_SOC_DRAM_HEX, u_soc.u_dram.mem);
end

always @(posedge clk) begin
    if (!rst_n)
        cycle_count <= 0;
    else
        cycle_count <= cycle_count + 1;
end

reg pass_seen;
reg fail_seen;

initial begin
    pass_seen = 0;
    fail_seen = 0;
end

always @(posedge clk) begin
    if (rst_n) begin
        if (u_soc.u_dram.mem[`REP_TILE_SOC_MARKER_ADDR >> 2] == PASS_MARKER)
            pass_seen <= 1;
        if (u_soc.u_dram.mem[`REP_TILE_SOC_MARKER_ADDR >> 2] == FAIL_MARKER)
            fail_seen <= 1;
    end
end

task print_summary;
    begin
        $display("  window: M[%0d:%0d) N[%0d:%0d)",
                 `REP_TILE_SOC_M_BASE,
                 `REP_TILE_SOC_M_BASE + `REP_TILE_SOC_M_TILES * 4,
                 `REP_TILE_SOC_N_BASE,
                 `REP_TILE_SOC_N_BASE + `REP_TILE_SOC_N_TILES * 4);
        $display("  tiles scheduled by CPU: %0d", `REP_TILE_SOC_TILE_COUNT);
        $display("  first tile result[0] = %0d",
                 $signed(u_soc.u_dram.mem[`REP_TILE_SOC_R_ADDR_0 >> 2]));
        $display("  first postprocess q[0] = %0d",
                 $signed(u_soc.u_dram.mem[`REP_TILE_SOC_Q_BASE >> 2]));
`ifdef REP_TILE_SOC_IFM1_BASE
        $display("  first IFM word = 0x%08h",
                 u_soc.u_dram.mem[`REP_TILE_SOC_IFM1_BASE >> 2]);
`endif
    end
endtask

function results_match;
    reg ok;
    begin
        ok = 1'b1;
        for (raw_i = 0; raw_i < `REP_TILE_SOC_RAW_COUNT; raw_i = raw_i + 1) begin
            if (u_soc.u_dram.mem[expected_raw_addr[raw_i] >> 2] !== expected_raw[raw_i])
                ok = 1'b0;
        end
        for (q_i = 0; q_i < `REP_TILE_SOC_Q_COUNT; q_i = q_i + 1) begin
            if (u_soc.u_dram.mem[(`REP_TILE_SOC_Q_BASE >> 2) + q_i] !== expected_q[q_i])
                ok = 1'b0;
        end
`ifdef REP_TILE_SOC_Q1_BASE
        for (q_i = 0; q_i < `REP_TILE_SOC_L1_Q_COUNT; q_i = q_i + 1) begin
            if (u_soc.u_dram.mem[(`REP_TILE_SOC_Q1_BASE >> 2) + q_i] !== expected_q1[q_i])
                ok = 1'b0;
        end
`endif
        results_match = ok;
    end
endfunction

task print_mismatch;
    reg printed;
    begin
        printed = 1'b0;
        for (raw_i = 0; raw_i < `REP_TILE_SOC_RAW_COUNT; raw_i = raw_i + 1) begin
            if (!printed && u_soc.u_dram.mem[expected_raw_addr[raw_i] >> 2] !== expected_raw[raw_i]) begin
                $display("  raw mismatch[%0d] addr=0x%08h got=%0d expected=%0d",
                         raw_i,
                         expected_raw_addr[raw_i],
                         $signed(u_soc.u_dram.mem[expected_raw_addr[raw_i] >> 2]),
                         $signed(expected_raw[raw_i]));
                printed = 1'b1;
            end
        end
        for (q_i = 0; q_i < `REP_TILE_SOC_Q_COUNT; q_i = q_i + 1) begin
            if (!printed && u_soc.u_dram.mem[(`REP_TILE_SOC_Q_BASE >> 2) + q_i] !== expected_q[q_i]) begin
                $display("  q mismatch[%0d] got=%0d expected=%0d",
                         q_i,
                         $signed(u_soc.u_dram.mem[(`REP_TILE_SOC_Q_BASE >> 2) + q_i]),
                         $signed(expected_q[q_i]));
                printed = 1'b1;
            end
        end
    end
endtask

initial begin
    wait (rst_n);

    fork
        begin : wait_done
            #100;
            forever begin
                if (pass_seen) begin
                    $display("");
                    $display("========================================");
                    if (results_match()) begin
                        $display("  [PASS] RepOpt tile-window SoC MMIO + CPU postprocess test PASSED!");
`ifdef REP_TILE_SOC_IFM1_BASE
                        begin
                            integer ifm_dump_fd, ifm_i;
                            ifm_dump_fd = $fopen("ifm_dump.hex", "w");
                            for (ifm_i = 0; ifm_i < `REP_TILE_SOC_IFM1_WORDS; ifm_i = ifm_i + 1)
                                $fwrite(ifm_dump_fd, "%08h\n", u_soc.u_dram.mem[(`REP_TILE_SOC_IFM1_BASE >> 2) + ifm_i]);
                            $fclose(ifm_dump_fd);
                            $display("  [IFM] dumped %0d words to ifm_dump.hex", `REP_TILE_SOC_IFM1_WORDS);
                        end
`endif
                    end                     else begin
                        $display("  [FAIL] PASS marker seen but raw/q result memory mismatched!");
                        print_mismatch();
                    end
`ifdef REP_TILE_SOC_IFM1_BASE
                    // Also dump IFM for debug regardless of pass/fail
                    begin
                        integer ifm_dump_fd, ifm_i;
                        ifm_dump_fd = $fopen("ifm_dump.hex", "w");
                        for (ifm_i = 0; ifm_i < `REP_TILE_SOC_IFM1_WORDS; ifm_i = ifm_i + 1)
                            $fwrite(ifm_dump_fd, "%08h\n", u_soc.u_dram.mem[(`REP_TILE_SOC_IFM1_BASE >> 2) + ifm_i]);
                        $fclose(ifm_dump_fd);
                        $display("  [IFM] dumped %0d words to ifm_dump.hex", `REP_TILE_SOC_IFM1_WORDS);
                    end
`endif
                    // Debug: dump first 4 raw results of L1
                    $display("  L1 raw[0..3]: %0d %0d %0d %0d",
                             $signed(u_soc.u_dram.mem[(`REP_TILE_SOC_R_ADDR_0 >> 2) + 64 + 0]),
                             $signed(u_soc.u_dram.mem[(`REP_TILE_SOC_R_ADDR_0 >> 2) + 64 + 1]),
                             $signed(u_soc.u_dram.mem[(`REP_TILE_SOC_R_ADDR_0 >> 2) + 64 + 2]),
                             $signed(u_soc.u_dram.mem[(`REP_TILE_SOC_R_ADDR_0 >> 2) + 64 + 3]));
                    $display("  L1 q[0..3]: %0d %0d %0d %0d",
                             $signed(u_soc.u_dram.mem[(`REP_TILE_SOC_Q_BASE >> 2) + 64 + 0]),
                             $signed(u_soc.u_dram.mem[(`REP_TILE_SOC_Q_BASE >> 2) + 64 + 1]),
                             $signed(u_soc.u_dram.mem[(`REP_TILE_SOC_Q_BASE >> 2) + 64 + 2]),
                             $signed(u_soc.u_dram.mem[(`REP_TILE_SOC_Q_BASE >> 2) + 64 + 3]));
                    $display("  Cycles: %0d", cycle_count);
                    print_summary();
                    $display("========================================");
                    $finish;
                end
                if (fail_seen) begin
                    $display("");
                    $display("========================================");
                    $display("  [FAIL] RepOpt tile-window SoC firmware reported failure!");
                    $display("  Cycles: %0d", cycle_count);
                    print_summary();
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
            $display("  [TIMEOUT] RepOpt tile-window SoC test exceeded %0d cycles", TIMEOUT_CYCLES);
            $display("  pass_seen=%0b fail_seen=%0b", pass_seen, fail_seen);
            print_summary();
            $display("========================================");
            $finish;
        end
    join
end

initial begin
`ifdef DUMP_VCD
    $dumpfile("soc_repopt_tile_window.vcd");
    $dumpvars(0, tb_soc_repopt_tile_window);
`endif
end

endmodule
