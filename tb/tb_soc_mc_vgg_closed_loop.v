// tb_soc_mc_vgg_closed_loop.v - multi-core runtime closed-loop RepOpt VGG test
`timescale 1ns/1ps
`include "soc_vgg_closed_loop_params.vh"

`ifndef VGG_CLOSED_INT8_SIMD_LANES
`define VGG_CLOSED_INT8_SIMD_LANES 4
`endif

`ifndef VGG_CLOSED_NPU_DATA_W
`define VGG_CLOSED_NPU_DATA_W 32
`endif

`ifndef VGG_CLOSED_NUM_CORES
`define VGG_CLOSED_NUM_CORES 1
`endif

`ifndef VGG_CLOSED_PPB_DEPTH
`define VGG_CLOSED_PPB_DEPTH 8192
`endif

module tb_soc_mc_vgg_closed_loop;
    localparam CLK_T = 10;
    localparam TIMEOUT = `VGG_CLOSED_TIMEOUT_CYCLES;
    localparam [63:0] CLK_T_TICKS = CLK_T;
    localparam [63:0] TIMEOUT_CYCLES = TIMEOUT;
    localparam [63:0] TIMEOUT_TICKS = TIMEOUT_CYCLES * CLK_T_TICKS;
    localparam DRAM_W = `VGG_CLOSED_DRAM_WORDS;
    localparam FW_LAST = `VGG_CLOSED_FW_WORDS - 1;
    localparam MARKER_OK = `VGG_CLOSED_LABEL + 32'h100;
    localparam MARKER_CLASS_MIN = 32'h100;
    localparam MARKER_CLASS_MAX = 32'h109;
    localparam NUM_CORES = `VGG_CLOSED_NUM_CORES;
    localparam PERF_FREQ_MHZ = 500;

    reg clk, rst_n, pass_seen, fail_seen, perf_dump_req;
    reg [31:0] last_marker;
    integer cyc, j;

    function [31:0] ratio_bp;
        input [63:0] num;
        input [63:0] den;
        reg [95:0] scaled;
        begin
            if (den == 64'd0) begin
                ratio_bp = 32'd0;
            end else begin
                scaled = {32'd0, num} * 96'd10000;
                ratio_bp = scaled / den;
            end
        end
    endfunction

    function [31:0] peak_ops_for_shape;
        input [1:0] cfg_shape;
        begin
            case (cfg_shape)
                2'b00: peak_ops_for_shape = 32'd4  * 32'd4  * `VGG_CLOSED_INT8_SIMD_LANES * 32'd2;
                2'b01: peak_ops_for_shape = 32'd8  * 32'd8  * `VGG_CLOSED_INT8_SIMD_LANES * 32'd2;
                2'b10: peak_ops_for_shape = 32'd16 * 32'd16 * `VGG_CLOSED_INT8_SIMD_LANES * 32'd2;
                2'b11: peak_ops_for_shape = 32'd8  * 32'd32 * `VGG_CLOSED_INT8_SIMD_LANES * 32'd2;
            endcase
        end
    endfunction

    task dump_perf;
        begin
            perf_dump_req = 1'b1;
            #1;
            perf_dump_req = 1'b0;
            #1;
        end
    endtask

    soc_mc_top #(
        .MEM_WORDS(`VGG_CLOSED_MEM_WORDS), .DRAM_WORDS(DRAM_W),
        .NUM_CORES(NUM_CORES),
        .NPU_ROWS(4), .NPU_COLS(4), .NPU_DATA_W(`VGG_CLOSED_NPU_DATA_W), .NPU_ACC_W(32),
        .NPU_PPB_DEPTH(`VGG_CLOSED_PPB_DEPTH), .NPU_PPB_THRESH(16),
        .NPU_INT8_SIMD_LANES(`VGG_CLOSED_INT8_SIMD_LANES)
    ) u_soc (.clk(clk), .rst_n(rst_n));

    initial clk = 0;
    always #(CLK_T/2) clk = ~clk;

    initial begin
        rst_n = 0;
        cyc = 0;
        pass_seen = 0;
        fail_seen = 0;
        perf_dump_req = 0;
        last_marker = 0;
        $display("[INIT] loading firmware"); $fflush();
        $readmemh(`VGG_CLOSED_FW_HEX, u_soc.u_sram.mem, 0, FW_LAST);
        $display("[INIT] fw[0]=0x%08h", u_soc.u_sram.mem[0]); $fflush();
        repeat(5) @(posedge clk);
        rst_n = 1;
    end

    initial begin
        $display("[INIT] clear DRAM words=%0d", DRAM_W);
        for (j = 0; j < DRAM_W; j = j + 1) u_soc.u_dram.mem[j] = 32'h0;
        $display("[INIT] read DRAM hex");
        $readmemh(`VGG_CLOSED_DRAM_HEX, u_soc.u_dram.mem);
        $display("[INIT] DRAM ready");
    end

    genvar perf_core_idx;
    generate
        for (perf_core_idx = 0; perf_core_idx < NUM_CORES; perf_core_idx = perf_core_idx + 1) begin : gen_perf_dump
            reg [31:0] peak_ops_per_cycle;
            reg [31:0] peak_tops_x1e6;
            reg [31:0] rd_burst_util_bp;
            reg [31:0] wr_burst_util_bp;
            always @(posedge perf_dump_req) begin
                peak_ops_per_cycle = peak_ops_for_shape(
                    u_soc.u_npu_mc.gen_cores[perf_core_idx].u_npu_core.cfg_shape_r);
                peak_tops_x1e6 = peak_ops_per_cycle * PERF_FREQ_MHZ;
                rd_burst_util_bp = ratio_bp(
                    {32'd0, u_soc.u_npu_mc.gen_cores[perf_core_idx].u_npu_core.perf_m_axi_rd_beats},
                    {32'd0, u_soc.u_npu_mc.gen_cores[perf_core_idx].u_npu_core.perf_m_axi_rd_lat});
                wr_burst_util_bp = ratio_bp(
                    {32'd0, u_soc.u_npu_mc.gen_cores[perf_core_idx].u_npu_core.perf_m_axi_wr_beats},
                    {32'd0, u_soc.u_npu_mc.gen_cores[perf_core_idx].u_npu_core.perf_m_axi_wr_lat});
                $display("[PERF]");
                $display("| core             | %-10d |", perf_core_idx);
                $display("| peak_ops_per_cycle | %-8d |", peak_ops_per_cycle);
                $display("| peak_tops_x1e6   | %-8d |", peak_tops_x1e6);
                $display("| mac_ops          | %-8d |", u_soc.u_npu_mc.gen_cores[perf_core_idx].u_npu_core.perf_mac_ops);
                $display("| ops              | %-8d |", u_soc.u_npu_mc.gen_cores[perf_core_idx].u_npu_core.perf_ops);
                $display("| busy_cycles      | %-8d |", u_soc.u_npu_mc.gen_cores[perf_core_idx].u_npu_core.perf_busy_cycles);
                $display("| compute_cycles   | %-8d |", u_soc.u_npu_mc.gen_cores[perf_core_idx].u_npu_core.perf_compute_cycles);
                $display("| dma_cycles       | %-8d |", u_soc.u_npu_mc.gen_cores[perf_core_idx].u_npu_core.perf_dma_cycles);
                $display("| rd_beats         | %-8d |", u_soc.u_npu_mc.gen_cores[perf_core_idx].u_npu_core.perf_m_axi_rd_beats);
                $display("| rd_bursts        | %-8d |", u_soc.u_npu_mc.gen_cores[perf_core_idx].u_npu_core.perf_m_axi_rd_cnt);
                $display("| rd_burst_cycles  | %-8d |", u_soc.u_npu_mc.gen_cores[perf_core_idx].u_npu_core.perf_m_axi_rd_lat);
                    $display("| rd_burst_util    | %0d.%0d%0d%% |", rd_burst_util_bp / 100, (rd_burst_util_bp % 100) / 10, rd_burst_util_bp % 10);
                $display("| wr_beats         | %-8d |", u_soc.u_npu_mc.gen_cores[perf_core_idx].u_npu_core.perf_m_axi_wr_beats);
                $display("| wr_bursts        | %-8d |", u_soc.u_npu_mc.gen_cores[perf_core_idx].u_npu_core.perf_m_axi_wr_cnt);
                $display("| wr_burst_cycles  | %-8d |", u_soc.u_npu_mc.gen_cores[perf_core_idx].u_npu_core.perf_m_axi_wr_lat);
                    $display("| wr_burst_util    | %0d.%0d%0d%% |", wr_burst_util_bp / 100, (wr_burst_util_bp % 100) / 10, wr_burst_util_bp % 10);
                $fflush();
            end
        end
    endgenerate

    always @(posedge clk) begin
        if (rst_n) begin
            cyc <= cyc + 1;
            if (u_soc.u_dram.mem[`VGG_CLOSED_MARKER_ADDR >> 2] == MARKER_OK)
                pass_seen <= 1;
            if (u_soc.u_dram.mem[`VGG_CLOSED_MARKER_ADDR >> 2] == 32'h000000FF)
                fail_seen <= 1;
            if (u_soc.u_dram.mem[`VGG_CLOSED_MARKER_ADDR >> 2] >= MARKER_CLASS_MIN &&
                u_soc.u_dram.mem[`VGG_CLOSED_MARKER_ADDR >> 2] <= MARKER_CLASS_MAX &&
                u_soc.u_dram.mem[`VGG_CLOSED_MARKER_ADDR >> 2] != MARKER_OK)
                fail_seen <= 1;
            if (u_soc.u_dram.mem[`VGG_CLOSED_MARKER_ADDR >> 2] != last_marker) begin
                last_marker <= u_soc.u_dram.mem[`VGG_CLOSED_MARKER_ADDR >> 2];
                if (u_soc.u_dram.mem[`VGG_CLOSED_MARKER_ADDR >> 2] >= 32'h200)
                    $display("[PROGRESS] marker=0x%08h cycles=%0d",
                             u_soc.u_dram.mem[`VGG_CLOSED_MARKER_ADDR >> 2], cyc);
            end
        end
    end

`ifdef DIAG_VGG_HEARTBEAT
    `ifndef DIAG_VGG_HEARTBEAT_INTERVAL
    `define DIAG_VGG_HEARTBEAT_INTERVAL 10000000
    `endif
    reg [31:0] heartbeat_cnt;
    always @(posedge clk) begin
        if (!rst_n) begin
            heartbeat_cnt <= 32'd0;
        end else if (heartbeat_cnt == (`DIAG_VGG_HEARTBEAT_INTERVAL - 1)) begin
            heartbeat_cnt <= 32'd0;
            if (NUM_CORES == 1) begin
                $display("[HB] cyc=%0d pc=0x%08h marker=0x%08h busy0=%b a4=%0d a7=%0d",
                         cyc,
                         u_soc.u_cpu.reg_pc,
                         u_soc.u_dram.mem[`VGG_CLOSED_MARKER_ADDR >> 2],
                         u_soc.u_npu_mc.gen_cores[0].u_npu_core.u_ctrl.busy,
                         u_soc.u_cpu.cpuregs[14],
                         u_soc.u_cpu.cpuregs[17]);
            end else begin
                $display("[HB] cyc=%0d pc=0x%08h marker=0x%08h busy0=%b busy1=%b a4=%0d a7=%0d",
                         cyc,
                         u_soc.u_cpu.reg_pc,
                         u_soc.u_dram.mem[`VGG_CLOSED_MARKER_ADDR >> 2],
                         u_soc.u_npu_mc.gen_cores[0].u_npu_core.u_ctrl.busy,
                         u_soc.u_npu_mc.gen_cores[1].u_npu_core.u_ctrl.busy,
                         u_soc.u_cpu.cpuregs[14],
                         u_soc.u_cpu.cpuregs[17]);
            end
            $fflush();
        end else begin
            heartbeat_cnt <= heartbeat_cnt + 32'd1;
        end
    end
`endif

    initial begin
        wait(rst_n);
        #100;
        forever begin
            #(CLK_T);
            if (pass_seen) begin
                $display("[PASS] Multi-core closed-loop VGG PASSED  cycles=%0d  cores=%0d", cyc, NUM_CORES);
                $display("  Predicted class: %0d",
                         u_soc.u_dram.mem[`VGG_CLOSED_MARKER_ADDR >> 2] - 32'h100);
                dump_perf();
                $finish;
            end
            if (fail_seen) begin
                if (u_soc.u_dram.mem[`VGG_CLOSED_MARKER_ADDR >> 2] == 32'h000000FF)
                    $display("[FAIL] Multi-core closed-loop firmware failure");
                else
                    $display("[FAIL] Multi-core closed-loop classification mismatch  predicted=%0d",
                             u_soc.u_dram.mem[`VGG_CLOSED_MARKER_ADDR >> 2] - 32'h100);
                $display("  Cycles: %0d", cyc);
                dump_perf();
                $finish;
            end
        end
    end

    initial begin
        #(TIMEOUT_TICKS);
        $display("[FAIL] Multi-core closed-loop timeout at %0d cycles", TIMEOUT);
        dump_perf();
        $finish;
    end
endmodule
