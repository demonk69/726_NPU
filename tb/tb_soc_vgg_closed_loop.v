// tb_soc_vgg_closed_loop.v - runtime closed-loop RepOpt VGG test
`timescale 1ns/1ps
`include "soc_vgg_closed_loop_params.vh"

`ifndef VGG_CLOSED_INT8_SIMD_LANES
`define VGG_CLOSED_INT8_SIMD_LANES 4
`endif

`ifndef VGG_CLOSED_NPU_DATA_W
`define VGG_CLOSED_NPU_DATA_W 32
`endif

`ifndef VGG_CLOSED_PPB_DEPTH
`define VGG_CLOSED_PPB_DEPTH 8192
`endif

module tb_soc_vgg_closed_loop;
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
    localparam PERF_FREQ_MHZ = 500;

    reg clk, rst_n, pass_seen, fail_seen;
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

    task print_perf;
        reg [31:0] peak_ops_per_cycle;
        reg [31:0] peak_tops_x1e6;
        reg [31:0] rd_burst_util_bp;
        reg [31:0] wr_burst_util_bp;
        begin
            peak_ops_per_cycle = peak_ops_for_shape(u_soc.u_npu.cfg_shape_r);
            peak_tops_x1e6 = peak_ops_per_cycle * PERF_FREQ_MHZ;
            rd_burst_util_bp = ratio_bp({32'd0, u_soc.u_npu.perf_m_axi_rd_beats},
                                        {32'd0, u_soc.u_npu.perf_m_axi_rd_lat});
            wr_burst_util_bp = ratio_bp({32'd0, u_soc.u_npu.perf_m_axi_wr_beats},
                                        {32'd0, u_soc.u_npu.perf_m_axi_wr_lat});
            $display("[PERF]");
            $display("| core             | %-10d |", 0);
            $display("| peak_ops_per_cycle | %-8d |", peak_ops_per_cycle);
            $display("| peak_tops_x1e6   | %-8d |", peak_tops_x1e6);
            $display("| mac_ops          | %-8d |", u_soc.u_npu.perf_mac_ops);
            $display("| ops              | %-8d |", u_soc.u_npu.perf_ops);
            $display("| busy_cycles      | %-8d |", u_soc.u_npu.perf_busy_cycles);
            $display("| compute_cycles   | %-8d |", u_soc.u_npu.perf_compute_cycles);
            $display("| dma_cycles       | %-8d |", u_soc.u_npu.perf_dma_cycles);
            $display("| rd_beats         | %-8d |", u_soc.u_npu.perf_m_axi_rd_beats);
            $display("| rd_bursts        | %-8d |", u_soc.u_npu.perf_m_axi_rd_cnt);
            $display("| rd_burst_cycles  | %-8d |", u_soc.u_npu.perf_m_axi_rd_lat);
            $display("| rd_burst_util    | %0d.%0d%0d%% |", rd_burst_util_bp / 100, (rd_burst_util_bp % 100) / 10, rd_burst_util_bp % 10);
            $display("| wr_beats         | %-8d |", u_soc.u_npu.perf_m_axi_wr_beats);
            $display("| wr_bursts        | %-8d |", u_soc.u_npu.perf_m_axi_wr_cnt);
            $display("| wr_burst_cycles  | %-8d |", u_soc.u_npu.perf_m_axi_wr_lat);
            $display("| wr_burst_util    | %0d.%0d%0d%% |", wr_burst_util_bp / 100, (wr_burst_util_bp % 100) / 10, wr_burst_util_bp % 10);
            $fflush();
        end
    endtask

    soc_top #(
        .MEM_WORDS(`VGG_CLOSED_MEM_WORDS), .DRAM_WORDS(DRAM_W),
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
        last_marker = 0;
        #100;
        rst_n = 1;
    end

    initial $readmemh(`VGG_CLOSED_FW_HEX, u_soc.u_sram.mem, 0, FW_LAST);

    initial begin
        $display("[INIT] clear DRAM words=%0d", DRAM_W);
        $fflush();
        for (j = 0; j < DRAM_W; j = j + 1) u_soc.u_dram.mem[j] = 32'h0;
        $display("[INIT] read DRAM hex");
        $fflush();
        $readmemh(`VGG_CLOSED_DRAM_HEX, u_soc.u_dram.mem);
        $display("[INIT] DRAM ready");
        $fflush();
    end

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
                $fflush();
            end
        end
    end

`ifdef DIAG_VGG_HEARTBEAT
    reg [31:0] heartbeat_cnt;
    always @(posedge clk) begin
        if (!rst_n) begin
            heartbeat_cnt <= 32'd0;
        end else if (heartbeat_cnt == 32'd999999) begin
            heartbeat_cnt <= 32'd0;
            $display("[HEARTBEAT] cycles=%0d marker=0x%08h pc=0x%08h trap=%0d mem_valid=%0d mem_addr=0x%08h npu_busy=%0d npu_done=%0d npu_error=%0d err=0x%08h ctrl_state=%0d tile_i=%0d tile_j=%0d k_tile=%0d dma_load=%0d dma_wb=%0d w_full=%0d a_full=%0d r_full=%0d",
                     cyc,
                     u_soc.u_dram.mem[`VGG_CLOSED_MARKER_ADDR >> 2],
                     u_soc.u_cpu.reg_pc,
                     u_soc.u_cpu.trap,
                     u_soc.mem_valid,
                     u_soc.mem_addr,
                     u_soc.u_npu.status_busy,
                     u_soc.u_npu.status_done,
                     u_soc.u_npu.status_error,
                     u_soc.u_npu.err_status,
                     u_soc.u_npu.u_ctrl.state,
                     u_soc.u_npu.u_ctrl.tile_i,
                     u_soc.u_npu.u_ctrl.tile_j,
                     u_soc.u_npu.u_ctrl.k_tile_idx,
                     u_soc.u_npu.u_dma.load_state,
                     u_soc.u_npu.u_dma.wb_state,
                     u_soc.u_npu.w_ppb_full,
                     u_soc.u_npu.a_ppb_full,
                     u_soc.u_npu.r_fifo_full);
            $fflush();
        end else begin
            heartbeat_cnt <= heartbeat_cnt + 32'd1;
        end
    end
`endif

    initial begin
        wait(rst_n);
        fork
            begin : wait_done
                #100;
                forever begin
                    if (pass_seen) begin
                        $display("[PASS] Runtime closed-loop VGG classification PASSED");
                        $display("  Predicted class: %0d (expected exact-python: %0d, fixed-runtime: %0d)",
                                 u_soc.u_dram.mem[`VGG_CLOSED_MARKER_ADDR >> 2] - 32'h100,
                                 `VGG_CLOSED_EXACT_LABEL, `VGG_CLOSED_FIXED_LABEL);
                        $display("  Cycles: %0d", cyc);
                        print_perf();
                        $fflush();
                        $finish;
                    end
                    if (fail_seen) begin
                        if (u_soc.u_dram.mem[`VGG_CLOSED_MARKER_ADDR >> 2] == 32'h000000FF) begin
                            $display("[FAIL] Runtime closed-loop firmware failure");
                        end else begin
                            $display("[FAIL] Runtime closed-loop classification mismatch");
                            $display("  Predicted class: %0d (expected exact-python: %0d, fixed-runtime: %0d)",
                                     u_soc.u_dram.mem[`VGG_CLOSED_MARKER_ADDR >> 2] - 32'h100,
                                     `VGG_CLOSED_EXACT_LABEL, `VGG_CLOSED_FIXED_LABEL);
                        end
                        $display("  Cycles: %0d", cyc);
                        print_perf();
                        $fflush();
                        $finish;
                    end
                    #(CLK_T);
                end
            end
            begin
                #(TIMEOUT_TICKS);
                $display("[TIMEOUT] %0d cycles", TIMEOUT);
                $display("  marker=0x%08h pc=0x%08h trap=%0d mem_valid=%0d mem_addr=0x%08h",
                         u_soc.u_dram.mem[`VGG_CLOSED_MARKER_ADDR >> 2],
                         u_soc.u_cpu.reg_pc, u_soc.u_cpu.trap,
                         u_soc.mem_valid, u_soc.mem_addr);
                $display("  npu busy=%0d done=%0d error=%0d err=0x%08h state=%0d tile_i=%0d tile_j=%0d k_tile=%0d",
                         u_soc.u_npu.status_busy, u_soc.u_npu.status_done,
                         u_soc.u_npu.status_error, u_soc.u_npu.err_status,
                         u_soc.u_npu.u_ctrl.state, u_soc.u_npu.u_ctrl.tile_i,
                         u_soc.u_npu.u_ctrl.tile_j, u_soc.u_npu.u_ctrl.k_tile_idx);
                print_perf();
                $fflush();
                $finish;
            end
        join
    end
endmodule
