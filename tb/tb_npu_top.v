// =============================================================================
// Module  : tb_npu_top
// Project : NPU_prj
// Desc    : NPU top-level system testbench.
//           Includes: AXI-Lite BFM, AXI4 DRAM model, AXI monitor, op_counter.
//           Tests:
//             1. INT8 4x4 matrix multiply - Weight Stationary
//             2. INT8 4x4 matrix multiply - Output Stationary
//             3. INT8 8x8 tiled on 4x4 array
//             4. Print bandwidth & operation statistics
// =============================================================================

`timescale 1ns/1ps

module tb_npu_top;

// ---------------------------------------------------------------------------
// Parameters
// ---------------------------------------------------------------------------
localparam ROWS   = 4;
localparam COLS   = 4;
localparam DATA_W = 16;
localparam ACC_W  = 32;
localparam CLK_T  = 10;   // 100 MHz
localparam DRAM_SZ = 16384;

// ---------------------------------------------------------------------------
// Clock & Reset
// ---------------------------------------------------------------------------
reg clk = 0;
always #(CLK_T/2) clk = ~clk;

reg rst_n = 0;
initial begin
    #(CLK_T*5);
    rst_n = 1;
end

// ---------------------------------------------------------------------------
// AXI4-Lite Slave Signals (CPU config port)
// ---------------------------------------------------------------------------
reg  [31:0] s_awaddr, s_wdata;
reg  [3:0]  s_wstrb;
reg         s_awvalid, s_wvalid, s_bready;
reg  [31:0] s_araddr;
reg         s_arvalid, s_rready;
wire        s_awready, s_wready, s_bvalid;
wire [1:0]  s_bresp;
wire        s_arready, s_rvalid;
wire [31:0] s_rdata;
wire [1:0]  s_rresp;

// ---------------------------------------------------------------------------
// AXI4 Master Signals (DMA port)
// ---------------------------------------------------------------------------
wire [31:0]     m_awaddr;
wire [7:0]      m_awlen;
wire [2:0]      m_awsize;
wire [1:0]      m_awburst;
wire            m_awvalid;
reg             m_awready;  // driven in initial + always
wire [ACC_W-1:0]m_wdata;
wire [ACC_W/8-1:0] m_wstrb;
wire            m_wlast;
wire            m_wvalid;
reg             m_wready;
wire [1:0]      m_bresp;
reg             m_bvalid;
wire            m_bready;
wire [31:0]     m_araddr;
wire [7:0]      m_arlen;
wire [2:0]      m_arsize;
wire [1:0]      m_arburst;
wire            m_arvalid;
reg             m_arready;
wire [ACC_W-1:0]m_rdata;
wire [1:0]      m_rresp;
reg             m_rvalid;
reg             m_rlast;
wire            m_rready;

wire npu_irq;

// ---------------------------------------------------------------------------
// DRAM Model (simple behavioral RAM)
// ---------------------------------------------------------------------------
reg [ACC_W-1:0] dram [0:DRAM_SZ-1];

// ---------------------------------------------------------------------------
// AXI4 Master DRAM Read Model
// ---------------------------------------------------------------------------
reg [31:0] rd_base_addr, rd_len;
reg [7:0]  rd_cnt;
reg        rd_active;

// No continuous assign for m_arready (reg, driven in always block)

always @(posedge clk) begin
    if (!rst_n) begin
        rd_active <= 0; rd_cnt <= 0;
        m_rvalid  <= 0; m_rlast <= 0;
    end else if (!rd_active && m_arvalid && m_arready) begin
        rd_active  <= 1;
        rd_base_addr <= m_araddr;
        rd_len     <= m_arlen;
        rd_cnt     <= 0;
    end else if (rd_active) begin
        if (!m_rready) begin
            // DMA stopped reading, force complete
            rd_active <= 0;
            m_rvalid  <= 0;
            m_rlast   <= 0;
        end else if (m_rvalid && m_rready) begin
            if (m_rlast || rd_cnt >= rd_len) begin
                rd_active <= 0;
                m_rvalid <= 0;
            end
            rd_cnt <= rd_cnt + 1;
        end else begin
            m_rvalid <= 1;
            m_rlast  <= (rd_cnt >= rd_len);
        end
    end
end

wire [31:0] rd_word_addr = (rd_base_addr + rd_cnt * 4) >> 2;
assign m_rdata = dram[rd_word_addr % DRAM_SZ];
assign m_rresp = 2'b00;

// ---------------------------------------------------------------------------
// AXI4 Master DRAM Write Model
// ---------------------------------------------------------------------------
reg [31:0] wr_base_addr, wr_len;
reg [7:0]  wr_cnt;
reg        wr_phase; // 0=addr, 1=data
reg        b_pending;

always @(posedge clk) begin
    if (!rst_n) begin
        m_awready <= 1'b0;
        m_arready <= 1'b0;
    end else begin
        m_awready <= 1'b1;
        m_arready <= 1'b1;
    end
end

always @(posedge clk) begin
    if (!rst_n) begin
        wr_phase <= 0; wr_cnt <= 0;
        m_wready <= 0; m_bvalid <= 0; b_pending <= 0;
    end else begin
        // AW handshake
        if (m_awvalid && m_awready && !wr_phase) begin
            wr_phase <= 1;
            wr_base_addr <= m_awaddr;
            wr_len <= m_awlen;
            wr_cnt <= 0;
            m_wready <= 1;
        end
        // W data
        if (wr_phase && m_wvalid && m_wready) begin
            dram[(wr_base_addr + wr_cnt * 4) >> 2] <= m_wdata;
            wr_cnt <= wr_cnt + 1;
            if (m_wlast) begin
                wr_phase <= 0;
                m_wready <= 0;
                b_pending <= 1;
            end
        end
        // B response
        if (b_pending && m_bready) begin
            m_bvalid <= 1'b1;
            b_pending <= 0;
        end else if (m_bvalid && m_bready) begin
            m_bvalid <= 1'b0;
        end
    end
end
assign m_bresp = 2'b00;

// ---------------------------------------------------------------------------
// NPU DUT
// ---------------------------------------------------------------------------
npu_top #(
    .PHY_ROWS(16), .PHY_COLS(16), .DATA_W(DATA_W), .ACC_W(ACC_W)
) u_npu (
    .sys_clk       (clk),
    .sys_rst_n     (rst_n),
    .s_axi_awaddr  (s_awaddr),
    .s_axi_awvalid (s_awvalid),
    .s_axi_awready (s_awready),
    .s_axi_wdata   (s_wdata),
    .s_axi_wstrb   (s_wstrb),
    .s_axi_wvalid  (s_wvalid),
    .s_axi_wready  (s_wready),
    .s_axi_bresp   (s_bresp),
    .s_axi_bvalid  (s_bvalid),
    .s_axi_bready  (s_bready),
    .s_axi_araddr  (s_araddr),
    .s_axi_arvalid (s_arvalid),
    .s_axi_arready (s_arready),
    .s_axi_rdata   (s_rdata),
    .s_axi_rresp   (s_rresp),
    .s_axi_rvalid  (s_rvalid),
    .s_axi_rready  (s_rready),
    .m_axi_awaddr  (m_awaddr),
    .m_axi_awlen   (m_awlen),
    .m_axi_awsize  (m_awsize),
    .m_axi_awburst (m_awburst),
    .m_axi_awvalid (m_awvalid),
    .m_axi_awready (m_awready),
    .m_axi_wdata   (m_wdata),
    .m_axi_wstrb   (m_wstrb),
    .m_axi_wlast   (m_wlast),
    .m_axi_wvalid  (m_wvalid),
    .m_axi_wready  (m_wready),
    .m_axi_bresp   (m_bresp),
    .m_axi_bvalid  (m_bvalid),
    .m_axi_bready  (m_bready),
    .m_axi_araddr  (m_araddr),
    .m_axi_arlen   (m_arlen),
    .m_axi_arsize  (m_arsize),
    .m_axi_arburst (m_arburst),
    .m_axi_arvalid (m_arvalid),
    .m_axi_arready (m_arready),
    .m_axi_rdata   (m_rdata),
    .m_axi_rresp   (m_rresp),
    .m_axi_rvalid  (m_rvalid),
    .m_axi_rready  (m_rready),
    .m_axi_rlast   (m_rlast),
    .npu_irq       (npu_irq)
);

// ---------------------------------------------------------------------------
// AXI Monitor
// ---------------------------------------------------------------------------
wire [31:0] s_axi_wr_cnt, s_axi_rd_cnt, s_axi_wr_beats, s_axi_rd_beats;
wire [31:0] s_axi_wr_lat, s_axi_rd_lat;
wire [31:0] m_axi_wr_cnt, m_axi_rd_cnt, m_axi_wr_bytes, m_axi_rd_bytes;
wire [31:0] m_axi_wr_beats, m_axi_rd_beats, m_axi_wr_lat, m_axi_rd_lat;
wire [31:0] total_cycles, m_axi_rd_bw, m_axi_wr_bw;

axi_monitor #(.ACC_W(ACC_W)) u_monitor (
    .clk(clk), .rst_n(rst_n),
    .s_awvalid(s_awvalid),  .s_awready(s_awready),
    .s_wvalid(s_wvalid),    .s_wready(s_wready),
    .s_bvalid(s_bvalid),    .s_bready(s_bready),
    .s_arvalid(s_arvalid),  .s_arready(s_arready),
    .s_rvalid(s_rvalid),    .s_rready(s_rready),
    .m_awlen(m_awlen),      .m_awvalid(m_awvalid),
    .m_awready(m_awready),  .m_wlast(m_wlast),
    .m_wvalid(m_wvalid),    .m_wready(m_wready),
    .m_bvalid(m_bvalid),    .m_bready(m_bready),
    .m_arlen(m_arlen),      .m_arvalid(m_arvalid),
    .m_arready(m_arready),  .m_rlast(m_rlast),
    .m_rvalid(m_rvalid),    .m_rready(m_rready),
    .s_axi_wr_cnt(s_axi_wr_cnt),
    .s_axi_rd_cnt(s_axi_rd_cnt),
    .s_axi_wr_beats(s_axi_wr_beats),
    .s_axi_rd_beats(s_axi_rd_beats),
    .s_axi_wr_lat(s_axi_wr_lat),
    .s_axi_rd_lat(s_axi_rd_lat),
    .m_axi_wr_cnt(m_axi_wr_cnt),
    .m_axi_rd_cnt(m_axi_rd_cnt),
    .m_axi_wr_bytes(m_axi_wr_bytes),
    .m_axi_rd_bytes(m_axi_rd_bytes),
    .m_axi_wr_beats(m_axi_wr_beats),
    .m_axi_rd_beats(m_axi_rd_beats),
    .m_axi_wr_lat(m_axi_wr_lat),
    .m_axi_rd_lat(m_axi_rd_lat),
    .total_cycles(total_cycles),
    .m_axi_rd_bw(m_axi_rd_bw),
    .m_axi_wr_bw(m_axi_wr_bw)
);

// ---------------------------------------------------------------------------
// Operation Counter
// ---------------------------------------------------------------------------
wire [63:0] total_mac_ops;
wire [31:0] total_pe_cycles, total_busy_cycles, total_compute_cycles;
wire [31:0] total_dma_cycles, active_pe_cnt, peak_active_pe;
wire [31:0] fsm_trans_cnt, util_pct, mac_per_cyc, eff_pct;

op_counter #(.ROWS(ROWS), .COLS(COLS)) u_opcnt (
    .clk(clk), .rst_n(rst_n),
    .pe_en(u_npu.pe_en),
    .pe_flush(u_npu.pe_flush),
    .ctrl_busy(u_npu.status_busy),
    .ctrl_done(u_npu.status_done),
    .dma_w_done(u_npu.dma_w_done),
    .dma_a_done(u_npu.dma_a_done),
    .dma_r_done(u_npu.dma_r_done),
    .pe_valid(u_npu.pe_array_valid[COLS-1:0]),
    .m_dim(u_npu.m_dim_r),

    .n_dim(u_npu.n_dim_r),
    .k_dim(u_npu.k_dim_r),
    .total_mac_ops(total_mac_ops),
    .total_pe_cycles(total_pe_cycles),
    .total_busy_cycles(total_busy_cycles),
    .total_compute_cycles(total_compute_cycles),
    .total_dma_cycles(total_dma_cycles),
    .active_pe_cnt(active_pe_cnt),
    .peak_active_pe(peak_active_pe),
    .fsm_transitions(fsm_trans_cnt),
    .utilization_pct(util_pct),
    .mac_per_cycle(mac_per_cyc),
    .efficiency_pct(eff_pct)
);

// ===========================================================================
// AXI4-Lite BFM Tasks
// ===========================================================================
task axi_write;
    input [31:0] addr;
    input [31:0] data;
    integer cnt;
    begin
        s_awaddr  <= addr;  s_awvalid <= 1;
        s_wdata   <= data;  s_wstrb <= 4'hF; s_wvalid <= 1;
        s_bready  <= 1;
        @(posedge clk); // AW handshake cycle
        @(posedge clk); // wready should be high now
        s_awvalid <= 0; s_wvalid <= 0;
        // Wait for B
        cnt = 0;
        while (!s_bvalid && cnt < 50) begin
            @(posedge clk); cnt = cnt + 1;
        end
        s_bready <= 0;
        if (cnt >= 50)
            $display("    [BFM] W 0x%08h TIMEOUT!", addr);
        else
            $display("    [BFM] W 0x%08h = 0x%08h OK", addr, data);
    end
endtask

task axi_read;
    input  [31:0] addr;
    output [31:0] data;
    integer cnt;
    begin
        s_araddr <= addr; s_arvalid <= 1; s_rready <= 1;
        @(posedge clk); // AR handshake
        @(posedge clk); // rvalid should be high
        s_arvalid <= 0;
        cnt = 0;
        while (!s_rvalid && cnt < 50) begin
            @(posedge clk); cnt = cnt + 1;
        end
        data = s_rdata;
        s_rready <= 0;
    end
endtask

// ===========================================================================
// Statistics Print Task
// ===========================================================================
task print_report;
    input [127:0] label;
    integer avg_wr_lat, avg_rd_lat;
    begin
        avg_wr_lat = (s_axi_wr_cnt > 0) ? (s_axi_wr_lat / s_axi_wr_cnt) : 0;
        avg_rd_lat = (s_axi_rd_cnt > 0) ? (s_axi_rd_lat / s_axi_rd_cnt) : 0;
        $display("");
        $display("================================================================");
        $display("  NPU PERFORMANCE REPORT: %0s", label);
        $display("================================================================");
        $display("  [AXI4-Lite CPU Port]");
        $display("    Write Txns      : %6d    Read Txns      : %6d", s_axi_wr_cnt, s_axi_rd_cnt);
        $display("    Write Beats     : %6d    Read Beats     : %6d", s_axi_wr_beats, s_axi_rd_beats);
        $display("    Avg Wr Latency  : %6d cyc  Avg Rd Latency  : %6d cyc", avg_wr_lat, avg_rd_lat);
        $display("  [AXI4 Master DMA Port]");
        $display("    Write Bursts    : %6d    Read Bursts     : %6d", m_axi_wr_cnt, m_axi_rd_cnt);
        $display("    Write Beats     : %6d    Read Beats      : %6d", m_axi_wr_beats, m_axi_rd_beats);
        $display("    Write Bytes     : %6d    Read Bytes      : %6d", m_axi_wr_bytes, m_axi_rd_bytes);
        $display("    Rd Bandwidth    : %6d.%02d B/cyc", m_axi_rd_bw/1000, (m_axi_rd_bw%1000)/10);
        $display("    Wr Bandwidth    : %6d.%02d B/cyc", m_axi_wr_bw/1000, (m_axi_wr_bw%1000)/10);
        $display("  [NPU Compute]");
        $display("    Total MAC Ops   : %6d", total_mac_ops);
        $display("    PE Active Cycles: %6d    Busy Cycles    : %6d", total_pe_cycles, total_busy_cycles);
        $display("    Compute Cycles  : %6d    DMA Cycles     : %6d", total_compute_cycles, total_dma_cycles);
        $display("    Peak Active PEs : %6d / %0d", peak_active_pe, ROWS*COLS);
        $display("  [Performance]");
        $display("    PE Utilization  : %6d.%02d %%", util_pct/100, util_pct%100);
        $display("    MACs/Cycle      : %6d.%02d", mac_per_cyc/100, mac_per_cyc%100);
        $display("    Efficiency      : %6d.%02d %%", eff_pct/100, eff_pct%100);
        $display("    Total Cycles    : %6d", total_cycles);
        $display("================================================================");
        $display("");
    end
endtask

// ===========================================================================
// DRAM Init Helper
// ===========================================================================
task init_dram_seq;
    input [31:0] base;
    input [31:0] count;
    integer i;
    reg [31:0] val;
    begin
        for (i = 0; i < count; i = i + 1) begin
            val = i * 32'h01010101 + 32'h03020100;
            dram[(base >> 2) + i] = val;
        end
    end
endtask

// ===========================================================================
// Wait-for-done helper
// ===========================================================================
reg [31:0] rdata_tmp;

task wait_done;
    input [31:0] timeout_val;
    integer cnt;
    begin
        cnt = 0;
        while (cnt < timeout_val) begin
            @(posedge clk);
            axi_read(32'h04, rdata_tmp);
            if (rdata_tmp[1]) begin
                $display("    -> NPU DONE at cycle %0d", total_cycles);
                cnt = timeout_val; // break
            end
            cnt = cnt + 1;
        end
        if (rdata_tmp[1]) begin
            // Clear ctrl_reg to prevent re-trigger
            axi_write(32'h00, 32'h00);
        end else begin
            $display("    -> TIMEOUT after %0d cycles!", timeout_val);
        end
    end
endtask

// ===========================================================================
// TEST 1: INT8 4x4, Weight Stationary
// ===========================================================================
task test1;
    begin
        $display("\n--- TEST 1: INT8 4x4 WS ---");
        init_dram_seq(32'h0000, 16);   // weights  @ 0x0000
        init_dram_seq(32'h0100, 16);   // activations @ 0x0100

        axi_write(32'h10, 32'd4);   // M=4
        axi_write(32'h14, 32'd4);   // N=4
        axi_write(32'h18, 32'd4);   // K=4
        axi_write(32'h20, 32'h0);   // W addr
        axi_write(32'h24, 32'h100); // A addr
        axi_write(32'h28, 32'h200); // R addr
        axi_write(32'h00, 32'h01);  // start, INT8, WS
        wait_done(20000);
        print_report("INT8 4x4 WS");
    end
endtask

// ===========================================================================
// TEST 2: INT8 4x4, Output Stationary
// ===========================================================================
task test2;
    begin
        $display("\n--- TEST 2: INT8 4x4 OS ---");
        init_dram_seq(32'h1000, 16);  // weights  @ 0x1000
        init_dram_seq(32'h1100, 16);  // activations @ 0x1100

        axi_write(32'h10, 32'd4);
        axi_write(32'h14, 32'd4);
        axi_write(32'h18, 32'd4);
        axi_write(32'h20, 32'h1000);
        axi_write(32'h24, 32'h1100);
        axi_write(32'h28, 32'h1200);
        axi_write(32'h00, 32'h11);  // start, INT8, OS
        wait_done(20000);
        print_report("INT8 4x4 OS");
    end
endtask

// ===========================================================================
// TEST 3: INT8 8x8 tiled on 4x4 array
// ===========================================================================
task test3;
    begin
        $display("\n--- TEST 3: INT8 8x8 (tiled) ---");
        init_dram_seq(32'h2000, 64);  // weights  @ 0x2000
        init_dram_seq(32'h3000, 64);  // activations @ 0x3000

        axi_write(32'h10, 32'd8);   // M=8
        axi_write(32'h14, 32'd8);   // N=8
        axi_write(32'h18, 32'd8);   // K=8
        axi_write(32'h20, 32'h2000);
        axi_write(32'h24, 32'h3000);
        axi_write(32'h28, 32'h4000);
        axi_write(32'h00, 32'h01);
        wait_done(50000);
        print_report("INT8 8x8 Tiled");
    end
endtask

// ===========================================================================
// Main Sequence
// ===========================================================================
initial begin
    // Defaults
    s_awaddr<=0;s_awvalid<=0;s_wdata<=0;s_wstrb<=0;s_wvalid<=0;s_bready<=0;
    s_araddr<=0;s_arvalid<=0;s_rready<=0;
    m_awready<=1;m_wready<=0;

    `ifdef DUMP_VCD
    $dumpfile("tb_npu_top.vcd");
    $dumpvars(0, tb_npu_top);
    `endif

    @(posedge rst_n);
    @(posedge clk);
    $display("============================================================");
    $display("  NPU System Testbench  |  %0dx%0d Array  |  %0d-bit PE", ROWS, COLS, DATA_W);
    $display("============================================================");

    test1;
    #(CLK_T*100);

    $display("\n============================================================");
    $display("  ALL TESTS COMPLETE");
    $display("============================================================\n");
    $finish;
end

// Timeout watchdog
initial begin
    #(CLK_T * 200000);
    $display("\nFATAL: Global timeout!");
    $finish;
end

endmodule
