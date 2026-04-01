// =============================================================================
// Module  : axi_monitor
// Project : NPU_prj
// Desc    : AXI bus bandwidth & utilization monitor.
//           Tracks read/write transactions, burst lengths, throughput, latency.
//           Monitors both AXI4-Lite (s_axi) and AXI4 Master (m_axi) ports.
//           Outputs statistics via debug registers.
// =============================================================================

`timescale 1ns/1ps

module axi_monitor #(
    parameter ACC_W = 32
)(
    input  wire        clk,
    input  wire        rst_n,
    // AXI4-Lite slave (CPU config port) - monitor
    input  wire        s_awvalid,
    input  wire        s_awready,
    input  wire        s_wvalid,
    input  wire        s_wready,
    input  wire        s_bvalid,
    input  wire        s_bready,
    input  wire        s_arvalid,
    input  wire        s_arready,
    input  wire        s_rvalid,
    input  wire        s_rready,
    // AXI4 Master (DMA port) - monitor
    input  wire [7:0]  m_awlen,
    input  wire        m_awvalid,
    input  wire        m_awready,
    input  wire        m_wlast,
    input  wire        m_wvalid,
    input  wire        m_wready,
    input  wire        m_bvalid,
    input  wire        m_bready,
    input  wire [7:0]  m_arlen,
    input  wire        m_arvalid,
    input  wire        m_arready,
    input  wire        m_rlast,
    input  wire        m_rvalid,
    input  wire        m_rready,
    // Statistics outputs (32-bit counters)
    output wire [31:0] s_axi_wr_cnt,     // s_axi write transactions
    output wire [31:0] s_axi_rd_cnt,     // s_axi read transactions
    output wire [31:0] s_axi_wr_beats,   // s_axi write data beats
    output wire [31:0] s_axi_rd_beats,   // s_axi read data beats
    output wire [31:0] s_axi_wr_lat,     // s_axi write latency sum (cycles)
    output wire [31:0] s_axi_rd_lat,     // s_axi read latency sum (cycles)
    output wire [31:0] m_axi_wr_cnt,     // m_axi write bursts
    output wire [31:0] m_axi_rd_cnt,     // m_axi read bursts
    output wire [31:0] m_axi_wr_bytes,   // m_axi total write bytes
    output wire [31:0] m_axi_rd_bytes,   // m_axi total read bytes
    output wire [31:0] m_axi_wr_beats,   // m_axi write data beats
    output wire [31:0] m_axi_rd_beats,   // m_axi read data beats
    output wire [31:0] m_axi_wr_lat,     // m_axi write latency sum
    output wire [31:0] m_axi_rd_lat,     // m_axi read latency sum
    output wire [31:0] total_cycles,     // monitoring duration
    // Derived metrics (computed each cycle)
    output wire [31:0] m_axi_rd_bw,      // read bandwidth (bytes/cycle)
    output wire [31:0] m_axi_wr_bw       // write bandwidth (bytes/cycle)
);

// ---------------------------------------------------------------------------
// AXI4-Lite Slave Monitor
// ---------------------------------------------------------------------------
reg [31:0] s_wr_cnt_r, s_rd_cnt_r;
reg [31:0] s_wr_beats_r, s_rd_beats_r;
reg [31:0] s_wr_lat_r, s_rd_lat_r;
reg [7:0]  s_wr_lat_cnt, s_rd_lat_cnt;

// Write transaction tracking
always @(posedge clk) begin
    if (!rst_n) begin
        s_wr_cnt_r  <= 0;
        s_wr_beats_r <= 0;
        s_wr_lat_r <= 0;
        s_wr_lat_cnt <= 0;
    end else begin
        // Count write address handshake
        if (s_awvalid && s_awready)
            s_wr_cnt_r <= s_wr_cnt_r + 1;
        // Count write data beats
        if (s_wvalid && s_wready)
            s_wr_beats_r <= s_wr_beats_r + 1;
        // Write latency: AW handshake → B response
        if (s_awvalid && s_awready)
            s_wr_lat_cnt <= 8'd1;
        else if (s_wr_lat_cnt > 0 && !(s_bvalid && s_bready))
            s_wr_lat_cnt <= s_wr_lat_cnt + 1;
        if (s_bvalid && s_bready && s_wr_lat_cnt > 0) begin
            s_wr_lat_r <= s_wr_lat_r + {24'b0, s_wr_lat_cnt};
            s_wr_lat_cnt <= 0;
        end
    end
end

// Read transaction tracking
always @(posedge clk) begin
    if (!rst_n) begin
        s_rd_cnt_r  <= 0;
        s_rd_beats_r <= 0;
        s_rd_lat_r <= 0;
        s_rd_lat_cnt <= 0;
    end else begin
        if (s_arvalid && s_arready)
            s_rd_cnt_r <= s_rd_cnt_r + 1;
        if (s_rvalid && s_rready)
            s_rd_beats_r <= s_rd_beats_r + 1;
        if (s_arvalid && s_arready)
            s_rd_lat_cnt <= 8'd1;
        else if (s_rd_lat_cnt > 0 && !(s_rvalid && s_rready))
            s_rd_lat_cnt <= s_rd_lat_cnt + 1;
        if (s_rvalid && s_rready && s_rd_lat_cnt > 0) begin
            s_rd_lat_r <= s_rd_lat_r + {24'b0, s_rd_lat_cnt};
            s_rd_lat_cnt <= 0;
        end
    end
end

// ---------------------------------------------------------------------------
// AXI4 Master (DMA) Monitor
// ---------------------------------------------------------------------------
reg [31:0] m_wr_cnt_r, m_rd_cnt_r;
reg [31:0] m_wr_bytes_r, m_rd_bytes_r;
reg [31:0] m_wr_beats_r, m_rd_beats_r;
reg [31:0] m_wr_lat_r, m_rd_lat_r;
reg [7:0]  m_wr_lat_cnt, m_rd_lat_cnt;

// Write burst tracking
always @(posedge clk) begin
    if (!rst_n) begin
        m_wr_cnt_r   <= 0;
        m_wr_bytes_r <= 0;
        m_wr_beats_r <= 0;
        m_wr_lat_r   <= 0;
        m_wr_lat_cnt <= 0;
    end else begin
        // Write burst start
        if (m_awvalid && m_awready) begin
            m_wr_cnt_r <= m_wr_cnt_r + 1;
            m_wr_lat_cnt <= 8'd1;
        end
        // Write data beat
        if (m_wvalid && m_wready) begin
            m_wr_beats_r <= m_wr_beats_r + 1;
            m_wr_bytes_r <= m_wr_bytes_r + (ACC_W / 8);
        end
        // Write response latency
        if (m_awvalid && m_awready && m_wr_lat_cnt == 0)
            m_wr_lat_cnt <= 8'd1;
        else if (m_wr_lat_cnt > 0 && !(m_bvalid && m_bready))
            m_wr_lat_cnt <= m_wr_lat_cnt + 1;
        if (m_bvalid && m_bready && m_wr_lat_cnt > 0) begin
            m_wr_lat_r <= m_wr_lat_r + {24'b0, m_wr_lat_cnt};
            m_wr_lat_cnt <= 0;
        end
    end
end

// Read burst tracking
always @(posedge clk) begin
    if (!rst_n) begin
        m_rd_cnt_r   <= 0;
        m_rd_bytes_r <= 0;
        m_rd_beats_r <= 0;
        m_rd_lat_r   <= 0;
        m_rd_lat_cnt <= 0;
    end else begin
        if (m_arvalid && m_arready) begin
            m_rd_cnt_r <= m_rd_cnt_r + 1;
            m_rd_lat_cnt <= 8'd1;
        end
        if (m_rvalid && m_rready) begin
            m_rd_beats_r <= m_rd_beats_r + 1;
            m_rd_bytes_r <= m_rd_bytes_r + (ACC_W / 8);
        end
        if (m_arvalid && m_arready && m_rd_lat_cnt == 0)
            m_rd_lat_cnt <= 8'd1;
        else if (m_rd_lat_cnt > 0 && !(m_rvalid && m_rready && m_rlast))
            m_rd_lat_cnt <= m_rd_lat_cnt + 1;
        if (m_rvalid && m_rready && m_rlast && m_rd_lat_cnt > 0) begin
            m_rd_lat_r <= m_rd_lat_r + {24'b0, m_rd_lat_cnt};
            m_rd_lat_cnt <= 0;
        end
    end
end

// ---------------------------------------------------------------------------
// Total cycle counter
// ---------------------------------------------------------------------------
reg [31:0] cycle_cnt;
always @(posedge clk) begin
    if (!rst_n) cycle_cnt <= 0;
    else        cycle_cnt <= cycle_cnt + 1;
end

// ---------------------------------------------------------------------------
// Bandwidth computation (bytes per cycle, ×1000 for fixed-point)
// ---------------------------------------------------------------------------
assign m_axi_rd_bw = (cycle_cnt > 0) ? (m_rd_bytes_r * 1000 / cycle_cnt) : 0;
assign m_axi_wr_bw = (cycle_cnt > 0) ? (m_wr_bytes_r * 1000 / cycle_cnt) : 0;

// ---------------------------------------------------------------------------
// Output assignments
// ---------------------------------------------------------------------------
assign s_axi_wr_cnt   = s_wr_cnt_r;
assign s_axi_rd_cnt   = s_rd_cnt_r;
assign s_axi_wr_beats = s_wr_beats_r;
assign s_axi_rd_beats = s_rd_beats_r;
assign s_axi_wr_lat   = s_wr_lat_r;
assign s_axi_rd_lat   = s_rd_lat_r;

assign m_axi_wr_cnt   = m_wr_cnt_r;
assign m_axi_rd_cnt   = m_rd_cnt_r;
assign m_axi_wr_bytes = m_wr_bytes_r;
assign m_axi_rd_bytes = m_rd_bytes_r;
assign m_axi_wr_beats = m_wr_beats_r;
assign m_axi_rd_beats = m_rd_beats_r;
assign m_axi_wr_lat   = m_wr_lat_r;
assign m_axi_rd_lat   = m_rd_lat_r;

assign total_cycles   = cycle_cnt;

endmodule
