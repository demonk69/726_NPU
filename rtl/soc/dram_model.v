// =============================================================================
// Module  : dram_model
// Project : NPU_prj
// Desc    : Behavioral DRAM model with dual-port access.
//           Port 1 (CPU side): Simple read/write via valid/ready handshake.
//           Port 2 (NPU DMA side): AXI4 slave interface for burst transfers.
//
//           In simulation, both ports access the same storage array.
//           No real arbitration needed in simulation - timing is naturally
//           serialized by the testbench.
//
//           Address mapping:
//             0x0000_0100 - 0x0000_FFFF  (DRAM space visible to SoC)
//             Internally word-indexed: addr[21:2]
// =============================================================================

`timescale 1ns/1ps

module dram_model #(
    parameter WORDS     = 15360,     // ~60KB default
    parameter DATA_W    = 32
)(
    input  wire                      clk,
    input  wire                      rst_n,

    // ---- Port 1: CPU side (simple memory interface) ----
    input  wire                      cpu_valid,
    output wire                      cpu_ready,
    input  wire                      cpu_we,          // 1=write, 0=read
    input  wire [3:0]                cpu_wstrb,
    input  wire [31:0]               cpu_addr,
    input  wire [31:0]               cpu_wdata,
    output wire [31:0]               cpu_rdata,

    // ---- Port 2: NPU DMA side (AXI4 slave) ----
    // AW
    input  wire [31:0]               axi_awaddr,
    input  wire                      axi_awvalid,
    output wire                      axi_awready,
    // W
    input  wire [DATA_W-1:0]         axi_wdata,
    input  wire [DATA_W/8-1:0]       axi_wstrb,
    input  wire                      axi_wlast,
    input  wire                      axi_wvalid,
    output wire                      axi_wready,
    // B
    output wire [1:0]                axi_bresp,
    output wire                      axi_bvalid,
    input  wire                      axi_bready,
    // AR
    input  wire [31:0]               axi_araddr,
    input  wire [7:0]                axi_arlen,
    input  wire                      axi_arvalid,
    output wire                      axi_arready,
    // R
    output wire [DATA_W-1:0]         axi_rdata,
    output wire [1:0]                axi_rresp,
    output wire                      axi_rvalid,
    output wire                      axi_rlast,
    input  wire                      axi_rready
);

localparam ADDR_W = $clog2(WORDS);

// ---------------------------------------------------------------------------
// Shared storage
// ---------------------------------------------------------------------------
reg [DATA_W-1:0] mem [0:WORDS-1];

// ---------------------------------------------------------------------------
// Port 1: CPU read/write
// ---------------------------------------------------------------------------
assign cpu_ready = 1'b1;  // CPU port always ready (no backpressure in sim)
assign cpu_rdata = mem[cpu_addr[ADDR_W+1:2]];

always @(posedge clk) begin
    if (cpu_valid && cpu_we) begin
        if (cpu_wstrb[0]) mem[cpu_addr[ADDR_W+1:2]][ 7: 0] <= cpu_wdata[ 7: 0];
        if (cpu_wstrb[1]) mem[cpu_addr[ADDR_W+1:2]][15: 8] <= cpu_wdata[15: 8];
        if (cpu_wstrb[2]) mem[cpu_addr[ADDR_W+1:2]][23:16] <= cpu_wdata[23:16];
        if (cpu_wstrb[3]) mem[cpu_addr[ADDR_W+1:2]][31:24] <= cpu_wdata[31:24];
    end
end

// ---------------------------------------------------------------------------
// Port 2: AXI4 Write (AW + W + B)
// ---------------------------------------------------------------------------
reg        wr_active;
reg [31:0] w_addr_cnt;
reg        b_q;

assign axi_awready = !wr_active;
assign axi_wready  = wr_active;
assign axi_bvalid  = b_q;
assign axi_bresp   = 2'b00;

wire do_axi_write = wr_active && axi_wvalid && axi_wready;

always @(posedge clk) begin
    if (!rst_n) begin
        wr_active  <= 1'b0;
        w_addr_cnt <= 32'd0;
        b_q        <= 1'b0;
    end else begin
        if (!wr_active && axi_awvalid) begin
            wr_active  <= 1'b1;
            w_addr_cnt <= axi_awaddr;
        end

        if (do_axi_write) begin
            if (axi_wstrb[0]) mem[w_addr_cnt[ADDR_W+1:2]][ 7: 0] <= axi_wdata[ 7: 0];
            if (axi_wstrb[1]) mem[w_addr_cnt[ADDR_W+1:2]][15: 8] <= axi_wdata[15: 8];
            if (axi_wstrb[2]) mem[w_addr_cnt[ADDR_W+1:2]][23:16] <= axi_wdata[23:16];
            if (axi_wstrb[3]) mem[w_addr_cnt[ADDR_W+1:2]][31:24] <= axi_wdata[31:24];

            if (axi_wlast) begin
                wr_active <= 1'b0;
                b_q       <= 1'b1;
            end else begin
                w_addr_cnt <= w_addr_cnt + DATA_W/8;
            end
        end

        if (b_q && axi_bready)
            b_q <= 1'b0;
    end
end

// ---------------------------------------------------------------------------
// Port 2: AXI4 Read (AR + R)
// ---------------------------------------------------------------------------
reg [7:0] ar_len_cnt;
reg [31:0] ar_addr_cnt;
reg ar_active;
reg ar_first;

assign axi_arready = !ar_active;

always @(posedge clk) begin
    if (!rst_n) begin
        ar_active  <= 0;
        ar_len_cnt <= 0;
        ar_addr_cnt <= 0;
        ar_first   <= 0;
    end else begin
        if (!ar_active && axi_arvalid) begin
            ar_active  <= 1;
            ar_len_cnt <= axi_arlen;  // burst length - 1 already
            ar_addr_cnt <= axi_araddr;
            ar_first   <= 1;
        end else if (ar_active && axi_rvalid && axi_rready) begin
            if (ar_len_cnt == 0) begin
                ar_active <= 0;
            end else begin
                ar_len_cnt <= ar_len_cnt - 1;
                ar_addr_cnt <= ar_addr_cnt + DATA_W/8;
            end
            ar_first <= 0;
        end
    end
end

assign axi_rvalid = ar_active;
assign axi_rdata  = mem[ar_addr_cnt[ADDR_W+1:2]];
assign axi_rresp  = 2'b00;
assign axi_rlast  = ar_active && (ar_len_cnt == 0);

endmodule
