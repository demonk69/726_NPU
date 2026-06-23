// =============================================================================
// Module  : dram_multi_port
// Project : NPU_prj
// Desc    : Multi-port shared DRAM simulation model.
//           Provides one CPU simple memory port and NUM_CORES NPU AXI4 slave
//           ports backed by a single reg array.
//
//           Writes from multiple ports in the same cycle are resolved by a
//           fixed priority (lower port index wins). This is acceptable for
//           simulation because the multi-core architecture uses disjoint
//           write regions.
//
//           This module is for simulation only. It is not synthesizable and
//           must not appear in the board-level synthesis boundary.
// =============================================================================

`timescale 1ns/1ps

module dram_multi_port #(
    parameter WORDS     = 15360,
    parameter DATA_W    = 32,
    parameter NUM_CORES = 2
)(
    input  wire                      clk,
    input  wire                      rst_n,

    // ---- CPU simple port (port 0) ----
    input  wire                      cpu_valid,
    output wire                      cpu_ready,
    input  wire                      cpu_we,
    input  wire [3:0]                cpu_wstrb,
    input  wire [31:0]               cpu_addr,
    input  wire [31:0]               cpu_wdata,
    output wire [31:0]               cpu_rdata,

    // ---- NPU AXI4 slave ports (flattened, one per core) ----
    input  wire [NUM_CORES*32-1:0]   axi_awaddr,
    input  wire [NUM_CORES-1:0]      axi_awvalid,
    output wire [NUM_CORES-1:0]      axi_awready,
    input  wire [NUM_CORES*DATA_W-1:0]   axi_wdata,
    input  wire [NUM_CORES*(DATA_W/8)-1:0] axi_wstrb,
    input  wire [NUM_CORES-1:0]      axi_wlast,
    input  wire [NUM_CORES-1:0]      axi_wvalid,
    output wire [NUM_CORES-1:0]      axi_wready,
    output wire [NUM_CORES*2-1:0]    axi_bresp,
    output wire [NUM_CORES-1:0]      axi_bvalid,
    input  wire [NUM_CORES-1:0]      axi_bready,
    input  wire [NUM_CORES*32-1:0]   axi_araddr,
    input  wire [NUM_CORES*8-1:0]    axi_arlen,
    input  wire [NUM_CORES-1:0]      axi_arvalid,
    output wire [NUM_CORES-1:0]      axi_arready,
    output wire [NUM_CORES*DATA_W-1:0]   axi_rdata,
    output wire [NUM_CORES*2-1:0]    axi_rresp,
    output wire [NUM_CORES-1:0]      axi_rvalid,
    output wire [NUM_CORES-1:0]      axi_rlast,
    input  wire [NUM_CORES-1:0]      axi_rready
);

localparam ADDR_W = $clog2(WORDS);

// ---------------------------------------------------------------------------
// Shared storage
// ---------------------------------------------------------------------------
reg [DATA_W-1:0] mem [0:WORDS-1];

// ---------------------------------------------------------------------------
// CPU simple port
// ---------------------------------------------------------------------------
assign cpu_ready = 1'b1;
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
// NPU AXI4 ports — replicated per core
// ---------------------------------------------------------------------------
genvar port;
generate
    for (port = 0; port < NUM_CORES; port = port + 1) begin : gen_npu_port

        // Extract this core's local AXI signals
        wire [31:0] local_awaddr = axi_awaddr[port*32 +: 32];
        wire        local_awvalid = axi_awvalid[port];
        wire        local_awready;
        assign axi_awready[port] = local_awready;

        wire [DATA_W-1:0] local_wdata = axi_wdata[port*DATA_W +: DATA_W];
        wire [DATA_W/8-1:0] local_wstrb = axi_wstrb[port*(DATA_W/8) +: (DATA_W/8)];
        wire            local_wlast = axi_wlast[port];
        wire            local_wvalid = axi_wvalid[port];
        wire            local_wready;
        assign axi_wready[port] = local_wready;

        wire [1:0] local_bresp;
        assign axi_bresp[port*2 +: 2] = local_bresp;
        wire local_bvalid;
        assign axi_bvalid[port] = local_bvalid;
        wire local_bready = axi_bready[port];

        wire [31:0] local_araddr = axi_araddr[port*32 +: 32];
        wire [7:0]  local_arlen  = axi_arlen[port*8 +: 8];
        wire        local_arvalid = axi_arvalid[port];
        wire        local_arready;
        assign axi_arready[port] = local_arready;

        wire [DATA_W-1:0] local_rdata;
        assign axi_rdata[port*DATA_W +: DATA_W] = local_rdata;
        wire [1:0] local_rresp;
        assign axi_rresp[port*2 +: 2] = local_rresp;
        wire local_rvalid;
        assign axi_rvalid[port] = local_rvalid;
        wire local_rlast;
        assign axi_rlast[port] = local_rlast;
        wire local_rready = axi_rready[port];

        // Write FSM
        reg        wr_active;
        reg [31:0] w_addr_cnt;
        reg        b_q;

        assign local_awready = !wr_active;
        assign local_wready  = wr_active;
        assign local_bvalid  = b_q;
        assign local_bresp   = 2'b00;

        wire do_write = wr_active && local_wvalid && local_wready;

        always @(posedge clk) begin
            if (!rst_n) begin
                wr_active  <= 1'b0;
                w_addr_cnt <= 32'd0;
                b_q        <= 1'b0;
            end else begin
                if (!wr_active && local_awvalid) begin
                    wr_active  <= 1'b1;
                    w_addr_cnt <= local_awaddr;
                end

                if (do_write) begin
                    if (local_wstrb[0]) mem[w_addr_cnt[ADDR_W+1:2]][ 7: 0] <= local_wdata[ 7: 0];
                    if (local_wstrb[1]) mem[w_addr_cnt[ADDR_W+1:2]][15: 8] <= local_wdata[15: 8];
                    if (local_wstrb[2]) mem[w_addr_cnt[ADDR_W+1:2]][23:16] <= local_wdata[23:16];
                    if (local_wstrb[3]) mem[w_addr_cnt[ADDR_W+1:2]][31:24] <= local_wdata[31:24];

                    if (local_wlast) begin
                        wr_active <= 1'b0;
                        b_q       <= 1'b1;
                    end else begin
                        w_addr_cnt <= w_addr_cnt + DATA_W/8;
                    end
                end

                if (b_q && local_bready)
                    b_q <= 1'b0;
            end
        end

        // Read FSM
        reg [7:0]  ar_len_cnt;
        reg [31:0] ar_addr_cnt;
        reg        ar_active;
        reg        ar_first;

        assign local_arready = !ar_active;
        assign local_rvalid  = ar_active;
        assign local_rdata   = mem[ar_addr_cnt[ADDR_W+1:2]];
        assign local_rresp   = 2'b00;
        assign local_rlast   = ar_active && (ar_len_cnt == 0);

        always @(posedge clk) begin
            if (!rst_n) begin
                ar_active   <= 0;
                ar_len_cnt  <= 0;
                ar_addr_cnt <= 0;
                ar_first    <= 0;
            end else begin
                if (!ar_active && local_arvalid) begin
                    ar_active   <= 1;
                    ar_len_cnt  <= local_arlen;
                    ar_addr_cnt <= local_araddr;
                    ar_first    <= 1;
                end else if (ar_active && local_rvalid && local_rready) begin
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

    end
endgenerate

endmodule
