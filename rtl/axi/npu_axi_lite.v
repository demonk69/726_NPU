// =============================================================================
// Module  : npu_axi_lite
// Project : NPU_prj
// Desc    : AXI4-Lite slave with register file for NPU configuration.
//           Address map:
//             0x00  CTRL      - bit0=start, bit1=abort, [3:2]=data_mode, [5:4]=stat_mode
//             0x04  STATUS    - bit0=busy, bit1=done
//             0x08  INT_EN    - interrupt enable
//             0x0C  INT_CLR   - interrupt clear (write-1-to-clear)
//             0x10  M_DIM     - matrix M dimension
//             0x14  N_DIM     - matrix N dimension
//             0x18  K_DIM     - matrix K dimension
//             0x20  W_ADDR    - weight base address in DRAM
//             0x24  A_ADDR    - activation base address in DRAM
//             0x28  R_ADDR    - result base address in DRAM
//             0x30  ARR_CFG   - [3:0]=act_rows, [7:4]=act_cols
//             0x34  CLK_DIV   - [2:0]=div_sel
//             0x38  CG_EN     - clock gating enable
// =============================================================================

`timescale 1ns/1ps

module npu_axi_lite (
    // AXI4-Lite slave
    input  wire                  aclk,
    input  wire                  aresetn,
    // AW channel
    input  wire [31:0]           awaddr,
    input  wire                  awvalid,
    output wire                  awready,
    // W channel
    input  wire [31:0]           wdata,
    input  wire [3:0]            wstrb,
    input  wire                  wvalid,
    output wire                  wready,
    // B channel
    output wire [1:0]            bresp,
    output wire                  bvalid,
    input  wire                  bready,
    // AR channel
    input  wire [31:0]           araddr,
    input  wire                  arvalid,
    output wire                  arready,
    // R channel
    output wire [31:0]           rdata,
    output wire [1:0]            rresp,
    output wire                  rvalid,
    input  wire                  rready,
    // Control interface to NPU controller
    output reg  [31:0]           ctrl_reg,
    output reg  [31:0]           m_dim,
    output reg  [31:0]           n_dim,
    output reg  [31:0]           k_dim,
    output reg  [31:0]           w_addr,
    output reg  [31:0]           a_addr,
    output reg  [31:0]           r_addr,
    output reg  [7:0]            arr_cfg,    // [3:0]=rows, [7:4]=cols
    output reg  [2:0]            clk_div,
    output reg                   cg_en,
    // Status from NPU controller
    input  wire                  status_busy,
    input  wire                  status_done,
    input  wire                  irq_flag,
    // Interrupt output
    output wire                  npu_irq
);

// ---------------------------------------------------------------------------
// Internal registers
// ---------------------------------------------------------------------------
reg [31:0] int_en_reg;
reg [31:0] int_clr_reg;
reg        int_pending;

// ---------------------------------------------------------------------------
// Write FSM
// ---------------------------------------------------------------------------
reg        aw_q, ar_q;
reg [31:0] awaddr_q, araddr_q;
wire       wr_en = aw_q && wvalid && wready;
wire [3:0] w_strb = wstrb;

assign awready = !aw_q;
assign wready  = aw_q;
assign bvalid  = aw_q && wvalid && wready;
assign bresp   = 2'b00; // OKAY

always @(posedge aclk) begin
    if (!aresetn) aw_q <= 0;
    else if (awvalid && !aw_q) aw_q <= 1;
    else if (wr_en) aw_q <= 0;
end

always @(posedge aclk) begin
    if (awvalid && !aw_q) awaddr_q <= awaddr;
end

// Write data to register file
always @(posedge aclk) begin
    if (!aresetn) begin
        ctrl_reg  <= 0;
        int_en_reg <= 0;
        m_dim     <= 0;
        n_dim     <= 0;
        k_dim     <= 0;
        w_addr    <= 0;
        a_addr    <= 0;
        r_addr    <= 0;
        arr_cfg   <= 0;
        clk_div   <= 0;
        cg_en     <= 0;
    end else if (wr_en) begin
        case (awaddr_q)
            32'h00: begin
                if (w_strb[0]) begin
                    // $display("[AXI-WR] addr=%h wdata=%h (prev ctrl=%h)", awaddr_q, wdata, ctrl_reg);
                    ctrl_reg <= wdata;
                end
            end
            32'h08: if (w_strb[0]) int_en_reg <= wdata;
            32'h0C: int_clr_reg  <= wdata;  // W1C handled below
            32'h10: if (w_strb[0]) m_dim   <= wdata;
            32'h14: if (w_strb[0]) n_dim   <= wdata;
            32'h18: if (w_strb[0]) k_dim   <= wdata;
            32'h20: if (w_strb[0]) w_addr  <= wdata;
            32'h24: if (w_strb[0]) a_addr  <= wdata;
            32'h28: if (w_strb[0]) r_addr  <= wdata;
            32'h30: if (w_strb[0]) arr_cfg <= wdata[7:0];
            32'h34: if (w_strb[0]) clk_div <= wdata[2:0];
            32'h38: if (w_strb[0]) cg_en   <= wdata[0];
            default: ;
        endcase
    end
end

// ---------------------------------------------------------------------------
// Read FSM
// ---------------------------------------------------------------------------
assign arready = !ar_q;
assign rvalid  = ar_q;
assign rresp   = 2'b00;

always @(posedge aclk) begin
    if (!aresetn) ar_q <= 0;
    else if (arvalid && !ar_q) ar_q <= 1;
    else if (rvalid && rready) ar_q <= 0;
end

always @(posedge aclk) begin
    if (arvalid && !ar_q) araddr_q <= araddr;
end

reg [31:0] rdata_r;
always @(*) begin
    case (araddr_q)
        32'h00: rdata_r = ctrl_reg;
        32'h04: rdata_r = {30'b0, status_done, status_busy};
        32'h08: rdata_r = int_en_reg;
        32'h0C: rdata_r = int_pending;
        32'h10: rdata_r = m_dim;
        32'h14: rdata_r = n_dim;
        32'h18: rdata_r = k_dim;
        32'h20: rdata_r = w_addr;
        32'h24: rdata_r = a_addr;
        32'h28: rdata_r = r_addr;
        32'h30: rdata_r = {24'b0, arr_cfg};
        32'h34: rdata_r = {29'b0, clk_div};
        32'h38: rdata_r = {31'b0, cg_en};
        default: rdata_r = 32'hDEADBEEF;
    endcase
end
assign rdata = rdata_r;

// ---------------------------------------------------------------------------
// Interrupt logic
// ---------------------------------------------------------------------------
always @(posedge aclk) begin
    if (!aresetn)
        int_pending <= 0;
    else begin
        if (irq_flag && int_en_reg[0])
            int_pending <= 1'b1;
        if (wr_en && awaddr_q == 32'h0C && int_clr_reg[0])
            int_pending <= 1'b0;
    end
end
assign npu_irq = int_pending;

endmodule
