// =============================================================================
// Module  : router_local_adapter
// Project : NPU_prj
// Desc    : Local-port adapter for router_node_lite.
//
//           Router Local -> PE/Buffer:
//             Demux flits by data_type and expose payload/last on four local
//             valid-ready channels: activation, weight, psum, control.
//
//           PE/Buffer -> Router Local:
//             Priority-arbitrate local payload channels, pack one flit, and
//             present it to the router Local input. Control has highest
//             priority, then psum, weight, activation.
// =============================================================================

`timescale 1ns/1ps

module router_local_adapter #(
    parameter XW        = 4,
    parameter YW        = 4,
    parameter LANES     = 8,
    parameter LANE_W    = 16,
    parameter PAYLOAD_W = LANES * LANE_W,
    parameter FLIT_W    = PAYLOAD_W + 1 + (2 * XW) + (2 * YW) + 2
)(
    input  wire                 clk,
    input  wire                 rst_n,

    input  wire [XW-1:0]        local_x,
    input  wire [YW-1:0]        local_y,

    input  wire [XW-1:0]        act_dst_x,
    input  wire [YW-1:0]        act_dst_y,
    input  wire [XW-1:0]        weight_dst_x,
    input  wire [YW-1:0]        weight_dst_y,
    input  wire [XW-1:0]        psum_dst_x,
    input  wire [YW-1:0]        psum_dst_y,
    input  wire [XW-1:0]        ctrl_dst_x,
    input  wire [YW-1:0]        ctrl_dst_y,

    // Router Local output -> local PE/Buffer sink.
    input  wire                 rtr_rx_valid,
    output wire                 rtr_rx_ready,
    input  wire [FLIT_W-1:0]    rtr_rx_data,

    output wire                 act_rx_valid,
    input  wire                 act_rx_ready,
    output wire [PAYLOAD_W-1:0] act_rx_payload,
    output wire                 act_rx_last,

    output wire                 weight_rx_valid,
    input  wire                 weight_rx_ready,
    output wire [PAYLOAD_W-1:0] weight_rx_payload,
    output wire                 weight_rx_last,

    output wire                 psum_rx_valid,
    input  wire                 psum_rx_ready,
    output wire [PAYLOAD_W-1:0] psum_rx_payload,
    output wire                 psum_rx_last,

    output wire                 ctrl_rx_valid,
    input  wire                 ctrl_rx_ready,
    output wire [PAYLOAD_W-1:0] ctrl_rx_payload,
    output wire                 ctrl_rx_last,

    // Local PE/Buffer source -> Router Local input.
    input  wire                 act_tx_valid,
    output wire                 act_tx_ready,
    input  wire [PAYLOAD_W-1:0] act_tx_payload,
    input  wire                 act_tx_last,

    input  wire                 weight_tx_valid,
    output wire                 weight_tx_ready,
    input  wire [PAYLOAD_W-1:0] weight_tx_payload,
    input  wire                 weight_tx_last,

    input  wire                 psum_tx_valid,
    output wire                 psum_tx_ready,
    input  wire [PAYLOAD_W-1:0] psum_tx_payload,
    input  wire                 psum_tx_last,

    input  wire                 ctrl_tx_valid,
    output wire                 ctrl_tx_ready,
    input  wire [PAYLOAD_W-1:0] ctrl_tx_payload,
    input  wire                 ctrl_tx_last,

    output wire                 rtr_tx_valid,
    input  wire                 rtr_tx_ready,
    output wire [FLIT_W-1:0]    rtr_tx_data
);

localparam TYPE_ACT    = 2'b00;
localparam TYPE_WEIGHT = 2'b01;
localparam TYPE_PSUM   = 2'b10;
localparam TYPE_CTRL   = 2'b11;

localparam LAST_BIT  = PAYLOAD_W;
localparam SRC_Y_LSB = LAST_BIT + 1;
localparam SRC_X_LSB = SRC_Y_LSB + YW;
localparam DST_Y_LSB = SRC_X_LSB + XW;
localparam DST_X_LSB = DST_Y_LSB + YW;
localparam TYPE_LSB  = DST_X_LSB + XW;

wire [1:0] rx_type    = rtr_rx_data[TYPE_LSB +: 2];
wire       rx_last    = rtr_rx_data[LAST_BIT];
wire [PAYLOAD_W-1:0] rx_payload = rtr_rx_data[PAYLOAD_W-1:0];

reg                 act_rx_valid_q;
reg [PAYLOAD_W-1:0] act_rx_payload_q;
reg                 act_rx_last_q;
reg                 weight_rx_valid_q;
reg [PAYLOAD_W-1:0] weight_rx_payload_q;
reg                 weight_rx_last_q;
reg                 psum_rx_valid_q;
reg [PAYLOAD_W-1:0] psum_rx_payload_q;
reg                 psum_rx_last_q;
reg                 ctrl_rx_valid_q;
reg [PAYLOAD_W-1:0] ctrl_rx_payload_q;
reg                 ctrl_rx_last_q;

wire act_rx_can_accept    = !act_rx_valid_q    || act_rx_ready;
wire weight_rx_can_accept = !weight_rx_valid_q || weight_rx_ready;
wire psum_rx_can_accept   = !psum_rx_valid_q   || psum_rx_ready;
wire ctrl_rx_can_accept   = !ctrl_rx_valid_q   || ctrl_rx_ready;

assign rtr_rx_ready = (rx_type == TYPE_ACT)    ? act_rx_can_accept :
                      (rx_type == TYPE_WEIGHT) ? weight_rx_can_accept :
                      (rx_type == TYPE_PSUM)   ? psum_rx_can_accept :
                                                ctrl_rx_can_accept;

assign act_rx_valid     = act_rx_valid_q;
assign act_rx_payload   = act_rx_payload_q;
assign act_rx_last      = act_rx_last_q;
assign weight_rx_valid  = weight_rx_valid_q;
assign weight_rx_payload= weight_rx_payload_q;
assign weight_rx_last   = weight_rx_last_q;
assign psum_rx_valid    = psum_rx_valid_q;
assign psum_rx_payload  = psum_rx_payload_q;
assign psum_rx_last     = psum_rx_last_q;
assign ctrl_rx_valid    = ctrl_rx_valid_q;
assign ctrl_rx_payload  = ctrl_rx_payload_q;
assign ctrl_rx_last     = ctrl_rx_last_q;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        act_rx_valid_q     <= 1'b0;
        act_rx_payload_q   <= {PAYLOAD_W{1'b0}};
        act_rx_last_q      <= 1'b0;
        weight_rx_valid_q  <= 1'b0;
        weight_rx_payload_q<= {PAYLOAD_W{1'b0}};
        weight_rx_last_q   <= 1'b0;
        psum_rx_valid_q    <= 1'b0;
        psum_rx_payload_q  <= {PAYLOAD_W{1'b0}};
        psum_rx_last_q     <= 1'b0;
        ctrl_rx_valid_q    <= 1'b0;
        ctrl_rx_payload_q  <= {PAYLOAD_W{1'b0}};
        ctrl_rx_last_q     <= 1'b0;
    end else begin
        if (act_rx_valid_q && act_rx_ready)
            act_rx_valid_q <= 1'b0;
        if (weight_rx_valid_q && weight_rx_ready)
            weight_rx_valid_q <= 1'b0;
        if (psum_rx_valid_q && psum_rx_ready)
            psum_rx_valid_q <= 1'b0;
        if (ctrl_rx_valid_q && ctrl_rx_ready)
            ctrl_rx_valid_q <= 1'b0;

        if (rtr_rx_valid && rtr_rx_ready) begin
            case (rx_type)
                TYPE_ACT: begin
                    act_rx_valid_q   <= 1'b1;
                    act_rx_payload_q <= rx_payload;
                    act_rx_last_q    <= rx_last;
                end
                TYPE_WEIGHT: begin
                    weight_rx_valid_q   <= 1'b1;
                    weight_rx_payload_q <= rx_payload;
                    weight_rx_last_q    <= rx_last;
                end
                TYPE_PSUM: begin
                    psum_rx_valid_q   <= 1'b1;
                    psum_rx_payload_q <= rx_payload;
                    psum_rx_last_q    <= rx_last;
                end
                default: begin
                    ctrl_rx_valid_q   <= 1'b1;
                    ctrl_rx_payload_q <= rx_payload;
                    ctrl_rx_last_q    <= rx_last;
                end
            endcase
        end
    end
end

function [FLIT_W-1:0] pack_flit;
    input [1:0]            data_type;
    input [XW-1:0]         dst_x;
    input [YW-1:0]         dst_y;
    input                  last;
    input [PAYLOAD_W-1:0]  payload;
    begin
        pack_flit = {FLIT_W{1'b0}};
        pack_flit[PAYLOAD_W-1:0] = payload;
        pack_flit[LAST_BIT] = last;
        pack_flit[SRC_Y_LSB +: YW] = local_y;
        pack_flit[SRC_X_LSB +: XW] = local_x;
        pack_flit[DST_Y_LSB +: YW] = dst_y;
        pack_flit[DST_X_LSB +: XW] = dst_x;
        pack_flit[TYPE_LSB +: 2] = data_type;
    end
endfunction

reg                 tx_valid_q;
reg [FLIT_W-1:0]    tx_data_q;
reg [FLIT_W-1:0]    selected_flit;
reg [3:0]           selected_ready;
reg                 selected_valid;

wire tx_can_load = !tx_valid_q || rtr_tx_ready;

always @(*) begin
    selected_valid = 1'b0;
    selected_flit  = {FLIT_W{1'b0}};
    selected_ready = 4'b0000;

    if (ctrl_tx_valid) begin
        selected_valid = 1'b1;
        selected_flit = pack_flit(TYPE_CTRL, ctrl_dst_x, ctrl_dst_y,
                                  ctrl_tx_last, ctrl_tx_payload);
        selected_ready[3] = 1'b1;
    end else if (psum_tx_valid) begin
        selected_valid = 1'b1;
        selected_flit = pack_flit(TYPE_PSUM, psum_dst_x, psum_dst_y,
                                  psum_tx_last, psum_tx_payload);
        selected_ready[2] = 1'b1;
    end else if (weight_tx_valid) begin
        selected_valid = 1'b1;
        selected_flit = pack_flit(TYPE_WEIGHT, weight_dst_x, weight_dst_y,
                                  weight_tx_last, weight_tx_payload);
        selected_ready[1] = 1'b1;
    end else if (act_tx_valid) begin
        selected_valid = 1'b1;
        selected_flit = pack_flit(TYPE_ACT, act_dst_x, act_dst_y,
                                  act_tx_last, act_tx_payload);
        selected_ready[0] = 1'b1;
    end
end

assign act_tx_ready    = tx_can_load && selected_ready[0];
assign weight_tx_ready = tx_can_load && selected_ready[1];
assign psum_tx_ready   = tx_can_load && selected_ready[2];
assign ctrl_tx_ready   = tx_can_load && selected_ready[3];

assign rtr_tx_valid = tx_valid_q;
assign rtr_tx_data  = tx_data_q;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        tx_valid_q <= 1'b0;
        tx_data_q  <= {FLIT_W{1'b0}};
    end else if (tx_can_load) begin
        tx_valid_q <= selected_valid;
        if (selected_valid)
            tx_data_q <= selected_flit;
        else
            tx_data_q <= {FLIT_W{1'b0}};
    end
end

endmodule
