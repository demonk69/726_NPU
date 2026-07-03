// =============================================================================
// Module  : router_node_lite
// Project : NPU_prj
// Desc    : Lightweight 5-port routing node for a reconfigurable 2D PE mesh.
//
// Ports:
//   0 = North, 1 = South, 2 = West, 3 = East, 4 = Local
//
// Flit layout, LSB first:
//   payload[PAYLOAD_W-1:0]
//   last
//   src_y[YW-1:0], src_x[XW-1:0]
//   dst_y[YW-1:0], dst_x[XW-1:0]
//   data_type[1:0]
//
// data_type:
//   2'b00 activation, 2'b01 weight, 2'b10 psum, 2'b11 control
// =============================================================================

`timescale 1ns/1ps

module router_node_lite #(
    parameter XW        = 4,
    parameter YW        = 4,
    parameter LANES     = 8,
    parameter LANE_W    = 16,
    parameter PAYLOAD_W = LANES * LANE_W,
    parameter FLIT_W    = PAYLOAD_W + 1 + (2 * XW) + (2 * YW) + 2,
    parameter PORTS     = 5
)(
    input  wire                     clk,
    input  wire                     rst_n,

    input  wire [XW-1:0]            cur_x,
    input  wire [YW-1:0]            cur_y,

    // 0 = static route_cfg, 1 = dynamic XY route.
    input  wire                     route_mode_xy,
    input  wire [PORTS-1:0]         act_route_cfg,
    input  wire [PORTS-1:0]         weight_route_cfg,
    input  wire [PORTS-1:0]         psum_route_cfg,
    input  wire [PORTS-1:0]         ctrl_route_cfg,
    input  wire [PORTS-1:0]         port_enable_mask,

    input  wire [PORTS-1:0]         in_valid,
    output wire [PORTS-1:0]         in_ready,
    input  wire [PORTS*FLIT_W-1:0]  in_data,

    output wire [PORTS-1:0]         out_valid,
    input  wire [PORTS-1:0]         out_ready,
    output wire [PORTS*FLIT_W-1:0]  out_data
);

localparam DIR_N = 0;
localparam DIR_S = 1;
localparam DIR_W = 2;
localparam DIR_E = 3;
localparam DIR_L = 4;

wire [PORTS-1:0]        ibuf_valid;
wire [PORTS-1:0]        ibuf_ready;
wire [PORTS*FLIT_W-1:0] ibuf_data;
wire [PORTS*PORTS-1:0]  req_mask_flat;
wire [PORTS*PORTS-1:0]  grant_mask_flat;
wire [PORTS-1:0]        input_fire;
wire [PORTS-1:0]        out_stage_ready;
wire [PORTS-1:0]        xbar_valid;
wire [PORTS*FLIT_W-1:0] xbar_data;

genvar gp;
generate
    for (gp = 0; gp < PORTS; gp = gp + 1) begin : gen_input_buffer
        router_reg_slice #(
            .W(FLIT_W)
        ) u_input_buffer (
            .clk       (clk),
            .rst_n     (rst_n),
            .in_valid  (in_valid[gp]),
            .in_ready  (in_ready[gp]),
            .in_data   (in_data[gp*FLIT_W +: FLIT_W]),
            .out_valid (ibuf_valid[gp]),
            .out_ready (ibuf_ready[gp]),
            .out_data  (ibuf_data[gp*FLIT_W +: FLIT_W])
        );

        router_route_decode #(
            .XW(XW),
            .YW(YW),
            .PAYLOAD_W(PAYLOAD_W),
            .FLIT_W(FLIT_W),
            .PORTS(PORTS)
        ) u_route_decode (
            .cur_x           (cur_x),
            .cur_y           (cur_y),
            .route_mode_xy   (route_mode_xy),
            .act_route_cfg   (act_route_cfg),
            .weight_route_cfg(weight_route_cfg),
            .psum_route_cfg  (psum_route_cfg),
            .ctrl_route_cfg  (ctrl_route_cfg),
            .port_enable_mask(port_enable_mask),
            .flit            (ibuf_data[gp*FLIT_W +: FLIT_W]),
            .route_mask      (req_mask_flat[gp*PORTS +: PORTS])
        );

        assign ibuf_ready[gp] = input_fire[gp];
    end
endgenerate

// Atomic multicast issue stage. A selected input is released only when all
// target output register slices can accept the replicated flit. If route_cfg
// and port_enable_mask leave no legal output, the flit is dropped to avoid
// permanently wedging an edge node on illegal traffic.
router_multicast_arbiter_5x5 #(
    .PORTS(PORTS)
) u_multicast_arbiter (
    .clk        (clk),
    .rst_n      (rst_n),
    .in_valid   (ibuf_valid),
    .req_mask   (req_mask_flat),
    .out_ready  (out_stage_ready),
    .input_fire (input_fire),
    .grant_mask (grant_mask_flat)
);

// 5x5 crossbar. Grant masks are conflict-free: at most one input can drive any
// output in a cycle, while one selected input may drive multiple outputs.
generate
    for (gp = 0; gp < PORTS; gp = gp + 1) begin : gen_crossbar_out
        reg                 out_sel_valid;
        reg [FLIT_W-1:0]    out_sel_data;
        integer             gi;

        always @(*) begin
            out_sel_valid = 1'b0;
            out_sel_data  = {FLIT_W{1'b0}};
            for (gi = 0; gi < PORTS; gi = gi + 1) begin
                if (grant_mask_flat[gi*PORTS + gp]) begin
                    out_sel_valid = 1'b1;
                    out_sel_data  = ibuf_data[gi*FLIT_W +: FLIT_W];
                end
            end
        end

        assign xbar_valid[gp] = out_sel_valid;
        assign xbar_data[gp*FLIT_W +: FLIT_W] = out_sel_data;
    end
endgenerate

generate
    for (gp = 0; gp < PORTS; gp = gp + 1) begin : gen_output_stage
        router_reg_slice #(
            .W(FLIT_W)
        ) u_output_stage (
            .clk       (clk),
            .rst_n     (rst_n),
            .in_valid  (xbar_valid[gp]),
            .in_ready  (out_stage_ready[gp]),
            .in_data   (xbar_data[gp*FLIT_W +: FLIT_W]),
            .out_valid (out_valid[gp]),
            .out_ready (out_ready[gp]),
            .out_data  (out_data[gp*FLIT_W +: FLIT_W])
        );
    end
endgenerate

endmodule


module router_route_decode #(
    parameter XW        = 4,
    parameter YW        = 4,
    parameter PAYLOAD_W = 128,
    parameter FLIT_W    = PAYLOAD_W + 1 + (2 * XW) + (2 * YW) + 2,
    parameter PORTS     = 5
)(
    input  wire [XW-1:0]        cur_x,
    input  wire [YW-1:0]        cur_y,
    input  wire                 route_mode_xy,
    input  wire [PORTS-1:0]     act_route_cfg,
    input  wire [PORTS-1:0]     weight_route_cfg,
    input  wire [PORTS-1:0]     psum_route_cfg,
    input  wire [PORTS-1:0]     ctrl_route_cfg,
    input  wire [PORTS-1:0]     port_enable_mask,
    input  wire [FLIT_W-1:0]    flit,
    output reg  [PORTS-1:0]     route_mask
);

localparam DIR_N = 0;
localparam DIR_S = 1;
localparam DIR_W = 2;
localparam DIR_E = 3;
localparam DIR_L = 4;

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

wire [1:0]    data_type = flit[TYPE_LSB +: 2];
wire [XW-1:0] dst_x     = flit[DST_X_LSB +: XW];
wire [YW-1:0] dst_y     = flit[DST_Y_LSB +: YW];

reg [PORTS-1:0] static_mask;
reg [PORTS-1:0] xy_mask;

always @(*) begin
    case (data_type)
        TYPE_ACT:    static_mask = act_route_cfg;
        TYPE_WEIGHT: static_mask = weight_route_cfg;
        TYPE_PSUM:   static_mask = psum_route_cfg;
        TYPE_CTRL:   static_mask = ctrl_route_cfg;
        default:     static_mask = {PORTS{1'b0}};
    endcase

    xy_mask = {PORTS{1'b0}};
    if (dst_x > cur_x)
        xy_mask[DIR_E] = 1'b1;
    else if (dst_x < cur_x)
        xy_mask[DIR_W] = 1'b1;
    else if (dst_y > cur_y)
        xy_mask[DIR_S] = 1'b1;
    else if (dst_y < cur_y)
        xy_mask[DIR_N] = 1'b1;
    else
        xy_mask[DIR_L] = 1'b1;

    route_mask = (route_mode_xy ? xy_mask : static_mask) & port_enable_mask;
end

endmodule


module router_multicast_arbiter_5x5 #(
    parameter PORTS = 5
)(
    input  wire                     clk,
    input  wire                     rst_n,
    input  wire [PORTS-1:0]         in_valid,
    input  wire [PORTS*PORTS-1:0]   req_mask,
    input  wire [PORTS-1:0]         out_ready,
    output reg  [PORTS-1:0]         input_fire,
    output reg  [PORTS*PORTS-1:0]   grant_mask
);

reg [2:0] rr_ptr;
reg [2:0] last_sel;
reg       any_sel;

integer k;
integer idx;
reg [PORTS-1:0] used_outputs;
reg [PORTS-1:0] req_i;

always @(*) begin
    input_fire  = {PORTS{1'b0}};
    grant_mask  = {(PORTS*PORTS){1'b0}};
    used_outputs = {PORTS{1'b0}};
    last_sel = rr_ptr;
    any_sel = 1'b0;

    for (k = 0; k < PORTS; k = k + 1) begin
        idx = rr_ptr + k;
        if (idx >= PORTS)
            idx = idx - PORTS;

        req_i = req_mask[idx*PORTS +: PORTS];
        if (in_valid[idx] && (req_i == {PORTS{1'b0}})) begin
            input_fire[idx] = 1'b1;
            last_sel = idx[2:0];
            any_sel = 1'b1;
        end else if (in_valid[idx] &&
            ((req_i & ~out_ready) == {PORTS{1'b0}}) &&
            ((req_i & used_outputs) == {PORTS{1'b0}})) begin
            input_fire[idx] = 1'b1;
            grant_mask[idx*PORTS +: PORTS] = req_i;
            used_outputs = used_outputs | req_i;
            last_sel = idx[2:0];
            any_sel = 1'b1;
        end
    end
end

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        rr_ptr <= 3'd0;
    end else if (any_sel) begin
        if (last_sel == PORTS - 1)
            rr_ptr <= 3'd0;
        else
            rr_ptr <= last_sel + 3'd1;
    end
end

endmodule


module router_reg_slice #(
    parameter W = 128
)(
    input  wire         clk,
    input  wire         rst_n,
    input  wire         in_valid,
    output wire         in_ready,
    input  wire [W-1:0] in_data,
    output wire         out_valid,
    input  wire         out_ready,
    output wire [W-1:0] out_data
);

reg         full_q;
reg [W-1:0] data_q;

assign in_ready  = !full_q || out_ready;
assign out_valid = full_q;
assign out_data  = data_q;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        full_q <= 1'b0;
        data_q <= {W{1'b0}};
    end else if (in_ready) begin
        full_q <= in_valid;
        if (in_valid)
            data_q <= in_data;
    end
end

endmodule
