// =============================================================================
// Module  : router_mesh_lite
// Project : NPU_prj
// Desc    : Parameterized 2D mesh prototype built from router_node_lite.
//
//           The mesh exposes only each node's Local valid-ready-data port.
//           North/South/West/East ports are wired to neighboring router nodes.
//           Boundary directions are masked off per node through port_enable_mask.
// =============================================================================

`timescale 1ns/1ps

module router_mesh_lite #(
    parameter ROWS      = 4,
    parameter COLS      = 4,
    parameter XW        = 4,
    parameter YW        = 4,
    parameter LANES     = 8,
    parameter LANE_W    = 16,
    parameter PAYLOAD_W = LANES * LANE_W,
    parameter FLIT_W    = PAYLOAD_W + 1 + (2 * XW) + (2 * YW) + 2,
    parameter PORTS     = 5,
    parameter NODES     = ROWS * COLS
)(
    input  wire                     clk,
    input  wire                     rst_n,

    input  wire                     route_mode_xy,
    input  wire [PORTS-1:0]         act_route_cfg,
    input  wire [PORTS-1:0]         weight_route_cfg,
    input  wire [PORTS-1:0]         psum_route_cfg,
    input  wire [PORTS-1:0]         ctrl_route_cfg,

    input  wire [NODES-1:0]         local_in_valid,
    output wire [NODES-1:0]         local_in_ready,
    input  wire [NODES*FLIT_W-1:0]  local_in_data,

    output wire [NODES-1:0]         local_out_valid,
    input  wire [NODES-1:0]         local_out_ready,
    output wire [NODES*FLIT_W-1:0]  local_out_data
);

localparam DIR_N = 0;
localparam DIR_S = 1;
localparam DIR_W = 2;
localparam DIR_E = 3;
localparam DIR_L = 4;

wire [NODES*PORTS-1:0]          node_in_valid;
wire [NODES*PORTS-1:0]          node_in_ready;
wire [NODES*PORTS*FLIT_W-1:0]   node_in_data;
wire [NODES*PORTS-1:0]          node_out_valid;
wire [NODES*PORTS-1:0]          node_out_ready;
wire [NODES*PORTS*FLIT_W-1:0]   node_out_data;

genvar gy, gx;
generate
    for (gy = 0; gy < ROWS; gy = gy + 1) begin : gen_row
        for (gx = 0; gx < COLS; gx = gx + 1) begin : gen_col
            localparam integer NODE = gy * COLS + gx;
            localparam integer NORTH_NODE = (gy == 0) ? NODE : ((gy - 1) * COLS + gx);
            localparam integer SOUTH_NODE = (gy == ROWS - 1) ? NODE : ((gy + 1) * COLS + gx);
            localparam integer WEST_NODE  = (gx == 0) ? NODE : (gy * COLS + gx - 1);
            localparam integer EAST_NODE  = (gx == COLS - 1) ? NODE : (gy * COLS + gx + 1);
            localparam [XW-1:0] CUR_X = gx;
            localparam [YW-1:0] CUR_Y = gy;
            wire [PORTS-1:0] port_enable_mask;

            assign port_enable_mask[DIR_N] = (gy != 0);
            assign port_enable_mask[DIR_S] = (gy != ROWS - 1);
            assign port_enable_mask[DIR_W] = (gx != 0);
            assign port_enable_mask[DIR_E] = (gx != COLS - 1);
            assign port_enable_mask[DIR_L] = 1'b1;

            // Local input and output.
            assign node_in_valid[NODE*PORTS + DIR_L] = local_in_valid[NODE];
            assign node_in_data[(NODE*PORTS + DIR_L)*FLIT_W +: FLIT_W] =
                local_in_data[NODE*FLIT_W +: FLIT_W];
            assign local_in_ready[NODE] = node_in_ready[NODE*PORTS + DIR_L];

            assign local_out_valid[NODE] = node_out_valid[NODE*PORTS + DIR_L];
            assign local_out_data[NODE*FLIT_W +: FLIT_W] =
                node_out_data[(NODE*PORTS + DIR_L)*FLIT_W +: FLIT_W];
            assign node_out_ready[NODE*PORTS + DIR_L] = local_out_ready[NODE];

            // North input comes from the north neighbor's South output.
            if (gy == 0) begin : gen_no_north_in
                assign node_in_valid[NODE*PORTS + DIR_N] = 1'b0;
                assign node_in_data[(NODE*PORTS + DIR_N)*FLIT_W +: FLIT_W] = {FLIT_W{1'b0}};
                assign node_out_ready[NODE*PORTS + DIR_N] = 1'b1;
            end else begin : gen_north_link
                assign node_in_valid[NODE*PORTS + DIR_N] =
                    node_out_valid[NORTH_NODE*PORTS + DIR_S];
                assign node_in_data[(NODE*PORTS + DIR_N)*FLIT_W +: FLIT_W] =
                    node_out_data[(NORTH_NODE*PORTS + DIR_S)*FLIT_W +: FLIT_W];
                assign node_out_ready[NORTH_NODE*PORTS + DIR_S] =
                    node_in_ready[NODE*PORTS + DIR_N];
            end

            // South input comes from the south neighbor's North output.
            if (gy == ROWS - 1) begin : gen_no_south_in
                assign node_in_valid[NODE*PORTS + DIR_S] = 1'b0;
                assign node_in_data[(NODE*PORTS + DIR_S)*FLIT_W +: FLIT_W] = {FLIT_W{1'b0}};
                assign node_out_ready[NODE*PORTS + DIR_S] = 1'b1;
            end else begin : gen_south_link
                assign node_in_valid[NODE*PORTS + DIR_S] =
                    node_out_valid[SOUTH_NODE*PORTS + DIR_N];
                assign node_in_data[(NODE*PORTS + DIR_S)*FLIT_W +: FLIT_W] =
                    node_out_data[(SOUTH_NODE*PORTS + DIR_N)*FLIT_W +: FLIT_W];
                assign node_out_ready[SOUTH_NODE*PORTS + DIR_N] =
                    node_in_ready[NODE*PORTS + DIR_S];
            end

            // West input comes from the west neighbor's East output.
            if (gx == 0) begin : gen_no_west_in
                assign node_in_valid[NODE*PORTS + DIR_W] = 1'b0;
                assign node_in_data[(NODE*PORTS + DIR_W)*FLIT_W +: FLIT_W] = {FLIT_W{1'b0}};
                assign node_out_ready[NODE*PORTS + DIR_W] = 1'b1;
            end else begin : gen_west_link
                assign node_in_valid[NODE*PORTS + DIR_W] =
                    node_out_valid[WEST_NODE*PORTS + DIR_E];
                assign node_in_data[(NODE*PORTS + DIR_W)*FLIT_W +: FLIT_W] =
                    node_out_data[(WEST_NODE*PORTS + DIR_E)*FLIT_W +: FLIT_W];
                assign node_out_ready[WEST_NODE*PORTS + DIR_E] =
                    node_in_ready[NODE*PORTS + DIR_W];
            end

            // East input comes from the east neighbor's West output.
            if (gx == COLS - 1) begin : gen_no_east_in
                assign node_in_valid[NODE*PORTS + DIR_E] = 1'b0;
                assign node_in_data[(NODE*PORTS + DIR_E)*FLIT_W +: FLIT_W] = {FLIT_W{1'b0}};
                assign node_out_ready[NODE*PORTS + DIR_E] = 1'b1;
            end else begin : gen_east_link
                assign node_in_valid[NODE*PORTS + DIR_E] =
                    node_out_valid[EAST_NODE*PORTS + DIR_W];
                assign node_in_data[(NODE*PORTS + DIR_E)*FLIT_W +: FLIT_W] =
                    node_out_data[(EAST_NODE*PORTS + DIR_W)*FLIT_W +: FLIT_W];
                assign node_out_ready[EAST_NODE*PORTS + DIR_W] =
                    node_in_ready[NODE*PORTS + DIR_E];
            end

            router_node_lite #(
                .XW(XW),
                .YW(YW),
                .LANES(LANES),
                .LANE_W(LANE_W),
                .PAYLOAD_W(PAYLOAD_W),
                .FLIT_W(FLIT_W),
                .PORTS(PORTS)
            ) u_node (
                .clk(clk),
                .rst_n(rst_n),
                .cur_x(CUR_X),
                .cur_y(CUR_Y),
                .route_mode_xy(route_mode_xy),
                .act_route_cfg(act_route_cfg),
                .weight_route_cfg(weight_route_cfg),
                .psum_route_cfg(psum_route_cfg),
                .ctrl_route_cfg(ctrl_route_cfg),
                .port_enable_mask(port_enable_mask),
                .in_valid(node_in_valid[NODE*PORTS +: PORTS]),
                .in_ready(node_in_ready[NODE*PORTS +: PORTS]),
                .in_data(node_in_data[NODE*PORTS*FLIT_W +: PORTS*FLIT_W]),
                .out_valid(node_out_valid[NODE*PORTS +: PORTS]),
                .out_ready(node_out_ready[NODE*PORTS +: PORTS]),
                .out_data(node_out_data[NODE*PORTS*FLIT_W +: PORTS*FLIT_W])
            );
        end
    end
endgenerate

endmodule
