// =============================================================================
// Module  : router_pe_array_lite
// Project : NPU_prj
// Desc    : PE-array-style wrapper around router_pe_mesh_lite.
//
//           Boundary interface:
//             - act_in[r] enters the west edge node of row r and broadcasts east
//             - weight_in[c] enters the north edge node of column c and broadcasts south
//
//           This is still a standalone prototype. It is intended as the next
//           integration step toward replacing direct PE-array feeders with a
//           router-mesh-backed fabric.
// =============================================================================

`timescale 1ns/1ps

module router_pe_array_lite #(
    parameter ROWS      = 4,
    parameter COLS      = 4,
    parameter XW        = 4,
    parameter YW        = 4,
    parameter LANES     = 8,
    parameter LANE_W    = 16,
    parameter PE_DATA_W = 64,
    parameter ACC_W     = 32,
    parameter INT8_SIMD_LANES = (PE_DATA_W >= 64) ? 8 : ((PE_DATA_W >= 32) ? 4 : 2),
    parameter FP16_ENABLE = 0,
    parameter INT8_SCALAR_SIGNEXT_COMPAT = 0,
    parameter AUTO_FLUSH_ON_COMPUTE = 1,
    parameter PE_STREAM_BUF_DEPTH = 4,
    parameter PAYLOAD_W = LANES * LANE_W,
    parameter FLIT_W    = PAYLOAD_W + 1 + (2 * XW) + (2 * YW) + 2,
    parameter PORTS     = 5,
    parameter NODES     = ROWS * COLS
)(
    input  wire                      clk,
    input  wire                      rst_n,

    input  wire                      flush,
    input  wire                      mode,
    input  wire                      stat_mode,
    input  wire                      load_w,
    input  wire                      swap_w,
    input  wire                      ws_direct,
    input  wire [3:0]                ws_load_row,

    input  wire [ROWS-1:0]           act_valid,
    output wire [ROWS-1:0]           act_ready,
    input  wire [ROWS*PE_DATA_W-1:0] act_data,

    input  wire [COLS-1:0]           weight_valid,
    output wire [COLS-1:0]           weight_ready,
    input  wire [COLS*PE_DATA_W-1:0] weight_data,

    output wire [NODES-1:0]          pe_valid,
    output wire [NODES-1:0]          pe_compute_fire,
    output wire [NODES*ACC_W-1:0]    pe_acc_out
);

localparam DIR_N = 0;
localparam DIR_S = 1;
localparam DIR_W = 2;
localparam DIR_E = 3;
localparam DIR_L = 4;

reg  [NODES-1:0]           mesh_act_valid;
wire [NODES-1:0]           mesh_act_ready;
reg  [NODES*PAYLOAD_W-1:0] mesh_act_payload;
reg  [NODES-1:0]           mesh_act_last;
reg  [NODES-1:0]           mesh_weight_valid;
wire [NODES-1:0]           mesh_weight_ready;
reg  [NODES*PAYLOAD_W-1:0] mesh_weight_payload;
reg  [NODES-1:0]           mesh_weight_last;
reg  [NODES*XW-1:0]        mesh_act_dst_x_vec;
reg  [NODES*YW-1:0]        mesh_act_dst_y_vec;
reg  [NODES*XW-1:0]        mesh_weight_dst_x_vec;
reg  [NODES*YW-1:0]        mesh_weight_dst_y_vec;
wire [NODES-1:0]           mesh_psum_valid;
wire [NODES-1:0]           mesh_psum_ready;
wire [NODES*PAYLOAD_W-1:0] mesh_psum_payload;
wire [NODES-1:0]           mesh_psum_last;
wire [NODES-1:0]           mesh_ctrl_valid;
wire [NODES-1:0]           mesh_ctrl_ready;
wire [NODES*PAYLOAD_W-1:0] mesh_ctrl_payload;
wire [NODES-1:0]           mesh_ctrl_last;

/* verilator lint_off WIDTHCONCAT */
assign mesh_psum_valid = {NODES{1'b0}};
assign mesh_psum_payload = '0;
assign mesh_psum_last = {NODES{1'b1}};
assign mesh_ctrl_valid = {NODES{1'b0}};
assign mesh_ctrl_payload = '0;
assign mesh_ctrl_last = {NODES{1'b1}};

integer ai;
integer wi;
integer node_i;
wire true_ws_mode = !stat_mode && !ws_direct;

always @(*) begin
    node_i = 0;
    mesh_act_valid = {NODES{1'b0}};
    mesh_act_payload = '0;
    mesh_act_last = {NODES{flush}};
    mesh_act_dst_x_vec = {NODES*XW{1'b0}};
    mesh_act_dst_y_vec = {NODES*YW{1'b0}};

    mesh_weight_valid = {NODES{1'b0}};
    mesh_weight_payload = '0;
    mesh_weight_last = {NODES{flush}};
    mesh_weight_dst_x_vec = {NODES*XW{1'b0}};
    mesh_weight_dst_y_vec = {NODES*YW{1'b0}};

    if (true_ws_mode) begin
        mesh_act_last = {NODES{1'b0}};
        mesh_weight_last = {NODES{1'b0}};

        for (ai = 0; ai < ROWS; ai = ai + 1) begin
            for (wi = 0; wi < COLS; wi = wi + 1) begin
                node_i = ai * COLS + wi;
                mesh_act_valid[node_i] = act_valid[ai];
                mesh_act_payload[node_i * PAYLOAD_W +: PE_DATA_W] =
                    act_data[ai*PE_DATA_W +: PE_DATA_W];
                mesh_act_dst_x_vec[node_i*XW +: XW] = wi[XW-1:0];
                mesh_act_dst_y_vec[node_i*YW +: YW] = ai[YW-1:0];
            end
        end

        for (ai = 0; ai < ROWS; ai = ai + 1) begin
            for (wi = 0; wi < COLS; wi = wi + 1) begin
                node_i = ai * COLS + wi;
                mesh_weight_dst_x_vec[node_i*XW +: XW] = wi[XW-1:0];
                mesh_weight_dst_y_vec[node_i*YW +: YW] = ai[YW-1:0];
                if (ai[3:0] == ws_load_row) begin
                    mesh_weight_valid[node_i] = load_w && weight_valid[wi];
                    mesh_weight_payload[node_i * PAYLOAD_W +: PE_DATA_W] =
                        weight_data[wi*PE_DATA_W +: PE_DATA_W];
                end
            end
        end
    end else begin
        for (ai = 0; ai < ROWS; ai = ai + 1) begin
            mesh_act_valid[ai * COLS] = act_valid[ai];
            mesh_act_payload[(ai * COLS) * PAYLOAD_W +: PE_DATA_W] =
                act_data[ai*PE_DATA_W +: PE_DATA_W];
        end

        for (wi = 0; wi < COLS; wi = wi + 1) begin
            mesh_weight_valid[wi] = weight_valid[wi];
            mesh_weight_payload[wi * PAYLOAD_W +: PE_DATA_W] =
                weight_data[wi*PE_DATA_W +: PE_DATA_W];
        end
    end
end
/* verilator lint_on WIDTHCONCAT */

genvar r, c;
generate
    for (r = 0; r < ROWS; r = r + 1) begin : gen_act_edge
        localparam integer NODE = r * COLS;
        assign act_ready[r] = true_ws_mode ? (&mesh_act_ready[r*COLS +: COLS])
                                           : mesh_act_ready[NODE];
    end

    for (c = 0; c < COLS; c = c + 1) begin : gen_weight_edge
        localparam integer NODE = c;
        assign weight_ready[c] = true_ws_mode
                               ? ((ws_load_row < ROWS) ? mesh_weight_ready[ws_load_row*COLS + c]
                                                       : 1'b1)
                               : mesh_weight_ready[NODE];
    end
endgenerate

router_pe_mesh_lite #(
    .ROWS(ROWS),
    .COLS(COLS),
    .XW(XW),
    .YW(YW),
    .LANES(LANES),
    .LANE_W(LANE_W),
    .PE_DATA_W(PE_DATA_W),
    .ACC_W(ACC_W),
    .INT8_SIMD_LANES(INT8_SIMD_LANES),
    .FP16_ENABLE(FP16_ENABLE),
    .INT8_SCALAR_SIGNEXT_COMPAT(INT8_SCALAR_SIGNEXT_COMPAT),
    .AUTO_FLUSH_ON_COMPUTE(AUTO_FLUSH_ON_COMPUTE),
    .PE_STREAM_BUF_DEPTH(PE_STREAM_BUF_DEPTH),
    .PAYLOAD_W(PAYLOAD_W),
    .FLIT_W(FLIT_W),
    .PORTS(PORTS),
    .NODES(NODES)
) u_router_pe_mesh (
    .clk(clk),
    .rst_n(rst_n),
    .mode(mode),
    .stat_mode(stat_mode || ws_direct),
    .swap_w(swap_w),
    .ws_direct(ws_direct),
    .route_mode_xy(true_ws_mode),
    .per_node_dst(true_ws_mode),
    .act_route_cfg((5'b00001 << DIR_L) | (5'b00001 << DIR_E)),
    .weight_route_cfg((5'b00001 << DIR_L) | (5'b00001 << DIR_S)),
    .psum_route_cfg(5'b00001 << DIR_L),
    .ctrl_route_cfg(5'b00001 << DIR_L),
    .act_dst_x({XW{1'b0}}),
    .act_dst_y({YW{1'b0}}),
    .weight_dst_x({XW{1'b0}}),
    .weight_dst_y({YW{1'b0}}),
    .psum_dst_x({XW{1'b0}}),
    .psum_dst_y({YW{1'b0}}),
    .ctrl_dst_x({XW{1'b0}}),
    .ctrl_dst_y({YW{1'b0}}),
    .act_dst_x_vec(mesh_act_dst_x_vec),
    .act_dst_y_vec(mesh_act_dst_y_vec),
    .weight_dst_x_vec(mesh_weight_dst_x_vec),
    .weight_dst_y_vec(mesh_weight_dst_y_vec),
    .act_tx_valid(mesh_act_valid),
    .act_tx_ready(mesh_act_ready),
    .act_tx_payload(mesh_act_payload),
    .act_tx_last(mesh_act_last),
    .weight_tx_valid(mesh_weight_valid),
    .weight_tx_ready(mesh_weight_ready),
    .weight_tx_payload(mesh_weight_payload),
    .weight_tx_last(mesh_weight_last),
    .psum_tx_valid(mesh_psum_valid),
    .psum_tx_ready(mesh_psum_ready),
    .psum_tx_payload(mesh_psum_payload),
    .psum_tx_last(mesh_psum_last),
    .ctrl_tx_valid(mesh_ctrl_valid),
    .ctrl_tx_ready(mesh_ctrl_ready),
    .ctrl_tx_payload(mesh_ctrl_payload),
    .ctrl_tx_last(mesh_ctrl_last),
    .pe_valid(pe_valid),
    .pe_compute_fire(pe_compute_fire),
    .pe_acc_out(pe_acc_out)
);

endmodule
