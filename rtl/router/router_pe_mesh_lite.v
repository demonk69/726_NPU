// =============================================================================
// Module  : router_pe_mesh_lite
// Project : NPU_prj
// Desc    : Prototype integration of router_mesh_lite + router_local_adapter
//           + pe_top. This module is intentionally standalone and does not
//           replace the existing NPU datapath.
//
//           It proves the hardware path:
//             local payload -> flit -> router mesh -> local adapter -> PE inputs
// =============================================================================

`timescale 1ns/1ps

module router_pe_mesh_lite #(
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

    input  wire                      mode,
    input  wire                      stat_mode,
    input  wire                      swap_w,
    input  wire                      ws_direct,

    input  wire                      route_mode_xy,
    input  wire                      per_node_dst,
    input  wire [PORTS-1:0]          act_route_cfg,
    input  wire [PORTS-1:0]          weight_route_cfg,
    input  wire [PORTS-1:0]          psum_route_cfg,
    input  wire [PORTS-1:0]          ctrl_route_cfg,

    input  wire [XW-1:0]             act_dst_x,
    input  wire [YW-1:0]             act_dst_y,
    input  wire [XW-1:0]             weight_dst_x,
    input  wire [YW-1:0]             weight_dst_y,
    input  wire [XW-1:0]             psum_dst_x,
    input  wire [YW-1:0]             psum_dst_y,
    input  wire [XW-1:0]             ctrl_dst_x,
    input  wire [YW-1:0]             ctrl_dst_y,
    input  wire [NODES*XW-1:0]       act_dst_x_vec,
    input  wire [NODES*YW-1:0]       act_dst_y_vec,
    input  wire [NODES*XW-1:0]       weight_dst_x_vec,
    input  wire [NODES*YW-1:0]       weight_dst_y_vec,

    input  wire [NODES-1:0]          act_tx_valid,
    output wire [NODES-1:0]          act_tx_ready,
    input  wire [NODES*PAYLOAD_W-1:0] act_tx_payload,
    input  wire [NODES-1:0]          act_tx_last,

    input  wire [NODES-1:0]          weight_tx_valid,
    output wire [NODES-1:0]          weight_tx_ready,
    input  wire [NODES*PAYLOAD_W-1:0] weight_tx_payload,
    input  wire [NODES-1:0]          weight_tx_last,

    input  wire [NODES-1:0]          psum_tx_valid,
    output wire [NODES-1:0]          psum_tx_ready,
    input  wire [NODES*PAYLOAD_W-1:0] psum_tx_payload,
    input  wire [NODES-1:0]          psum_tx_last,

    input  wire [NODES-1:0]          ctrl_tx_valid,
    output wire [NODES-1:0]          ctrl_tx_ready,
    input  wire [NODES*PAYLOAD_W-1:0] ctrl_tx_payload,
    input  wire [NODES-1:0]          ctrl_tx_last,

    output wire [NODES-1:0]          pe_valid,
    output wire [NODES-1:0]          pe_compute_fire,
    output wire [NODES*ACC_W-1:0]    pe_acc_out
);

wire [NODES-1:0]          mesh_local_in_valid;
wire [NODES-1:0]          mesh_local_in_ready;
wire [NODES*FLIT_W-1:0]   mesh_local_in_data;
wire [NODES-1:0]          mesh_local_out_valid;
wire [NODES-1:0]          mesh_local_out_ready;
wire [NODES*FLIT_W-1:0]   mesh_local_out_data;
wire [ACC_W-1:0]          ws_acc_v [0:ROWS][0:COLS-1];
wire                      ws_valid_v [0:ROWS][0:COLS-1];
genvar gy, gx, gb;

localparam integer WS_SWAP_DELAY = 8;
reg [WS_SWAP_DELAY:0] ws_swap_pipe;
reg                   ws_compute_armed;
reg                   swap_w_d;
wire                  true_ws_mode = !stat_mode && !ws_direct;
wire                  swap_w_rise = swap_w && !swap_w_d;
wire                  ws_swap_pulse = true_ws_mode && ws_swap_pipe[WS_SWAP_DELAY];

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        ws_swap_pipe <= {(WS_SWAP_DELAY+1){1'b0}};
        ws_compute_armed <= 1'b0;
        swap_w_d <= 1'b0;
    end else if (!true_ws_mode) begin
        ws_swap_pipe <= {(WS_SWAP_DELAY+1){1'b0}};
        ws_compute_armed <= 1'b0;
        swap_w_d <= swap_w;
    end else begin
        swap_w_d <= swap_w;
        ws_swap_pipe <= {ws_swap_pipe[WS_SWAP_DELAY-1:0], swap_w_rise};
        if (swap_w_rise)
            ws_compute_armed <= 1'b0;
        else if (ws_swap_pipe[WS_SWAP_DELAY])
            ws_compute_armed <= 1'b1;
    end
end

generate
    for (gb = 0; gb < COLS; gb = gb + 1) begin : gen_ws_top_boundary
        assign ws_acc_v[0][gb] = {ACC_W{1'b0}};
        assign ws_valid_v[0][gb] = 1'b1;
    end
endgenerate

router_mesh_lite #(
    .ROWS(ROWS),
    .COLS(COLS),
    .XW(XW),
    .YW(YW),
    .LANES(LANES),
    .LANE_W(LANE_W),
    .PAYLOAD_W(PAYLOAD_W),
    .FLIT_W(FLIT_W),
    .PORTS(PORTS),
    .NODES(NODES)
) u_mesh (
    .clk(clk),
    .rst_n(rst_n),
    .route_mode_xy(route_mode_xy),
    .act_route_cfg(act_route_cfg),
    .weight_route_cfg(weight_route_cfg),
    .psum_route_cfg(psum_route_cfg),
    .ctrl_route_cfg(ctrl_route_cfg),
    .local_in_valid(mesh_local_in_valid),
    .local_in_ready(mesh_local_in_ready),
    .local_in_data(mesh_local_in_data),
    .local_out_valid(mesh_local_out_valid),
    .local_out_ready(mesh_local_out_ready),
    .local_out_data(mesh_local_out_data)
);

generate
    for (gy = 0; gy < ROWS; gy = gy + 1) begin : gen_row
        for (gx = 0; gx < COLS; gx = gx + 1) begin : gen_col
            localparam integer NODE = gy * COLS + gx;
            localparam [XW-1:0] LOCAL_X = gx;
            localparam [YW-1:0] LOCAL_Y = gy;

            wire                act_rx_valid;
            wire                act_rx_ready;
            wire [PAYLOAD_W-1:0] act_rx_payload;
            wire                act_rx_last;
            wire                weight_rx_valid;
            wire                weight_rx_ready;
            wire [PAYLOAD_W-1:0] weight_rx_payload;
            wire                weight_rx_last;
            wire                psum_rx_valid;
            wire                psum_rx_ready;
            wire [PAYLOAD_W-1:0] psum_rx_payload;
            wire                psum_rx_last;
            wire                ctrl_rx_valid;
            wire                ctrl_rx_ready;
            wire [PAYLOAD_W-1:0] ctrl_rx_payload;
            wire                ctrl_rx_last;

            localparam integer STREAM_DEPTH = (PE_STREAM_BUF_DEPTH < 1) ? 1 : PE_STREAM_BUF_DEPTH;
            localparam integer STREAM_PTR_W = (STREAM_DEPTH <= 1) ? 1 : $clog2(STREAM_DEPTH);
            localparam integer STREAM_CNT_W = (STREAM_DEPTH <= 1) ? 1 : $clog2(STREAM_DEPTH + 1);
            localparam [STREAM_PTR_W-1:0] STREAM_LAST_PTR = STREAM_DEPTH - 1;
            localparam [STREAM_CNT_W-1:0] STREAM_DEPTH_COUNT = STREAM_DEPTH;

            reg [PE_DATA_W-1:0] act_fifo_data [0:STREAM_DEPTH-1];
            reg                 act_fifo_last [0:STREAM_DEPTH-1];
            reg [STREAM_PTR_W-1:0] act_fifo_rd_ptr;
            reg [STREAM_PTR_W-1:0] act_fifo_wr_ptr;
            reg [STREAM_CNT_W-1:0] act_fifo_count;
            reg [PE_DATA_W-1:0] weight_fifo_data [0:STREAM_DEPTH-1];
            reg                 weight_fifo_last [0:STREAM_DEPTH-1];
            reg [STREAM_PTR_W-1:0] weight_fifo_rd_ptr;
            reg [STREAM_PTR_W-1:0] weight_fifo_wr_ptr;
            reg [STREAM_CNT_W-1:0] weight_fifo_count;
            wire                compute_fire;
            wire                compute_flush;
            wire                os_like_mode;
            wire                os_compute_fire;
            wire                os_compute_flush;
            wire                ws_compute_fire;
            wire                ws_weight_load_fire;
            wire                act_fifo_pop;
            wire                weight_fifo_pop;
            wire                act_fifo_empty;
            wire                act_fifo_full;
            wire                weight_fifo_empty;
            wire                weight_fifo_full;
            wire                act_fifo_push;
            wire                weight_fifo_push;
            wire [PE_DATA_W-1:0] act_fifo_head_data;
            wire [PE_DATA_W-1:0] weight_fifo_head_data;
            wire                act_fifo_head_last;
            wire                weight_fifo_head_last;
            integer             fifo_i;

            router_local_adapter #(
                .XW(XW),
                .YW(YW),
                .LANES(LANES),
                .LANE_W(LANE_W),
                .PAYLOAD_W(PAYLOAD_W),
                .FLIT_W(FLIT_W)
            ) u_adapter (
                .clk(clk),
                .rst_n(rst_n),
                .local_x(LOCAL_X),
                .local_y(LOCAL_Y),
                .act_dst_x(per_node_dst ? act_dst_x_vec[NODE*XW +: XW] : act_dst_x),
                .act_dst_y(per_node_dst ? act_dst_y_vec[NODE*YW +: YW] : act_dst_y),
                .weight_dst_x(per_node_dst ? weight_dst_x_vec[NODE*XW +: XW] : weight_dst_x),
                .weight_dst_y(per_node_dst ? weight_dst_y_vec[NODE*YW +: YW] : weight_dst_y),
                .psum_dst_x(psum_dst_x),
                .psum_dst_y(psum_dst_y),
                .ctrl_dst_x(ctrl_dst_x),
                .ctrl_dst_y(ctrl_dst_y),
                .rtr_rx_valid(mesh_local_out_valid[NODE]),
                .rtr_rx_ready(mesh_local_out_ready[NODE]),
                .rtr_rx_data(mesh_local_out_data[NODE*FLIT_W +: FLIT_W]),
                .act_rx_valid(act_rx_valid),
                .act_rx_ready(act_rx_ready),
                .act_rx_payload(act_rx_payload),
                .act_rx_last(act_rx_last),
                .weight_rx_valid(weight_rx_valid),
                .weight_rx_ready(weight_rx_ready),
                .weight_rx_payload(weight_rx_payload),
                .weight_rx_last(weight_rx_last),
                .psum_rx_valid(psum_rx_valid),
                .psum_rx_ready(psum_rx_ready),
                .psum_rx_payload(psum_rx_payload),
                .psum_rx_last(psum_rx_last),
                .ctrl_rx_valid(ctrl_rx_valid),
                .ctrl_rx_ready(ctrl_rx_ready),
                .ctrl_rx_payload(ctrl_rx_payload),
                .ctrl_rx_last(ctrl_rx_last),
                .act_tx_valid(act_tx_valid[NODE]),
                .act_tx_ready(act_tx_ready[NODE]),
                .act_tx_payload(act_tx_payload[NODE*PAYLOAD_W +: PAYLOAD_W]),
                .act_tx_last(act_tx_last[NODE]),
                .weight_tx_valid(weight_tx_valid[NODE]),
                .weight_tx_ready(weight_tx_ready[NODE]),
                .weight_tx_payload(weight_tx_payload[NODE*PAYLOAD_W +: PAYLOAD_W]),
                .weight_tx_last(weight_tx_last[NODE]),
                .psum_tx_valid(psum_tx_valid[NODE]),
                .psum_tx_ready(psum_tx_ready[NODE]),
                .psum_tx_payload(psum_tx_payload[NODE*PAYLOAD_W +: PAYLOAD_W]),
                .psum_tx_last(psum_tx_last[NODE]),
                .ctrl_tx_valid(ctrl_tx_valid[NODE]),
                .ctrl_tx_ready(ctrl_tx_ready[NODE]),
                .ctrl_tx_payload(ctrl_tx_payload[NODE*PAYLOAD_W +: PAYLOAD_W]),
                .ctrl_tx_last(ctrl_tx_last[NODE]),
                .rtr_tx_valid(mesh_local_in_valid[NODE]),
                .rtr_tx_ready(mesh_local_in_ready[NODE]),
                .rtr_tx_data(mesh_local_in_data[NODE*FLIT_W +: FLIT_W])
            );

            assign act_fifo_empty = (act_fifo_count == {STREAM_CNT_W{1'b0}});
            assign act_fifo_full = (act_fifo_count == STREAM_DEPTH_COUNT);
            assign weight_fifo_empty = (weight_fifo_count == {STREAM_CNT_W{1'b0}});
            assign weight_fifo_full = (weight_fifo_count == STREAM_DEPTH_COUNT);
            assign act_fifo_head_data = act_fifo_data[act_fifo_rd_ptr];
            assign act_fifo_head_last = act_fifo_last[act_fifo_rd_ptr];
            assign weight_fifo_head_data = weight_fifo_data[weight_fifo_rd_ptr];
            assign weight_fifo_head_last = weight_fifo_last[weight_fifo_rd_ptr];

            assign os_like_mode = stat_mode || ws_direct;
            assign os_compute_fire = !act_fifo_empty && !weight_fifo_empty;
            assign os_compute_flush = os_compute_fire &&
                                      ((AUTO_FLUSH_ON_COMPUTE != 0) ||
                                       (act_fifo_head_last && weight_fifo_head_last));
            assign ws_compute_fire = true_ws_mode &&
                                     ws_compute_armed &&
                                     !act_fifo_empty &&
                                     ws_valid_v[gy][gx];
            assign ws_weight_load_fire = true_ws_mode && !weight_fifo_empty;
            assign compute_fire = os_like_mode ? os_compute_fire : ws_compute_fire;
            assign compute_flush = os_like_mode ? os_compute_flush : 1'b0;
            assign pe_compute_fire[NODE] = compute_fire;

            assign act_fifo_pop = os_like_mode ? os_compute_fire : ws_compute_fire;
            assign weight_fifo_pop = os_like_mode ? os_compute_fire : ws_weight_load_fire;
            assign act_rx_ready = !act_fifo_full || act_fifo_pop;
            assign weight_rx_ready = !weight_fifo_full || weight_fifo_pop;
            assign act_fifo_push = act_rx_valid && act_rx_ready;
            assign weight_fifo_push = weight_rx_valid && weight_rx_ready;
            assign psum_rx_ready = 1'b1;
            assign ctrl_rx_ready = 1'b1;

            always @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    act_fifo_rd_ptr <= {STREAM_PTR_W{1'b0}};
                    act_fifo_wr_ptr <= {STREAM_PTR_W{1'b0}};
                    act_fifo_count <= {STREAM_CNT_W{1'b0}};
                    weight_fifo_rd_ptr <= {STREAM_PTR_W{1'b0}};
                    weight_fifo_wr_ptr <= {STREAM_PTR_W{1'b0}};
                    weight_fifo_count <= {STREAM_CNT_W{1'b0}};
                    for (fifo_i = 0; fifo_i < STREAM_DEPTH; fifo_i = fifo_i + 1) begin
                        act_fifo_data[fifo_i] <= {PE_DATA_W{1'b0}};
                        act_fifo_last[fifo_i] <= 1'b0;
                        weight_fifo_data[fifo_i] <= {PE_DATA_W{1'b0}};
                        weight_fifo_last[fifo_i] <= 1'b0;
                    end
                end else begin
                    if (act_fifo_push) begin
                        act_fifo_data[act_fifo_wr_ptr] <= act_rx_payload[PE_DATA_W-1:0];
                        act_fifo_last[act_fifo_wr_ptr] <= act_rx_last;
                        act_fifo_wr_ptr <= (act_fifo_wr_ptr == STREAM_LAST_PTR) ?
                                           {STREAM_PTR_W{1'b0}} :
                                           (act_fifo_wr_ptr + 1'b1);
                    end
                    if (act_fifo_pop) begin
                        act_fifo_rd_ptr <= (act_fifo_rd_ptr == STREAM_LAST_PTR) ?
                                           {STREAM_PTR_W{1'b0}} :
                                           (act_fifo_rd_ptr + 1'b1);
                    end

                    case ({act_fifo_push, act_fifo_pop})
                        2'b10: act_fifo_count <= act_fifo_count + 1'b1;
                        2'b01: act_fifo_count <= act_fifo_count - 1'b1;
                        default: act_fifo_count <= act_fifo_count;
                    endcase

                    if (weight_fifo_push) begin
                        weight_fifo_data[weight_fifo_wr_ptr] <= weight_rx_payload[PE_DATA_W-1:0];
                        weight_fifo_last[weight_fifo_wr_ptr] <= weight_rx_last;
                        weight_fifo_wr_ptr <= (weight_fifo_wr_ptr == STREAM_LAST_PTR) ?
                                              {STREAM_PTR_W{1'b0}} :
                                              (weight_fifo_wr_ptr + 1'b1);
                    end
                    if (weight_fifo_pop) begin
                        weight_fifo_rd_ptr <= (weight_fifo_rd_ptr == STREAM_LAST_PTR) ?
                                              {STREAM_PTR_W{1'b0}} :
                                              (weight_fifo_rd_ptr + 1'b1);
                    end

                    case ({weight_fifo_push, weight_fifo_pop})
                        2'b10: weight_fifo_count <= weight_fifo_count + 1'b1;
                        2'b01: weight_fifo_count <= weight_fifo_count - 1'b1;
                        default: weight_fifo_count <= weight_fifo_count;
                    endcase
                end
            end

            pe_top #(
                .DATA_W(PE_DATA_W),
                .ACC_W(ACC_W),
                .INT8_SIMD_LANES(INT8_SIMD_LANES),
                .FP16_ENABLE(FP16_ENABLE),
                .INT8_SCALAR_SIGNEXT_COMPAT(INT8_SCALAR_SIGNEXT_COMPAT)
            ) u_pe (
                .clk(clk),
                .rst_n(rst_n),
                .mode(mode),
                .stat_mode(os_like_mode),
                .en(compute_fire),
                .flush(compute_flush),
                .load_w(ws_weight_load_fire),
                .swap_w(ws_swap_pulse),
                .acc_init_en(1'b0),
                .w_in(weight_fifo_head_data),
                .a_in(act_fifo_head_data),
                .acc_in(true_ws_mode ? ws_acc_v[gy][gx] : {ACC_W{1'b0}}),
                .acc_init({ACC_W{1'b0}}),
                .acc_out(pe_acc_out[NODE*ACC_W +: ACC_W]),
                .valid_out(pe_valid[NODE])
            );

            assign ws_acc_v[gy+1][gx] = pe_acc_out[NODE*ACC_W +: ACC_W];
            assign ws_valid_v[gy+1][gx] = true_ws_mode ? pe_valid[NODE] : 1'b0;
        end
    end
endgenerate

endmodule
