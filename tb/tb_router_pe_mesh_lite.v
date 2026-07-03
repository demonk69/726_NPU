`timescale 1ns/1ps

module tb_router_pe_mesh_lite;
    localparam ROWS      = 4;
    localparam COLS      = 4;
    localparam XW        = 4;
    localparam YW        = 4;
    localparam LANES     = 8;
    localparam LANE_W    = 16;
    localparam PE_DATA_W = 64;
    localparam ACC_W     = 32;
    localparam PAYLOAD_W = LANES * LANE_W;
    localparam FLIT_W    = PAYLOAD_W + 1 + (2 * XW) + (2 * YW) + 2;
    localparam PORTS     = 5;
    localparam NODES     = ROWS * COLS;

    localparam DIR_N = 0;
    localparam DIR_S = 1;
    localparam DIR_W = 2;
    localparam DIR_E = 3;
    localparam DIR_L = 4;

    reg clk;
    reg rst_n;

    reg route_mode_xy;
    reg [PORTS-1:0] act_route_cfg;
    reg [PORTS-1:0] weight_route_cfg;
    reg [PORTS-1:0] psum_route_cfg;
    reg [PORTS-1:0] ctrl_route_cfg;

    reg [XW-1:0] act_dst_x, weight_dst_x, psum_dst_x, ctrl_dst_x;
    reg [YW-1:0] act_dst_y, weight_dst_y, psum_dst_y, ctrl_dst_y;

    reg  [NODES-1:0] act_tx_valid;
    wire [NODES-1:0] act_tx_ready;
    reg  [NODES*PAYLOAD_W-1:0] act_tx_payload;
    reg  [NODES-1:0] act_tx_last;
    reg  [NODES-1:0] weight_tx_valid;
    wire [NODES-1:0] weight_tx_ready;
    reg  [NODES*PAYLOAD_W-1:0] weight_tx_payload;
    reg  [NODES-1:0] weight_tx_last;
    reg  [NODES-1:0] psum_tx_valid;
    wire [NODES-1:0] psum_tx_ready;
    reg  [NODES*PAYLOAD_W-1:0] psum_tx_payload;
    reg  [NODES-1:0] psum_tx_last;
    reg  [NODES-1:0] ctrl_tx_valid;
    wire [NODES-1:0] ctrl_tx_ready;
    reg  [NODES*PAYLOAD_W-1:0] ctrl_tx_payload;
    reg  [NODES-1:0] ctrl_tx_last;

    wire [NODES-1:0] pe_valid;
    wire [NODES-1:0] pe_compute_fire;
    wire [NODES*ACC_W-1:0] pe_acc_out;

    integer errors;
    integer guard;

    router_pe_mesh_lite #(
        .ROWS(ROWS),
        .COLS(COLS),
        .XW(XW),
        .YW(YW),
        .LANES(LANES),
        .LANE_W(LANE_W),
        .PE_DATA_W(PE_DATA_W),
        .ACC_W(ACC_W),
        .PAYLOAD_W(PAYLOAD_W),
        .FLIT_W(FLIT_W),
        .PORTS(PORTS),
        .NODES(NODES)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .mode(1'b0),
        .stat_mode(1'b1),
        .swap_w(1'b0),
        .ws_direct(1'b0),
        .route_mode_xy(route_mode_xy),
        .per_node_dst(1'b0),
        .act_route_cfg(act_route_cfg),
        .weight_route_cfg(weight_route_cfg),
        .psum_route_cfg(psum_route_cfg),
        .ctrl_route_cfg(ctrl_route_cfg),
        .act_dst_x(act_dst_x),
        .act_dst_y(act_dst_y),
        .weight_dst_x(weight_dst_x),
        .weight_dst_y(weight_dst_y),
        .psum_dst_x(psum_dst_x),
        .psum_dst_y(psum_dst_y),
        .ctrl_dst_x(ctrl_dst_x),
        .ctrl_dst_y(ctrl_dst_y),
        .act_dst_x_vec({NODES*XW{1'b0}}),
        .act_dst_y_vec({NODES*YW{1'b0}}),
        .weight_dst_x_vec({NODES*XW{1'b0}}),
        .weight_dst_y_vec({NODES*YW{1'b0}}),
        .act_tx_valid(act_tx_valid),
        .act_tx_ready(act_tx_ready),
        .act_tx_payload(act_tx_payload),
        .act_tx_last(act_tx_last),
        .weight_tx_valid(weight_tx_valid),
        .weight_tx_ready(weight_tx_ready),
        .weight_tx_payload(weight_tx_payload),
        .weight_tx_last(weight_tx_last),
        .psum_tx_valid(psum_tx_valid),
        .psum_tx_ready(psum_tx_ready),
        .psum_tx_payload(psum_tx_payload),
        .psum_tx_last(psum_tx_last),
        .ctrl_tx_valid(ctrl_tx_valid),
        .ctrl_tx_ready(ctrl_tx_ready),
        .ctrl_tx_payload(ctrl_tx_payload),
        .ctrl_tx_last(ctrl_tx_last),
        .pe_valid(pe_valid),
        .pe_compute_fire(pe_compute_fire),
        .pe_acc_out(pe_acc_out)
    );

    initial clk = 1'b0;
    always #5 clk = ~clk;

    function integer node_idx;
        input integer row;
        input integer col;
        begin
            node_idx = row * COLS + col;
        end
    endfunction

    function [63:0] pack_s8x8;
        input integer l7;
        input integer l6;
        input integer l5;
        input integer l4;
        input integer l3;
        input integer l2;
        input integer l1;
        input integer l0;
        begin
            pack_s8x8 = {l7[7:0], l6[7:0], l5[7:0], l4[7:0],
                         l3[7:0], l2[7:0], l1[7:0], l0[7:0]};
        end
    endfunction

    function [PAYLOAD_W-1:0] payload_from_s8x8;
        input [63:0] packed_val;
        begin
            payload_from_s8x8 = {PAYLOAD_W{1'b0}};
            payload_from_s8x8[63:0] = packed_val;
        end
    endfunction

    task tick;
        begin
            @(posedge clk);
            #1;
        end
    endtask

    task reset_dut;
        begin
            rst_n = 1'b0;
            route_mode_xy = 1'b0;
            act_route_cfg = (5'b00001 << DIR_L) | (5'b00001 << DIR_E);
            weight_route_cfg = (5'b00001 << DIR_L) | (5'b00001 << DIR_S);
            psum_route_cfg = (5'b00001 << DIR_L);
            ctrl_route_cfg = (5'b00001 << DIR_L);
            act_dst_x = 4'd0; act_dst_y = 4'd0;
            weight_dst_x = 4'd0; weight_dst_y = 4'd0;
            psum_dst_x = 4'd0; psum_dst_y = 4'd0;
            ctrl_dst_x = 4'd0; ctrl_dst_y = 4'd0;
            act_tx_valid = {NODES{1'b0}};
            weight_tx_valid = {NODES{1'b0}};
            psum_tx_valid = {NODES{1'b0}};
            ctrl_tx_valid = {NODES{1'b0}};
            act_tx_payload = {(NODES*PAYLOAD_W){1'b0}};
            weight_tx_payload = {(NODES*PAYLOAD_W){1'b0}};
            psum_tx_payload = {(NODES*PAYLOAD_W){1'b0}};
            ctrl_tx_payload = {(NODES*PAYLOAD_W){1'b0}};
            act_tx_last = {NODES{1'b1}};
            weight_tx_last = {NODES{1'b1}};
            psum_tx_last = {NODES{1'b1}};
            ctrl_tx_last = {NODES{1'b1}};
            repeat (4) tick();
            rst_n = 1'b1;
            tick();
        end
    endtask

    task send_act;
        input integer node;
        input [PAYLOAD_W-1:0] payload;
        begin
            act_tx_payload[node*PAYLOAD_W +: PAYLOAD_W] = payload;
            act_tx_valid[node] = 1'b1;
            guard = 0;
            while (!act_tx_ready[node] && guard < 50) begin
                guard = guard + 1;
                tick();
            end
            if (!act_tx_ready[node]) begin
                $display("[FAIL] act tx node%0d not ready", node);
                errors = errors + 1;
            end
            tick();
            act_tx_valid[node] = 1'b0;
            act_tx_payload[node*PAYLOAD_W +: PAYLOAD_W] = {PAYLOAD_W{1'b0}};
        end
    endtask

    task send_weight;
        input integer node;
        input [PAYLOAD_W-1:0] payload;
        begin
            weight_tx_payload[node*PAYLOAD_W +: PAYLOAD_W] = payload;
            weight_tx_valid[node] = 1'b1;
            guard = 0;
            while (!weight_tx_ready[node] && guard < 50) begin
                guard = guard + 1;
                tick();
            end
            if (!weight_tx_ready[node]) begin
                $display("[FAIL] weight tx node%0d not ready", node);
                errors = errors + 1;
            end
            tick();
            weight_tx_valid[node] = 1'b0;
            weight_tx_payload[node*PAYLOAD_W +: PAYLOAD_W] = {PAYLOAD_W{1'b0}};
        end
    endtask

    task expect_pe_result;
        input integer node;
        input [ACC_W-1:0] expected;
        input [255:0] name;
        begin
            guard = 0;
            while (!pe_valid[node] && guard < 200) begin
                guard = guard + 1;
                tick();
            end
            if (!pe_valid[node]) begin
                $display("[FAIL] %0s node%0d no pe_valid", name, node);
                errors = errors + 1;
            end else if (pe_acc_out[node*ACC_W +: ACC_W] !== expected) begin
                $display("[FAIL] %0s node%0d got=0x%08h exp=0x%08h",
                         name, node, pe_acc_out[node*ACC_W +: ACC_W], expected);
                errors = errors + 1;
            end else begin
                $display("[PASS] %0s", name);
            end
            tick();
        end
    endtask

    initial begin
        errors = 0;

        reset_dut();
        // Activation row broadcast from (0,1) intersects weight column broadcast
        // from (2,0) at PE(2,1). Low 64 payload bits carry 8 INT8 lanes.
        send_act(node_idx(1, 0), payload_from_s8x8(pack_s8x8(1, 1, 1, 1, 1, 1, 1, 1)));
        send_weight(node_idx(0, 2), payload_from_s8x8(pack_s8x8(2, 2, 2, 2, 2, 2, 2, 2)));
        expect_pe_result(node_idx(1, 2), 32'd16, "ROUTER_PE_INTERSECTION_1x2");

        reset_dut();
        // Signed 8-lane dot product at PE(3,2):
        // act={1,2,3,4,5,6,7,8}, weight={-1,1,-1,1,-1,1,-1,1} => 4.
        send_act(node_idx(3, 0), payload_from_s8x8(pack_s8x8(8, 7, 6, 5, 4, 3, 2, 1)));
        send_weight(node_idx(0, 2), payload_from_s8x8(pack_s8x8(1, -1, 1, -1, 1, -1, 1, -1)));
        expect_pe_result(node_idx(3, 2), 32'd4, "ROUTER_PE_SIGNED_DOT_3x2");

        if (errors == 0) begin
            $display("[PASS] tb_router_pe_mesh_lite");
        end else begin
            $display("[FAIL] tb_router_pe_mesh_lite errors=%0d", errors);
            $fatal;
        end
        $finish;
    end
endmodule
