`timescale 1ns/1ps

module tb_router_mesh_lite;
    localparam ROWS      = 4;
    localparam COLS      = 4;
    localparam XW        = 4;
    localparam YW        = 4;
    localparam LANES     = 8;
    localparam LANE_W    = 16;
    localparam PAYLOAD_W = LANES * LANE_W;
    localparam FLIT_W    = PAYLOAD_W + 1 + (2 * XW) + (2 * YW) + 2;
    localparam PORTS     = 5;
    localparam NODES     = ROWS * COLS;

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

    reg clk;
    reg rst_n;

    reg route_mode_xy;
    reg [PORTS-1:0] act_route_cfg;
    reg [PORTS-1:0] weight_route_cfg;
    reg [PORTS-1:0] psum_route_cfg;
    reg [PORTS-1:0] ctrl_route_cfg;

    reg  [NODES-1:0]        local_in_valid;
    wire [NODES-1:0]        local_in_ready;
    reg  [NODES*FLIT_W-1:0] local_in_data;
    wire [NODES-1:0]        local_out_valid;
    reg  [NODES-1:0]        local_out_ready;
    wire [NODES*FLIT_W-1:0] local_out_data;

    integer errors;
    integer guard;
    integer i;

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
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .route_mode_xy(route_mode_xy),
        .act_route_cfg(act_route_cfg),
        .weight_route_cfg(weight_route_cfg),
        .psum_route_cfg(psum_route_cfg),
        .ctrl_route_cfg(ctrl_route_cfg),
        .local_in_valid(local_in_valid),
        .local_in_ready(local_in_ready),
        .local_in_data(local_in_data),
        .local_out_valid(local_out_valid),
        .local_out_ready(local_out_ready),
        .local_out_data(local_out_data)
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

    function [FLIT_W-1:0] make_flit;
        input [1:0] data_type;
        input [XW-1:0] dst_x;
        input [YW-1:0] dst_y;
        input [XW-1:0] src_x;
        input [YW-1:0] src_y;
        input last;
        input [PAYLOAD_W-1:0] payload;
        begin
            make_flit = {FLIT_W{1'b0}};
            make_flit[PAYLOAD_W-1:0] = payload;
            make_flit[LAST_BIT] = last;
            make_flit[SRC_Y_LSB +: YW] = src_y;
            make_flit[SRC_X_LSB +: XW] = src_x;
            make_flit[DST_Y_LSB +: YW] = dst_y;
            make_flit[DST_X_LSB +: XW] = dst_x;
            make_flit[TYPE_LSB +: 2] = data_type;
        end
    endfunction

    function [NODES-1:0] row_mask;
        input integer row;
        integer c;
        begin
            row_mask = {NODES{1'b0}};
            for (c = 0; c < COLS; c = c + 1)
                row_mask[node_idx(row, c)] = 1'b1;
        end
    endfunction

    function [NODES-1:0] col_mask;
        input integer col;
        integer r;
        begin
            col_mask = {NODES{1'b0}};
            for (r = 0; r < ROWS; r = r + 1)
                col_mask[node_idx(r, col)] = 1'b1;
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
            act_route_cfg = (5'b00001 << DIR_L);
            weight_route_cfg = (5'b00001 << DIR_L);
            psum_route_cfg = (5'b00001 << DIR_L);
            ctrl_route_cfg = (5'b00001 << DIR_L);
            local_in_valid = {NODES{1'b0}};
            local_in_data = {(NODES*FLIT_W){1'b0}};
            local_out_ready = {NODES{1'b1}};
            repeat (4) tick();
            rst_n = 1'b1;
            tick();
        end
    endtask

    task inject_local;
        input integer node;
        input [FLIT_W-1:0] flit;
        begin
            guard = 0;
            while (!local_in_ready[node] && guard < 50) begin
                guard = guard + 1;
                tick();
            end
            if (!local_in_ready[node]) begin
                $display("[FAIL] local input node%0d not ready", node);
                errors = errors + 1;
            end
            local_in_data[node*FLIT_W +: FLIT_W] = flit;
            local_in_valid[node] = 1'b1;
            tick();
            local_in_valid[node] = 1'b0;
            local_in_data[node*FLIT_W +: FLIT_W] = {FLIT_W{1'b0}};
        end
    endtask

    task expect_local_mask;
        input [NODES-1:0] mask;
        input [FLIT_W-1:0] expected;
        input [255:0] name;
        reg [NODES-1:0] seen;
        integer cyc;
        begin
            seen = {NODES{1'b0}};
            for (cyc = 0; cyc < 120; cyc = cyc + 1) begin
                for (i = 0; i < NODES; i = i + 1) begin
                    if (local_out_valid[i]) begin
                        if (!mask[i]) begin
                            $display("[FAIL] %0s unexpected node%0d", name, i);
                            errors = errors + 1;
                        end else if (local_out_data[i*FLIT_W +: FLIT_W] !== expected) begin
                            $display("[FAIL] %0s node%0d data mismatch", name, i);
                            errors = errors + 1;
                        end else begin
                            seen[i] = 1'b1;
                        end
                    end
                end
                if ((seen & mask) == mask)
                    cyc = 120;
                else
                    tick();
            end
            if ((seen & mask) != mask) begin
                $display("[FAIL] %0s seen=%h expected=%h", name, seen, mask);
                errors = errors + 1;
            end else begin
                $display("[PASS] %0s", name);
            end
            tick();
        end
    endtask

    reg [FLIT_W-1:0] f_act;
    reg [FLIT_W-1:0] f_weight;
    reg [FLIT_W-1:0] f_ctrl;
    reg [NODES-1:0] expected_mask;

    initial begin
        errors = 0;

        // Static activation row broadcast: Local + East from row 1, col 0.
        reset_dut();
        route_mode_xy = 1'b0;
        act_route_cfg = (5'b00001 << DIR_L) | (5'b00001 << DIR_E);
        f_act = make_flit(TYPE_ACT, 4'd0, 4'd0, 4'd0, 4'd1, 1'b1,
                          128'hA100_0000_0000_0000_0000_0000_0000_0001);
        inject_local(node_idx(1, 0), f_act);
        expect_local_mask(row_mask(1), f_act, "ACT_ROW_BROADCAST_R1");

        // Static weight column broadcast: Local + South from row 0, col 2.
        reset_dut();
        route_mode_xy = 1'b0;
        weight_route_cfg = (5'b00001 << DIR_L) | (5'b00001 << DIR_S);
        f_weight = make_flit(TYPE_WEIGHT, 4'd0, 4'd0, 4'd2, 4'd0, 1'b1,
                             128'hB200_0000_0000_0000_0000_0000_0000_0002);
        inject_local(node_idx(0, 2), f_weight);
        expect_local_mask(col_mask(2), f_weight, "WEIGHT_COL_BROADCAST_C2");

        // Boundary mask trims East at the last column, so only the source local
        // node should observe the activation flit.
        reset_dut();
        route_mode_xy = 1'b0;
        act_route_cfg = (5'b00001 << DIR_L) | (5'b00001 << DIR_E);
        f_act = make_flit(TYPE_ACT, 4'd0, 4'd0, 4'd3, 4'd2, 1'b1,
                          128'hA300_0000_0000_0000_0000_0000_0000_0003);
        expected_mask = {NODES{1'b0}};
        expected_mask[node_idx(2, 3)] = 1'b1;
        inject_local(node_idx(2, 3), f_act);
        expect_local_mask(expected_mask, f_act, "BOUNDARY_LOCAL_ONLY_LAST_COL");

        // Dynamic XY: from (0,0) to (3,2). Only the destination Local output
        // should see the flit because intermediate nodes route East/South only.
        reset_dut();
        route_mode_xy = 1'b1;
        f_ctrl = make_flit(TYPE_CTRL, 4'd3, 4'd2, 4'd0, 4'd0, 1'b1,
                           128'hC400_0000_0000_0000_0000_0000_0000_0004);
        expected_mask = {NODES{1'b0}};
        expected_mask[node_idx(2, 3)] = 1'b1;
        inject_local(node_idx(0, 0), f_ctrl);
        expect_local_mask(expected_mask, f_ctrl, "XY_MESH_00_TO_32");

        if (errors == 0) begin
            $display("[PASS] tb_router_mesh_lite");
        end else begin
            $display("[FAIL] tb_router_mesh_lite errors=%0d", errors);
            $fatal;
        end
        $finish;
    end
endmodule
