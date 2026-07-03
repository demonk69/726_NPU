`timescale 1ns/1ps

module tb_router_node_lite;
    localparam XW        = 4;
    localparam YW        = 4;
    localparam LANES     = 8;
    localparam LANE_W    = 16;
    localparam PAYLOAD_W = LANES * LANE_W;
    localparam FLIT_W    = PAYLOAD_W + 1 + (2 * XW) + (2 * YW) + 2;
    localparam PORTS     = 5;

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

    reg [XW-1:0] cur_x;
    reg [YW-1:0] cur_y;
    reg route_mode_xy;
    reg [PORTS-1:0] act_route_cfg;
    reg [PORTS-1:0] weight_route_cfg;
    reg [PORTS-1:0] psum_route_cfg;
    reg [PORTS-1:0] ctrl_route_cfg;
    reg [PORTS-1:0] port_enable_mask;

    reg  [PORTS-1:0] in_valid;
    wire [PORTS-1:0] in_ready;
    reg  [PORTS*FLIT_W-1:0] in_data;

    wire [PORTS-1:0] out_valid;
    reg  [PORTS-1:0] out_ready;
    wire [PORTS*FLIT_W-1:0] out_data;

    integer errors;
    integer guard;
    integer stable_i;
    reg [PORTS-1:0] prev_out_valid;
    reg [PORTS-1:0] prev_out_ready;
    reg [PORTS*FLIT_W-1:0] prev_out_data;

    router_node_lite #(
        .XW(XW),
        .YW(YW),
        .LANES(LANES),
        .LANE_W(LANE_W),
        .PAYLOAD_W(PAYLOAD_W),
        .FLIT_W(FLIT_W),
        .PORTS(PORTS)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .cur_x(cur_x),
        .cur_y(cur_y),
        .route_mode_xy(route_mode_xy),
        .act_route_cfg(act_route_cfg),
        .weight_route_cfg(weight_route_cfg),
        .psum_route_cfg(psum_route_cfg),
        .ctrl_route_cfg(ctrl_route_cfg),
        .port_enable_mask(port_enable_mask),
        .in_valid(in_valid),
        .in_ready(in_ready),
        .in_data(in_data),
        .out_valid(out_valid),
        .out_ready(out_ready),
        .out_data(out_data)
    );

    initial clk = 1'b0;
    always #5 clk = ~clk;

    always @(posedge clk) begin
        if (!rst_n) begin
            prev_out_valid <= {PORTS{1'b0}};
            prev_out_ready <= {PORTS{1'b0}};
            prev_out_data  <= {(PORTS*FLIT_W){1'b0}};
        end else begin
            for (stable_i = 0; stable_i < PORTS; stable_i = stable_i + 1) begin
                if (prev_out_valid[stable_i] && !prev_out_ready[stable_i]) begin
                    if (!out_valid[stable_i] ||
                        out_data[stable_i*FLIT_W +: FLIT_W] !== prev_out_data[stable_i*FLIT_W +: FLIT_W]) begin
                        $display("[FAIL] STABLE_WHEN_STALLED port%0d", stable_i);
                        errors = errors + 1;
                    end
                end
            end
            prev_out_valid <= out_valid;
            prev_out_ready <= out_ready;
            prev_out_data  <= out_data;
        end
    end

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

    task tick;
        begin
            @(posedge clk);
            #1;
        end
    endtask

    task reset_dut;
        begin
            rst_n = 1'b0;
            cur_x = 4'd2;
            cur_y = 4'd3;
            route_mode_xy = 1'b0;
            act_route_cfg = 5'b10000;
            weight_route_cfg = 5'b10000;
            psum_route_cfg = 5'b10000;
            ctrl_route_cfg = 5'b10000;
            port_enable_mask = 5'b11111;
            in_valid = 5'b00000;
            in_data = {(PORTS*FLIT_W){1'b0}};
            out_ready = 5'b11111;
            repeat (4) tick();
            rst_n = 1'b1;
            tick();
        end
    endtask

    task send_one;
        input integer port;
        input [FLIT_W-1:0] flit;
        begin
            guard = 0;
            while (!in_ready[port] && guard < 20) begin
                guard = guard + 1;
                tick();
            end
            if (!in_ready[port]) begin
                $display("[FAIL] input port %0d not ready", port);
                errors = errors + 1;
            end
            in_data[port*FLIT_W +: FLIT_W] = flit;
            in_valid[port] = 1'b1;
            tick();
            in_valid[port] = 1'b0;
            in_data[port*FLIT_W +: FLIT_W] = {FLIT_W{1'b0}};
        end
    endtask

    task expect_one;
        input integer port;
        input [FLIT_W-1:0] expected;
        input [255:0] name;
        begin
            guard = 0;
            while (!out_valid[port] && guard < 20) begin
                guard = guard + 1;
                tick();
            end
            if (!out_valid[port]) begin
                $display("[FAIL] %0s port%0d no valid", name, port);
                errors = errors + 1;
            end else if (out_data[port*FLIT_W +: FLIT_W] !== expected) begin
                $display("[FAIL] %0s port%0d data mismatch", name, port);
                errors = errors + 1;
            end else begin
                $display("[PASS] %0s", name);
            end
            tick();
        end
    endtask

    task expect_mask;
        input [PORTS-1:0] mask;
        input [FLIT_W-1:0] expected;
        input [255:0] name;
        integer p;
        begin
            guard = 0;
            while (((out_valid & mask) != mask) && guard < 20) begin
                guard = guard + 1;
                tick();
            end
            if ((out_valid & mask) != mask) begin
                $display("[FAIL] %0s valid mask got=%b exp=%b", name, out_valid, mask);
                errors = errors + 1;
            end else begin
                for (p = 0; p < PORTS; p = p + 1) begin
                    if (mask[p] && out_data[p*FLIT_W +: FLIT_W] !== expected) begin
                        $display("[FAIL] %0s port%0d multicast data mismatch", name, p);
                        errors = errors + 1;
                    end
                end
                if (errors == 0)
                    $display("[PASS] %0s", name);
            end
            tick();
        end
    endtask

    task expect_no_outputs;
        input integer cycles;
        input [255:0] name;
        integer c;
        begin
            for (c = 0; c < cycles; c = c + 1)
                tick();
            if (out_valid != {PORTS{1'b0}}) begin
                $display("[FAIL] %0s unexpected out_valid=%b", name, out_valid);
                errors = errors + 1;
            end else begin
                $display("[PASS] %0s", name);
            end
        end
    endtask

    reg [FLIT_W-1:0] f0;
    reg [FLIT_W-1:0] f1;
    reg [FLIT_W-1:0] f2;

    initial begin
        errors = 0;

        // Dynamic XY: Local input routes to all four mesh directions and Local.
        reset_dut();
        route_mode_xy = 1'b1;
        f0 = make_flit(TYPE_CTRL, 4'd3, 4'd3, 4'd2, 4'd3, 1'b1,
                       128'h0000_0000_0000_0000_0000_0000_0000_00A1);
        send_one(DIR_L, f0);
        expect_one(DIR_E, f0, "XY_LOCAL_TO_EAST");
        f0 = make_flit(TYPE_CTRL, 4'd1, 4'd3, 4'd2, 4'd3, 1'b1,
                       128'h0000_0000_0000_0000_0000_0000_0000_00A2);
        send_one(DIR_L, f0);
        expect_one(DIR_W, f0, "XY_LOCAL_TO_WEST");
        f0 = make_flit(TYPE_CTRL, 4'd2, 4'd4, 4'd2, 4'd3, 1'b1,
                       128'h0000_0000_0000_0000_0000_0000_0000_00A3);
        send_one(DIR_L, f0);
        expect_one(DIR_S, f0, "XY_LOCAL_TO_SOUTH");
        f0 = make_flit(TYPE_CTRL, 4'd2, 4'd2, 4'd2, 4'd3, 1'b1,
                       128'h0000_0000_0000_0000_0000_0000_0000_00A4);
        send_one(DIR_L, f0);
        expect_one(DIR_N, f0, "XY_LOCAL_TO_NORTH");
        f0 = make_flit(TYPE_CTRL, 4'd2, 4'd3, 4'd2, 4'd3, 1'b1,
                       128'h0000_0000_0000_0000_0000_0000_0000_00A5);
        send_one(DIR_L, f0);
        expect_one(DIR_L, f0, "XY_LOCAL_TO_LOCAL");

        // Boundary mask: illegal east route is masked and dropped rather than
        // wedging the input forever.
        reset_dut();
        route_mode_xy = 1'b1;
        port_enable_mask = 5'b11111 & ~(5'b00001 << DIR_E);
        f0 = make_flit(TYPE_CTRL, 4'd3, 4'd3, 4'd2, 4'd3, 1'b1,
                       128'h0000_0000_0000_0000_0000_0000_0000_00D0);
        send_one(DIR_L, f0);
        expect_no_outputs(4, "PORT_MASK_DROPS_ILLEGAL_EAST");
        if (!in_ready[DIR_L]) begin
            $display("[FAIL] PORT_MASK_DROP did not release input");
            errors = errors + 1;
        end

        // Static multicast: activation from west is replicated to Local + East.
        reset_dut();
        route_mode_xy = 1'b0;
        act_route_cfg = (5'b00001 << DIR_L) | (5'b00001 << DIR_E);
        f0 = make_flit(TYPE_ACT, 4'd0, 4'd0, 4'd0, 4'd0, 1'b1,
                       128'h1111_2222_3333_4444_5555_6666_7777_8888);
        send_one(DIR_W, f0);
        expect_mask((5'b00001 << DIR_L) | (5'b00001 << DIR_E), f0,
                    "ACT_LOCAL_PLUS_EAST");

        // Multicast conflict: two inputs compete for the same Local+East
        // multicast outputs; RR issue sends one complete multicast at a time.
        reset_dut();
        act_route_cfg = (5'b00001 << DIR_L) | (5'b00001 << DIR_E);
        f0 = make_flit(TYPE_ACT, 4'd0, 4'd0, 4'd0, 4'd0, 1'b1,
                       128'h1111_0000_0000_0000_0000_0000_0000_0001);
        f1 = make_flit(TYPE_ACT, 4'd0, 4'd0, 4'd0, 4'd0, 1'b1,
                       128'h2222_0000_0000_0000_0000_0000_0000_0002);
        in_data[DIR_N*FLIT_W +: FLIT_W] = f0;
        in_data[DIR_W*FLIT_W +: FLIT_W] = f1;
        in_valid[DIR_N] = 1'b1;
        in_valid[DIR_W] = 1'b1;
        tick();
        in_valid[DIR_N] = 1'b0;
        in_valid[DIR_W] = 1'b0;
        expect_mask((5'b00001 << DIR_L) | (5'b00001 << DIR_E), f0,
                    "MC_CONFLICT_FIRST");
        expect_mask((5'b00001 << DIR_L) | (5'b00001 << DIR_E), f1,
                    "MC_CONFLICT_SECOND");

        // Backpressure: East output holds data stable while not ready; a second
        // flit waits in the input buffer and replaces it only after ready rises.
        reset_dut();
        act_route_cfg = (5'b00001 << DIR_E);
        out_ready[DIR_E] = 1'b0;
        f0 = make_flit(TYPE_ACT, 4'd0, 4'd0, 4'd0, 4'd0, 1'b1,
                       128'hAAAA_0000_0000_0000_0000_0000_0000_0001);
        f1 = make_flit(TYPE_ACT, 4'd0, 4'd0, 4'd0, 4'd0, 1'b1,
                       128'hBBBB_0000_0000_0000_0000_0000_0000_0002);
        send_one(DIR_W, f0);
        guard = 0;
        while (!out_valid[DIR_E] && guard < 20) begin
            guard = guard + 1;
            tick();
        end
        send_one(DIR_W, f1);
        repeat (3) tick();
        if (out_data[DIR_E*FLIT_W +: FLIT_W] !== f0) begin
            $display("[FAIL] BACKPRESSURE_HOLD data changed while ready=0");
            errors = errors + 1;
        end else begin
            $display("[PASS] BACKPRESSURE_HOLD");
        end
        out_ready[DIR_E] = 1'b1;
        tick();
        if (out_valid[DIR_E] && out_data[DIR_E*FLIT_W +: FLIT_W] === f1) begin
            $display("[PASS] BACKPRESSURE_RELEASE");
        end else begin
            $display("[FAIL] BACKPRESSURE_RELEASE did not forward second flit");
            errors = errors + 1;
        end
        tick();

        // Arbitration: two inputs target Local. Initial RR pointer grants North
        // first, then West once Local output accepts the first flit.
        reset_dut();
        ctrl_route_cfg = (5'b00001 << DIR_L);
        f0 = make_flit(TYPE_CTRL, 4'd2, 4'd3, 4'd0, 4'd0, 1'b1,
                       128'h0000_0000_0000_0000_0000_0000_0000_0100);
        f1 = make_flit(TYPE_CTRL, 4'd2, 4'd3, 4'd0, 4'd0, 1'b1,
                       128'h0000_0000_0000_0000_0000_0000_0000_0200);
        in_data[DIR_N*FLIT_W +: FLIT_W] = f0;
        in_data[DIR_W*FLIT_W +: FLIT_W] = f1;
        in_valid[DIR_N] = 1'b1;
        in_valid[DIR_W] = 1'b1;
        tick();
        in_valid[DIR_N] = 1'b0;
        in_valid[DIR_W] = 1'b0;
        expect_one(DIR_L, f0, "ARB_LOCAL_FIRST");
        expect_one(DIR_L, f1, "ARB_LOCAL_SECOND");

        if (errors == 0) begin
            $display("[PASS] tb_router_node_lite");
        end else begin
            $display("[FAIL] tb_router_node_lite errors=%0d", errors);
            $fatal;
        end
        $finish;
    end
endmodule
