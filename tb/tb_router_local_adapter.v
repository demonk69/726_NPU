`timescale 1ns/1ps

module tb_router_local_adapter;
    localparam XW        = 4;
    localparam YW        = 4;
    localparam LANES     = 8;
    localparam LANE_W    = 16;
    localparam PAYLOAD_W = LANES * LANE_W;
    localparam FLIT_W    = PAYLOAD_W + 1 + (2 * XW) + (2 * YW) + 2;

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

    reg [XW-1:0] local_x;
    reg [YW-1:0] local_y;
    reg [XW-1:0] act_dst_x, weight_dst_x, psum_dst_x, ctrl_dst_x;
    reg [YW-1:0] act_dst_y, weight_dst_y, psum_dst_y, ctrl_dst_y;

    reg                 rtr_rx_valid;
    wire                rtr_rx_ready;
    reg  [FLIT_W-1:0]   rtr_rx_data;

    wire                act_rx_valid;
    reg                 act_rx_ready;
    wire [PAYLOAD_W-1:0] act_rx_payload;
    wire                act_rx_last;
    wire                weight_rx_valid;
    reg                 weight_rx_ready;
    wire [PAYLOAD_W-1:0] weight_rx_payload;
    wire                weight_rx_last;
    wire                psum_rx_valid;
    reg                 psum_rx_ready;
    wire [PAYLOAD_W-1:0] psum_rx_payload;
    wire                psum_rx_last;
    wire                ctrl_rx_valid;
    reg                 ctrl_rx_ready;
    wire [PAYLOAD_W-1:0] ctrl_rx_payload;
    wire                ctrl_rx_last;

    reg                 act_tx_valid;
    wire                act_tx_ready;
    reg [PAYLOAD_W-1:0] act_tx_payload;
    reg                 act_tx_last;
    reg                 weight_tx_valid;
    wire                weight_tx_ready;
    reg [PAYLOAD_W-1:0] weight_tx_payload;
    reg                 weight_tx_last;
    reg                 psum_tx_valid;
    wire                psum_tx_ready;
    reg [PAYLOAD_W-1:0] psum_tx_payload;
    reg                 psum_tx_last;
    reg                 ctrl_tx_valid;
    wire                ctrl_tx_ready;
    reg [PAYLOAD_W-1:0] ctrl_tx_payload;
    reg                 ctrl_tx_last;

    wire                rtr_tx_valid;
    reg                 rtr_tx_ready;
    wire [FLIT_W-1:0]   rtr_tx_data;

    integer errors;
    integer guard;

    router_local_adapter #(
        .XW(XW),
        .YW(YW),
        .LANES(LANES),
        .LANE_W(LANE_W),
        .PAYLOAD_W(PAYLOAD_W),
        .FLIT_W(FLIT_W)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .local_x(local_x),
        .local_y(local_y),
        .act_dst_x(act_dst_x),
        .act_dst_y(act_dst_y),
        .weight_dst_x(weight_dst_x),
        .weight_dst_y(weight_dst_y),
        .psum_dst_x(psum_dst_x),
        .psum_dst_y(psum_dst_y),
        .ctrl_dst_x(ctrl_dst_x),
        .ctrl_dst_y(ctrl_dst_y),
        .rtr_rx_valid(rtr_rx_valid),
        .rtr_rx_ready(rtr_rx_ready),
        .rtr_rx_data(rtr_rx_data),
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
        .rtr_tx_valid(rtr_tx_valid),
        .rtr_tx_ready(rtr_tx_ready),
        .rtr_tx_data(rtr_tx_data)
    );

    initial clk = 1'b0;
    always #5 clk = ~clk;

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
            local_x = 4'd2;
            local_y = 4'd3;
            act_dst_x = 4'd4; act_dst_y = 4'd3;
            weight_dst_x = 4'd2; weight_dst_y = 4'd5;
            psum_dst_x = 4'd1; psum_dst_y = 4'd3;
            ctrl_dst_x = 4'd2; ctrl_dst_y = 4'd3;
            rtr_rx_valid = 1'b0;
            rtr_rx_data = {FLIT_W{1'b0}};
            act_rx_ready = 1'b1;
            weight_rx_ready = 1'b1;
            psum_rx_ready = 1'b1;
            ctrl_rx_ready = 1'b1;
            act_tx_valid = 1'b0;
            weight_tx_valid = 1'b0;
            psum_tx_valid = 1'b0;
            ctrl_tx_valid = 1'b0;
            act_tx_payload = {PAYLOAD_W{1'b0}};
            weight_tx_payload = {PAYLOAD_W{1'b0}};
            psum_tx_payload = {PAYLOAD_W{1'b0}};
            ctrl_tx_payload = {PAYLOAD_W{1'b0}};
            act_tx_last = 1'b1;
            weight_tx_last = 1'b1;
            psum_tx_last = 1'b1;
            ctrl_tx_last = 1'b1;
            rtr_tx_ready = 1'b1;
            repeat (4) tick();
            rst_n = 1'b1;
            tick();
        end
    endtask

    task drive_router_rx;
        input [FLIT_W-1:0] flit;
        begin
            guard = 0;
            while (!rtr_rx_ready && guard < 20) begin
                guard = guard + 1;
                tick();
            end
            if (!rtr_rx_ready) begin
                $display("[FAIL] router rx not ready");
                errors = errors + 1;
            end
            rtr_rx_data = flit;
            rtr_rx_valid = 1'b1;
            tick();
            rtr_rx_valid = 1'b0;
            rtr_rx_data = {FLIT_W{1'b0}};
        end
    endtask

    task check_rx;
        input [1:0] data_type;
        input [PAYLOAD_W-1:0] expected_payload;
        input expected_last;
        input [255:0] name;
        begin
            guard = 0;
            while (guard < 20 &&
                   !((data_type == TYPE_ACT && act_rx_valid) ||
                     (data_type == TYPE_WEIGHT && weight_rx_valid) ||
                     (data_type == TYPE_PSUM && psum_rx_valid) ||
                     (data_type == TYPE_CTRL && ctrl_rx_valid))) begin
                guard = guard + 1;
                tick();
            end
            case (data_type)
                TYPE_ACT: begin
                    if (!act_rx_valid || act_rx_payload !== expected_payload || act_rx_last !== expected_last) begin
                        $display("[FAIL] %0s act mismatch", name);
                        errors = errors + 1;
                    end else begin
                        $display("[PASS] %0s", name);
                    end
                end
                TYPE_WEIGHT: begin
                    if (!weight_rx_valid || weight_rx_payload !== expected_payload || weight_rx_last !== expected_last) begin
                        $display("[FAIL] %0s weight mismatch", name);
                        errors = errors + 1;
                    end else begin
                        $display("[PASS] %0s", name);
                    end
                end
                TYPE_PSUM: begin
                    if (!psum_rx_valid || psum_rx_payload !== expected_payload || psum_rx_last !== expected_last) begin
                        $display("[FAIL] %0s psum mismatch", name);
                        errors = errors + 1;
                    end else begin
                        $display("[PASS] %0s", name);
                    end
                end
                default: begin
                    if (!ctrl_rx_valid || ctrl_rx_payload !== expected_payload || ctrl_rx_last !== expected_last) begin
                        $display("[FAIL] %0s ctrl mismatch", name);
                        errors = errors + 1;
                    end else begin
                        $display("[PASS] %0s", name);
                    end
                end
            endcase
            tick();
        end
    endtask

    task check_tx_flit;
        input [FLIT_W-1:0] expected;
        input [255:0] name;
        begin
            guard = 0;
            while (!rtr_tx_valid && guard < 20) begin
                guard = guard + 1;
                tick();
            end
            if (!rtr_tx_valid) begin
                $display("[FAIL] %0s no tx valid", name);
                errors = errors + 1;
            end else if (rtr_tx_data !== expected) begin
                $display("[FAIL] %0s tx data mismatch", name);
                errors = errors + 1;
            end else begin
                $display("[PASS] %0s", name);
            end
        end
    endtask

    reg [FLIT_W-1:0] f_act;
    reg [FLIT_W-1:0] f_weight;
    reg [FLIT_W-1:0] f_psum;
    reg [FLIT_W-1:0] f_ctrl;
    reg [PAYLOAD_W-1:0] p_act;
    reg [PAYLOAD_W-1:0] p_weight;
    reg [PAYLOAD_W-1:0] p_psum;
    reg [PAYLOAD_W-1:0] p_ctrl;

    initial begin
        errors = 0;

        reset_dut();
        p_act    = 128'hAAAA_0000_0000_0000_0000_0000_0000_0001;
        p_weight = 128'hBBBB_0000_0000_0000_0000_0000_0000_0002;
        p_psum   = 128'hCCCC_0000_0000_0000_0000_0000_0000_0003;
        p_ctrl   = 128'hDDDD_0000_0000_0000_0000_0000_0000_0004;
        f_act    = make_flit(TYPE_ACT, 4'd2, 4'd3, 4'd0, 4'd0, 1'b1, p_act);
        f_weight = make_flit(TYPE_WEIGHT, 4'd2, 4'd3, 4'd0, 4'd0, 1'b0, p_weight);
        f_psum   = make_flit(TYPE_PSUM, 4'd2, 4'd3, 4'd0, 4'd0, 1'b1, p_psum);
        f_ctrl   = make_flit(TYPE_CTRL, 4'd2, 4'd3, 4'd0, 4'd0, 1'b1, p_ctrl);
        drive_router_rx(f_act);
        check_rx(TYPE_ACT, p_act, 1'b1, "RX_DEMUX_ACT");
        drive_router_rx(f_weight);
        check_rx(TYPE_WEIGHT, p_weight, 1'b0, "RX_DEMUX_WEIGHT");
        drive_router_rx(f_psum);
        check_rx(TYPE_PSUM, p_psum, 1'b1, "RX_DEMUX_PSUM");
        drive_router_rx(f_ctrl);
        check_rx(TYPE_CTRL, p_ctrl, 1'b1, "RX_DEMUX_CTRL");

        // RX backpressure: a full weight output deasserts router ready until the
        // local weight sink accepts the first flit.
        reset_dut();
        weight_rx_ready = 1'b0;
        drive_router_rx(f_weight);
        if (!weight_rx_valid || weight_rx_payload !== p_weight) begin
            $display("[FAIL] RX_BACKPRESSURE initial weight missing");
            errors = errors + 1;
        end
        rtr_rx_data = make_flit(TYPE_WEIGHT, 4'd2, 4'd3, 4'd0, 4'd0, 1'b1, p_ctrl);
        rtr_rx_valid = 1'b1;
        #1;
        if (rtr_rx_ready) begin
            $display("[FAIL] RX_BACKPRESSURE rtr_rx_ready should be low");
            errors = errors + 1;
        end else begin
            $display("[PASS] RX_BACKPRESSURE_READY_LOW");
        end
        weight_rx_ready = 1'b1;
        tick();
        rtr_rx_valid = 1'b0;
        check_rx(TYPE_WEIGHT, p_ctrl, 1'b1, "RX_BACKPRESSURE_RELEASE");

        // TX priority: control beats activation, then activation is accepted on
        // the following cycle once control is consumed.
        reset_dut();
        p_act  = 128'h0000_0000_0000_0000_0000_0000_0000_00A1;
        p_ctrl = 128'h0000_0000_0000_0000_0000_0000_0000_00C1;
        act_tx_payload = p_act;
        ctrl_tx_payload = p_ctrl;
        act_tx_last = 1'b1;
        ctrl_tx_last = 1'b1;
        act_tx_valid = 1'b1;
        ctrl_tx_valid = 1'b1;
        #1;
        if (!ctrl_tx_ready || act_tx_ready) begin
            $display("[FAIL] TX_PRIORITY ready signals ctrl=%0d act=%0d", ctrl_tx_ready, act_tx_ready);
            errors = errors + 1;
        end
        tick();
        ctrl_tx_valid = 1'b0;
        f_ctrl = make_flit(TYPE_CTRL, ctrl_dst_x, ctrl_dst_y, local_x, local_y, 1'b1, p_ctrl);
        check_tx_flit(f_ctrl, "TX_PRIORITY_CTRL_FIRST");
        tick();
        act_tx_valid = 1'b0;
        f_act = make_flit(TYPE_ACT, act_dst_x, act_dst_y, local_x, local_y, 1'b1, p_act);
        check_tx_flit(f_act, "TX_PRIORITY_ACT_SECOND");
        tick();

        // TX stall: output flit remains stable while router is not ready, and
        // new higher-priority traffic is not accepted until release.
        reset_dut();
        rtr_tx_ready = 1'b0;
        p_act  = 128'h0000_0000_0000_0000_0000_0000_0000_0A55;
        p_ctrl = 128'h0000_0000_0000_0000_0000_0000_0000_0C55;
        act_tx_payload = p_act;
        act_tx_valid = 1'b1;
        tick();
        act_tx_valid = 1'b0;
        f_act = make_flit(TYPE_ACT, act_dst_x, act_dst_y, local_x, local_y, 1'b1, p_act);
        check_tx_flit(f_act, "TX_STALL_LOADED_ACT");
        ctrl_tx_payload = p_ctrl;
        ctrl_tx_valid = 1'b1;
        repeat (3) tick();
        if (rtr_tx_data !== f_act || ctrl_tx_ready) begin
            $display("[FAIL] TX_STALL_HOLD data changed or ctrl ready asserted");
            errors = errors + 1;
        end else begin
            $display("[PASS] TX_STALL_HOLD");
        end
        rtr_tx_ready = 1'b1;
        tick();
        ctrl_tx_valid = 1'b0;
        f_ctrl = make_flit(TYPE_CTRL, ctrl_dst_x, ctrl_dst_y, local_x, local_y, 1'b1, p_ctrl);
        check_tx_flit(f_ctrl, "TX_STALL_RELEASE_CTRL");

        if (errors == 0) begin
            $display("[PASS] tb_router_local_adapter");
        end else begin
            $display("[FAIL] tb_router_local_adapter errors=%0d", errors);
            $fatal;
        end
        $finish;
    end
endmodule
