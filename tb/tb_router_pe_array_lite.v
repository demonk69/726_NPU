`timescale 1ns/1ps

module tb_router_pe_array_lite;
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

    reg clk;
    reg rst_n;

    reg  [ROWS-1:0] act_valid;
    wire [ROWS-1:0] act_ready;
    reg  [ROWS*PE_DATA_W-1:0] act_data;
    reg  [COLS-1:0] weight_valid;
    wire [COLS-1:0] weight_ready;
    reg  [COLS*PE_DATA_W-1:0] weight_data;
    wire [NODES-1:0] pe_valid;
    wire [NODES-1:0] pe_compute_fire;
    wire [NODES*ACC_W-1:0] pe_acc_out;

    integer errors;
    integer guard;

    router_pe_array_lite #(
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
        .flush(1'b1),
        .mode(1'b0),
        .stat_mode(1'b1),
        .load_w(1'b0),
        .swap_w(1'b0),
        .ws_direct(1'b0),
        .ws_load_row(4'd0),
        .act_valid(act_valid),
        .act_ready(act_ready),
        .act_data(act_data),
        .weight_valid(weight_valid),
        .weight_ready(weight_ready),
        .weight_data(weight_data),
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

    task tick;
        begin
            @(posedge clk);
            #1;
        end
    endtask

    task reset_dut;
        begin
            rst_n = 1'b0;
            act_valid = {ROWS{1'b0}};
            act_data = {(ROWS*PE_DATA_W){1'b0}};
            weight_valid = {COLS{1'b0}};
            weight_data = {(COLS*PE_DATA_W){1'b0}};
            repeat (4) tick();
            rst_n = 1'b1;
            tick();
        end
    endtask

    task send_act_row;
        input integer row;
        input [PE_DATA_W-1:0] data;
        begin
            act_data[row*PE_DATA_W +: PE_DATA_W] = data;
            act_valid[row] = 1'b1;
            guard = 0;
            while (!act_ready[row] && guard < 80) begin
                guard = guard + 1;
                tick();
            end
            if (!act_ready[row]) begin
                $display("[FAIL] act row%0d not ready", row);
                errors = errors + 1;
            end
            tick();
            act_valid[row] = 1'b0;
            act_data[row*PE_DATA_W +: PE_DATA_W] = {PE_DATA_W{1'b0}};
        end
    endtask

    task send_weight_col;
        input integer col;
        input [PE_DATA_W-1:0] data;
        begin
            weight_data[col*PE_DATA_W +: PE_DATA_W] = data;
            weight_valid[col] = 1'b1;
            guard = 0;
            while (!weight_ready[col] && guard < 80) begin
                guard = guard + 1;
                tick();
            end
            if (!weight_ready[col]) begin
                $display("[FAIL] weight col%0d not ready", col);
                errors = errors + 1;
            end
            tick();
            weight_valid[col] = 1'b0;
            weight_data[col*PE_DATA_W +: PE_DATA_W] = {PE_DATA_W{1'b0}};
        end
    endtask

    task expect_pe_result;
        input integer row;
        input integer col;
        input [ACC_W-1:0] expected;
        input [255:0] name;
        integer node;
        begin
            node = node_idx(row, col);
            guard = 0;
            while (!pe_valid[node] && guard < 200) begin
                guard = guard + 1;
                tick();
            end
            if (!pe_valid[node]) begin
                $display("[FAIL] %0s PE(%0d,%0d) no valid", name, row, col);
                errors = errors + 1;
            end else if (pe_acc_out[node*ACC_W +: ACC_W] !== expected) begin
                $display("[FAIL] %0s PE(%0d,%0d) got=0x%08h exp=0x%08h",
                         name, row, col, pe_acc_out[node*ACC_W +: ACC_W], expected);
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
        send_act_row(1, pack_s8x8(1, 1, 1, 1, 1, 1, 1, 1));
        send_weight_col(2, pack_s8x8(2, 2, 2, 2, 2, 2, 2, 2));
        expect_pe_result(1, 2, 32'd16, "ARRAY_IF_ACT_ROW_WEIGHT_COL");

        reset_dut();
        send_act_row(3, pack_s8x8(8, 7, 6, 5, 4, 3, 2, 1));
        send_weight_col(2, pack_s8x8(1, -1, 1, -1, 1, -1, 1, -1));
        expect_pe_result(3, 2, 32'd4, "ARRAY_IF_SIGNED_DOT");

        reset_dut();
        send_weight_col(0, pack_s8x8(3, 3, 3, 3, 3, 3, 3, 3));
        send_act_row(0, pack_s8x8(1, 1, 1, 1, 1, 1, 1, 1));
        expect_pe_result(0, 0, 32'd24, "ARRAY_IF_CORNER_NODE_ARB");

        if (errors == 0) begin
            $display("[PASS] tb_router_pe_array_lite");
        end else begin
            $display("[FAIL] tb_router_pe_array_lite errors=%0d", errors);
            $fatal;
        end
        $finish;
    end
endmodule
