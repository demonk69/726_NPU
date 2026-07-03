`timescale 1ns/1ps

module tb_reconfig_pe_array_ws_psum;
    localparam PHY_ROWS = 4;
    localparam PHY_COLS = 4;
    localparam DATA_W   = 16;
    localparam ACC_W    = 32;
    localparam BOTTOM_COL0_IDX = 12; // row 3, col 0 in 4x4 row-major output

    reg clk;
    reg rst_n;
    reg [1:0] cfg_shape;
    reg mode;
    reg stat_mode;
    reg en;
    reg flush;
    reg load_w;
    reg swap_w;
    reg ws_direct;
    reg acc_init_en;
    reg half_en;
    reg array_ce;
    reg router_enable;
    reg os_act_systolic;
    reg os_weight_broadcast;
    reg [PHY_ROWS-1:0] row_ce;
    reg [PHY_COLS-1:0] col_ce;
    reg [PHY_COLS*DATA_W-1:0] w_in;
    reg [PHY_ROWS*DATA_W-1:0] act_in;
    reg [PHY_COLS*ACC_W-1:0] acc_in;
    reg [PHY_ROWS*PHY_COLS*ACC_W-1:0] acc_init;
    reg [PHY_ROWS*PHY_COLS-1:0] acc_init_mask;

    wire [256*ACC_W-1:0] acc_out;
    wire [255:0] valid_out;
    wire [3:0] ws_load_row_out;
    wire [PHY_ROWS*PHY_COLS-1:0] pe_active;

    integer row;
    integer cyc;
    integer errors;
    reg saw_bottom_sum;

    reconfig_pe_array #(
        .PHY_ROWS(PHY_ROWS),
        .PHY_COLS(PHY_COLS),
        .DATA_W(DATA_W),
        .ACC_W(ACC_W)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .cfg_shape(cfg_shape),
        .mode(mode),
        .stat_mode(stat_mode),
        .en(en),
        .flush(flush),
        .load_w(load_w),
        .swap_w(swap_w),
        .ws_direct(ws_direct),
        .acc_init_en(acc_init_en),
        .half_en(half_en),
        .array_ce(array_ce),
        .router_enable(router_enable),
        .os_act_systolic(os_act_systolic),
        .os_weight_broadcast(os_weight_broadcast),
        .row_ce(row_ce),
        .col_ce(col_ce),
        .w_in(w_in),
        .act_in(act_in),
        .acc_in(acc_in),
        .acc_init(acc_init),
        .acc_init_mask(acc_init_mask),
        .acc_out(acc_out),
        .valid_out(valid_out),
        .ws_load_row_out(ws_load_row_out),
        .pe_active(pe_active),
        .router_ready(),
        .router_overflow()
    );

    initial clk = 1'b0;
    always #5 clk = ~clk;

    function [15:0] s8_scalar;
        input integer value;
        reg [7:0] v;
        begin
            v = value[7:0];
            s8_scalar = {{8{v[7]}}, v};
        end
    endfunction

    task set_col0_weight;
        input integer value;
        begin
            w_in = {PHY_COLS*DATA_W{1'b0}};
            w_in[0*DATA_W +: DATA_W] = s8_scalar(value);
        end
    endtask

    task reset_dut;
        begin
            rst_n = 1'b0;
            cfg_shape = 2'b00;
            mode = 1'b0;
            stat_mode = 1'b0;
            en = 1'b0;
            flush = 1'b0;
            load_w = 1'b0;
            swap_w = 1'b0;
            ws_direct = 1'b0;
            acc_init_en = 1'b0;
            half_en = 1'b0;
            array_ce = 1'b1;
            router_enable = 1'b0;
            os_act_systolic = 1'b0;
            os_weight_broadcast = 1'b0;
            row_ce = {PHY_ROWS{1'b1}};
            col_ce = {PHY_COLS{1'b1}};
            w_in = {PHY_COLS*DATA_W{1'b0}};
            act_in = {PHY_ROWS*DATA_W{1'b0}};
            acc_in = {PHY_COLS*ACC_W{1'b0}};
            acc_init = {PHY_ROWS*PHY_COLS*ACC_W{1'b0}};
            acc_init_mask = {PHY_ROWS*PHY_COLS{1'b0}};
            repeat (4) @(posedge clk);
            rst_n = 1'b1;
            repeat (2) @(posedge clk);
        end
    endtask

    task load_weight_rows;
        begin
            @(negedge clk);
            load_w = 1'b1;
            for (row = 0; row < PHY_ROWS; row = row + 1) begin
                set_col0_weight(row + 1);
                @(negedge clk);
            end
            load_w = 1'b0;
            w_in = {PHY_COLS*DATA_W{1'b0}};
        end
    endtask

    task swap_weights_active;
        begin
            @(negedge clk);
            swap_w = 1'b1;
            @(negedge clk);
            swap_w = 1'b0;
        end
    endtask

    initial begin
        errors = 0;
        saw_bottom_sum = 1'b0;
        reset_dut();

        load_weight_rows();
        swap_weights_active();

        @(negedge clk);
        en = 1'b1;
        act_in = {PHY_ROWS*DATA_W{1'b0}};
        act_in[0*DATA_W +: DATA_W] = s8_scalar(1);
        act_in[1*DATA_W +: DATA_W] = s8_scalar(2);
        act_in[2*DATA_W +: DATA_W] = s8_scalar(3);
        act_in[3*DATA_W +: DATA_W] = s8_scalar(4);

        for (cyc = 0; cyc < 40; cyc = cyc + 1) begin
            @(posedge clk);
            #1;
            if (valid_out[BOTTOM_COL0_IDX]) begin
                if (acc_out[BOTTOM_COL0_IDX*ACC_W +: ACC_W] == 32'd30)
                    saw_bottom_sum = 1'b1;
            end
        end

        @(negedge clk);
        en = 1'b0;
        act_in = {PHY_ROWS*DATA_W{1'b0}};

        if (!saw_bottom_sum) begin
            $display("[FAIL] WS psum bottom never produced expected weighted sum 30");
            errors = errors + 1;
        end else begin
            $display("[PASS] WS psum bottom produced expected weighted sum 30");
        end

        if (errors == 0) begin
            $display("[PASS] tb_reconfig_pe_array_ws_psum");
        end else begin
            $display("[FAIL] tb_reconfig_pe_array_ws_psum errors=%0d", errors);
            $fatal;
        end
        $finish;
    end
endmodule
