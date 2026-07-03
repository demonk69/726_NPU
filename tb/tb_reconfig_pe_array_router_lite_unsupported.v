`timescale 1ns/1ps

module tb_reconfig_pe_array_router_lite_unsupported;
    localparam PHY_ROWS = 4;
    localparam PHY_COLS = 4;
    localparam DATA_W   = 64;
    localparam ACC_W    = 32;

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
    wire router_ready;
    wire router_overflow;

    integer errors;
    integer i;

    reconfig_pe_array #(
        .PHY_ROWS(PHY_ROWS),
        .PHY_COLS(PHY_COLS),
        .DATA_W(DATA_W),
        .ACC_W(ACC_W),
        .INT8_SIMD_LANES(8),
        .FP16_ENABLE(0),
        .INT8_SCALAR_SIGNEXT_COMPAT(0),
        .USE_ROUTER_MESH(1)
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
        .router_enable(1'b1),
        .os_act_systolic(1'b0),
        .os_weight_broadcast(1'b0),
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
        .router_ready(router_ready),
        .router_overflow(router_overflow)
    );

    initial clk = 1'b0;
    always #5 clk = ~clk;

    task tick;
        begin
            @(posedge clk);
            #1;
        end
    endtask

    task reset_dut;
        begin
            rst_n = 1'b0;
            cfg_shape = 2'b00;
            mode = 1'b0;
            stat_mode = 1'b1;
            en = 1'b0;
            flush = 1'b0;
            load_w = 1'b0;
            swap_w = 1'b0;
            ws_direct = 1'b0;
            acc_init_en = 1'b0;
            half_en = 1'b0;
            array_ce = 1'b1;
            row_ce = {PHY_ROWS{1'b1}};
            col_ce = {PHY_COLS{1'b1}};
            w_in = {PHY_COLS*DATA_W{1'b0}};
            act_in = {PHY_ROWS*DATA_W{1'b0}};
            acc_in = {PHY_COLS*ACC_W{1'b0}};
            acc_init = {PHY_ROWS*PHY_COLS*ACC_W{1'b0}};
            acc_init_mask = {PHY_ROWS*PHY_COLS{1'b0}};
            repeat (4) tick();
            rst_n = 1'b1;
            tick();
        end
    endtask

    task expect_unsupported_error;
        input [1:0] shape;
        input data_mode;
        input stationary_mode;
        input [255:0] name;
        begin
            reset_dut();
            cfg_shape = shape;
            mode = data_mode;
            stat_mode = stationary_mode;
            tick();
            if (router_ready !== 1'b0) begin
                $display("[FAIL] %0s router_ready expected 0 got %0b", name, router_ready);
                errors = errors + 1;
            end

            en = 1'b1;
            flush = 1'b1;
            w_in = {PHY_COLS*DATA_W{1'b1}};
            act_in = {PHY_ROWS*DATA_W{1'b1}};
            tick();
            en = 1'b0;
            flush = 1'b0;

            if (router_overflow !== 1'b1) begin
                $display("[FAIL] %0s router_overflow expected 1 got %0b", name, router_overflow);
                errors = errors + 1;
            end

            for (i = 0; i < 20; i = i + 1) begin
                tick();
                if (valid_out[15:0] !== 16'd0) begin
                    $display("[FAIL] %0s produced valid_out=0x%04h", name, valid_out[15:0]);
                    errors = errors + 1;
                end
            end
        end
    endtask

    initial begin
`ifndef SYNTHESIS
        $display("[ERROR] tb_reconfig_pe_array_router_lite_unsupported requires +define+SYNTHESIS to observe hardware error output without simulator fatal");
        $finish;
`else
        errors = 0;
        expect_unsupported_error(2'b00, 1'b1, 1'b1, "FP16_UNSUPPORTED");
        expect_unsupported_error(2'b00, 1'b1, 1'b0, "FP16_WS_UNSUPPORTED");

        if (errors == 0) begin
            $display("[PASS] tb_reconfig_pe_array_router_lite_unsupported");
        end else begin
            $display("[FAIL] tb_reconfig_pe_array_router_lite_unsupported errors=%0d", errors);
            $fatal;
        end
        $finish;
`endif
    end
endmodule
