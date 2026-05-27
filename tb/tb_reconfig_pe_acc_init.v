`timescale 1ns/1ps

module tb_reconfig_pe_acc_init;
    localparam PHY_ROWS = 16;
    localparam PHY_COLS = 16;
    localparam DATA_W   = 16;
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
    reg acc_init_en;
    reg [PHY_COLS*DATA_W-1:0] w_in;
    reg [PHY_ROWS*DATA_W-1:0] act_in;
    reg [PHY_COLS*ACC_W-1:0] acc_in;
    reg [PHY_ROWS*PHY_COLS*ACC_W-1:0] acc_init;
    reg [PHY_ROWS*PHY_COLS-1:0] acc_init_mask;

    wire [32*ACC_W-1:0] acc_out;
    wire [31:0] valid_out;
    wire [3:0] ws_load_row_out;
    wire [PHY_ROWS*PHY_COLS-1:0] pe_active;

    integer errors;
    integer r;
    integer c;
    integer idx;
    integer wait_i;
    integer init_val;
    integer prod_val;
    integer exp_val;
    reg found_valid;

    reconfig_pe_array #(
        .PHY_ROWS(PHY_ROWS),
        .PHY_COLS(PHY_COLS),
        .DATA_W  (DATA_W),
        .ACC_W   (ACC_W)
    ) dut (
        .clk            (clk),
        .rst_n          (rst_n),
        .cfg_shape      (cfg_shape),
        .mode           (mode),
        .stat_mode      (stat_mode),
        .en             (en),
        .flush          (flush),
        .load_w         (load_w),
        .swap_w         (swap_w),
        .ws_direct      (1'b0),
        .acc_init_en    (acc_init_en),
        .w_in           (w_in),
        .act_in         (act_in),
        .acc_in         (acc_in),
        .acc_init       (acc_init),
        .acc_init_mask  (acc_init_mask),
        .acc_out        (acc_out),
        .valid_out      (valid_out),
        .ws_load_row_out(ws_load_row_out),
        .pe_active      (pe_active)
    );

    initial clk = 1'b0;
    always #5 clk = ~clk;

    task tick;
        begin
            @(posedge clk);
            #1;
        end
    endtask

    function [15:0] s8_to_lane;
        input integer value;
        reg [7:0] v8;
        begin
            v8 = value[7:0];
            s8_to_lane = {{8{v8[7]}}, v8};
        end
    endfunction

    function integer act_value;
        input integer row;
        begin
            case (row)
                0: act_value = 4;
                1: act_value = -2;
                2: act_value = 3;
                3: act_value = 1;
                default: act_value = 0;
            endcase
        end
    endfunction

    function integer weight_value;
        input integer col;
        begin
            case (col)
                0: weight_value = 2;
                1: weight_value = -3;
                2: weight_value = 5;
                3: weight_value = 7;
                default: weight_value = 0;
            endcase
        end
    endfunction

    task drive_os_cycle;
        input integer cycle_idx;
        integer rr;
        integer cc;
        begin
            w_in = {PHY_COLS*DATA_W{1'b0}};
            act_in = {PHY_ROWS*DATA_W{1'b0}};
            if (cycle_idx == 0) begin
                for (cc = 0; cc < 4; cc = cc + 1)
                    w_in[cc*DATA_W +: DATA_W] = s8_to_lane(weight_value(cc));
            end
            for (rr = 0; rr < 4; rr = rr + 1) begin
                if (rr == cycle_idx)
                    act_in[rr*DATA_W +: DATA_W] = s8_to_lane(act_value(rr));
            end
            en = 1'b1;
            flush = 1'b0;
            tick();
        end
    endtask

    initial begin
        errors = 0;
        rst_n = 1'b0;
        cfg_shape = 2'b00; // 4x4
        mode = 1'b0;       // INT8
        stat_mode = 1'b1;  // OS
        en = 1'b0;
        flush = 1'b0;
        load_w = 1'b0;
        swap_w = 1'b0;
        acc_init_en = 1'b0;
        w_in = {PHY_COLS*DATA_W{1'b0}};
        act_in = {PHY_ROWS*DATA_W{1'b0}};
        acc_in = {PHY_COLS*ACC_W{1'b0}};
        acc_init = {PHY_ROWS*PHY_COLS*ACC_W{1'b0}};
        acc_init_mask = {PHY_ROWS*PHY_COLS{1'b0}};

        repeat (4) tick();
        rst_n = 1'b1;
        tick();

        for (r = 0; r < 4; r = r + 1) begin
            for (c = 0; c < 4; c = c + 1) begin
                idx = r * PHY_COLS + c;
                init_val = 100 + r * 10 + c;
                acc_init[idx*ACC_W +: ACC_W] = init_val[ACC_W-1:0];
                acc_init_mask[idx] = 1'b1;
            end
        end

        acc_init_en = 1'b1;
        tick();
        acc_init_en = 1'b0;
        acc_init = {PHY_ROWS*PHY_COLS*ACC_W{1'b0}};
        tick();

        for (wait_i = 0; wait_i < 4; wait_i = wait_i + 1)
            drive_os_cycle(wait_i);

        en = 1'b0;
        w_in = {PHY_COLS*DATA_W{1'b0}};
        act_in = {PHY_ROWS*DATA_W{1'b0}};
        repeat (4) tick();

        flush = 1'b1;
        en = 1'b1;
        tick();
        flush = 1'b0;
        en = 1'b0;

        found_valid = 1'b0;
        for (wait_i = 0; wait_i < 10; wait_i = wait_i + 1) begin
            tick();
            if (valid_out[15:0] == 16'hFFFF) begin
                found_valid = 1'b1;
                for (r = 0; r < 4; r = r + 1) begin
                    for (c = 0; c < 4; c = c + 1) begin
                        idx = r * 4 + c;
                        init_val = 100 + r * 10 + c;
                        prod_val = act_value(r) * weight_value(c);
                        exp_val = init_val + prod_val;
                        if ($signed(acc_out[idx*ACC_W +: ACC_W]) !== exp_val) begin
                            $display("[FAIL] PE(%0d,%0d) got=%0d expected=%0d",
                                     r, c, $signed(acc_out[idx*ACC_W +: ACC_W]), exp_val);
                            errors = errors + 1;
                        end
                    end
                end
            end
        end

        if (!found_valid) begin
            $display("[FAIL] valid_out[15:0] never asserted together");
            errors = errors + 1;
        end

        if (errors == 0) begin
            $display("[PASS] tb_reconfig_pe_acc_init: 4x4 PE array accumulator init continued MAC passed");
        end else begin
            $display("[FAIL] tb_reconfig_pe_acc_init errors=%0d", errors);
            $fatal;
        end
        $finish;
    end
endmodule
