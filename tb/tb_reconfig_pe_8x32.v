`timescale 1ns/1ps

module tb_reconfig_pe_8x32;
    localparam PHY_ROWS = 16;
    localparam PHY_COLS = 16;
    localparam DATA_W   = 16;
    localparam ACC_W    = 32;
    localparam CLK_T    = 10;

    localparam CFG_8X32 = 2'b11;

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
    integer i;
    integer wait_i;
    integer idx;
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
    always #(CLK_T/2) clk = ~clk;

    task tick;
        begin
            @(posedge clk);
            #1;
        end
    endtask

    function [15:0] s8_lane;
        input integer value;
        reg [7:0] byte_value;
        begin
            byte_value = value[7:0];
            s8_lane = {{8{byte_value[7]}}, byte_value};
        end
    endfunction

    task apply_reset;
        begin
            rst_n = 1'b0;
            cfg_shape = CFG_8X32;
            mode = 1'b0;
            stat_mode = 1'b1;
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
            repeat (2) tick();
        end
    endtask

    task pulse_flush;
        begin
            en = 1'b1;
            flush = 1'b1;
            w_in = {PHY_COLS*DATA_W{1'b0}};
            act_in = {PHY_ROWS*DATA_W{1'b0}};
            tick();
            flush = 1'b0;
            en = 1'b0;
        end
    endtask

    task check_output_order;
        integer col;
        integer exp_val;
        begin
            apply_reset();

            for (col = 0; col < 16; col = col + 1) begin
                idx = 7 * PHY_COLS + col;
                acc_init[idx*ACC_W +: ACC_W] = 32'h0000_0100 + col[31:0];
                acc_init_mask[idx] = 1'b1;

                idx = 15 * PHY_COLS + col;
                acc_init[idx*ACC_W +: ACC_W] = 32'h0000_0200 + col[31:0];
                acc_init_mask[idx] = 1'b1;
            end

            acc_init_en = 1'b1;
            tick();
            acc_init_en = 1'b0;
            acc_init = {PHY_ROWS*PHY_COLS*ACC_W{1'b0}};
            acc_init_mask = {PHY_ROWS*PHY_COLS{1'b0}};
            tick();

            pulse_flush();

            found_valid = 1'b0;
            for (wait_i = 0; wait_i < 10; wait_i = wait_i + 1) begin
                tick();
                if (valid_out === 32'hFFFF_FFFF) begin
                    found_valid = 1'b1;
                    for (col = 0; col < 32; col = col + 1) begin
                        exp_val = (col < 16) ? (32'h0000_0100 + col)
                                             : (32'h0000_0200 + (col - 16));
                        if (acc_out[col*ACC_W +: ACC_W] !== exp_val[31:0]) begin
                            $display("[FAIL] 8x32 output[%0d] got=0x%08h exp=0x%08h",
                                     col, acc_out[col*ACC_W +: ACC_W], exp_val[31:0]);
                            errors = errors + 1;
                        end
                    end
                end
            end

            if (!found_valid) begin
                $display("[FAIL] 8x32 output order valid_out never reached all 32 bits");
                errors = errors + 1;
            end else begin
                $display("[PASS] 8x32 output order");
            end
        end
    endtask

    task check_folded_activation_route;
        integer cyc;
        begin
            apply_reset();

            // A pulse entering top physical row7 must reappear at lower physical
            // row15 after the top 16-column horizontal path, producing logical
            // output column16 when the lower-half weight stream is active.
            en = 1'b1;
            flush = 1'b0;
            for (cyc = 0; cyc < 40; cyc = cyc + 1) begin
                w_in = {PHY_COLS*DATA_W{1'b0}};
                w_in[0*DATA_W +: DATA_W] = s8_lane(4);
                act_in = {PHY_ROWS*DATA_W{1'b0}};
                if (cyc == 0)
                    act_in[7*DATA_W +: DATA_W] = s8_lane(3);
                tick();
            end
            en = 1'b0;
            w_in = {PHY_COLS*DATA_W{1'b0}};
            act_in = {PHY_ROWS*DATA_W{1'b0}};
            repeat (4) tick();

            pulse_flush();

            found_valid = 1'b0;
            for (wait_i = 0; wait_i < 10; wait_i = wait_i + 1) begin
                tick();
                if (valid_out[16]) begin
                    found_valid = 1'b1;
                    if ($signed(acc_out[16*ACC_W +: ACC_W]) !== 12) begin
                        $display("[FAIL] 8x32 folded route output16 got=%0d exp=12",
                                 $signed(acc_out[16*ACC_W +: ACC_W]));
                        errors = errors + 1;
                    end
                end
            end

            if (!found_valid) begin
                $display("[FAIL] 8x32 folded route valid_out[16] never asserted");
                errors = errors + 1;
            end else begin
                $display("[PASS] 8x32 folded activation route");
            end
        end
    endtask

    task check_ws_load_wrap;
        begin
            apply_reset();
            stat_mode = 1'b0;
            load_w = 1'b1;

            for (i = 0; i < 8; i = i + 1) begin
                if (ws_load_row_out !== i[3:0]) begin
                    $display("[FAIL] 8x32 WS load row before tick%0d got=%0d exp=%0d",
                             i, ws_load_row_out, i);
                    errors = errors + 1;
                end
                tick();
            end

            if (ws_load_row_out !== 4'd0) begin
                $display("[FAIL] 8x32 WS load row did not wrap at 8, got=%0d",
                         ws_load_row_out);
                errors = errors + 1;
            end else begin
                $display("[PASS] 8x32 WS load row wraps at 8");
            end

            load_w = 1'b0;
            tick();
        end
    endtask

    initial begin
        errors = 0;

        check_output_order();
        check_folded_activation_route();
        check_ws_load_wrap();

        if (errors == 0) begin
            $display("[PASS] tb_reconfig_pe_8x32");
        end else begin
            $display("[FAIL] tb_reconfig_pe_8x32 errors=%0d", errors);
            $fatal;
        end
        $finish;
    end
endmodule
