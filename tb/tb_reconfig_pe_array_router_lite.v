`timescale 1ns/1ps

`ifndef DUT_PHY_ROWS
`define DUT_PHY_ROWS 4
`endif
`ifndef DUT_PHY_COLS
`define DUT_PHY_COLS 4
`endif
`ifndef DUT_CFG_SHAPE
`define DUT_CFG_SHAPE 2'b00
`endif

module tb_reconfig_pe_array_router_lite;
    localparam PHY_ROWS = `DUT_PHY_ROWS;
    localparam PHY_COLS = `DUT_PHY_COLS;
    localparam DATA_W   = 64;
    localparam ACC_W    = 32;
    localparam [1:0] CFG_SHAPE = `DUT_CFG_SHAPE;
    localparam ACTIVE_ROWS_RAW = (CFG_SHAPE == 2'b00) ? 4 :
                                 (CFG_SHAPE == 2'b01) ? 8 : 16;
    localparam ACTIVE_COLS_RAW = (CFG_SHAPE == 2'b00) ? 4 :
                                 (CFG_SHAPE == 2'b01) ? 8 : 16;
    localparam ACTIVE_ROWS = (ACTIVE_ROWS_RAW < PHY_ROWS) ? ACTIVE_ROWS_RAW : PHY_ROWS;
    localparam ACTIVE_COLS = (ACTIVE_COLS_RAW < PHY_COLS) ? ACTIVE_COLS_RAW : PHY_COLS;
    localparam ACTIVE_RESULTS = ACTIVE_ROWS * ACTIVE_COLS;

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
    integer r;
    integer c;
    integer idx;
    integer wait_i;
    integer count_i;
    reg [255:0] seen;
    integer seen_count [0:255];

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

    function [63:0] pack_s8x8;
        input integer value;
        begin
            pack_s8x8 = {value[7:0], value[7:0], value[7:0], value[7:0],
                         value[7:0], value[7:0], value[7:0], value[7:0]};
        end
    endfunction

    function integer act_value;
        input integer row;
        begin
            case (row)
                0: act_value = 1;
                1: act_value = -2;
                2: act_value = 3;
                3: act_value = 4;
                4: act_value = -5;
                5: act_value = 6;
                6: act_value = 7;
                7: act_value = -8;
                default: act_value = 0;
            endcase
        end
    endfunction

    function integer weight_value;
        input integer col;
        begin
            case (col)
                0: weight_value = 2;
                1: weight_value = 5;
                2: weight_value = -3;
                3: weight_value = 1;
                4: weight_value = 4;
                5: weight_value = -6;
                6: weight_value = 7;
                7: weight_value = 3;
                default: weight_value = 0;
            endcase
        end
    endfunction

    function [31:0] expected_dot;
        input integer row;
        input integer col;
        input integer scale;
        integer value;
        begin
            value = scale * 8 * act_value(row) * weight_value(col);
            expected_dot = value[31:0];
        end
    endfunction

    function integer ws_weight_value;
        input integer krow;
        input integer col;
        begin
            ws_weight_value = ((krow + 1) * (col + 2));
            if ((krow + col) & 1)
                ws_weight_value = -ws_weight_value;
        end
    endfunction

    function integer ws_act_value;
        input integer out_row;
        input integer krow;
        begin
            ws_act_value = (out_row + 1) * (krow + 1);
            if (krow & 1)
                ws_act_value = -ws_act_value;
        end
    endfunction

    function [31:0] expected_ws_dot;
        input integer out_row;
        input integer col;
        integer krow;
        integer value;
        begin
            value = 0;
            for (krow = 0; krow < ACTIVE_ROWS; krow = krow + 1)
                value = value + (8 * ws_act_value(out_row, krow) * ws_weight_value(krow, col));
            expected_ws_dot = value[31:0];
        end
    endfunction

    task reset_dut;
        begin
            rst_n = 1'b0;
            cfg_shape = CFG_SHAPE;
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
            repeat (6) tick();
            rst_n = 1'b1;
            tick();
        end
    endtask

    task drive_active_beat_scaled;
        input do_flush;
        input integer act_scale;
        begin
            w_in = {PHY_COLS*DATA_W{1'b0}};
            act_in = {PHY_ROWS*DATA_W{1'b0}};
            for (c = 0; c < ACTIVE_COLS; c = c + 1)
                w_in[c*DATA_W +: DATA_W] = pack_s8x8(weight_value(c));
            for (r = 0; r < ACTIVE_ROWS; r = r + 1)
                act_in[r*DATA_W +: DATA_W] = pack_s8x8(act_scale * act_value(r));
            en = 1'b1;
            flush = do_flush;
            tick();
            en = 1'b0;
            flush = 1'b0;
            w_in = {PHY_COLS*DATA_W{1'b0}};
            act_in = {PHY_ROWS*DATA_W{1'b0}};
        end
    endtask

    task drive_active_beat;
        input do_flush;
        begin
            drive_active_beat_scaled(do_flush, 1);
        end
    endtask

    task wait_router_ready_for;
        input [255:0] name;
        begin
            for (wait_i = 0; wait_i < 5000 && router_ready !== 1'b1; wait_i = wait_i + 1)
                tick();
            if (router_ready !== 1'b1) begin
                $display("[FAIL] %0s router_ready timeout", name);
                errors = errors + 1;
            end
        end
    endtask

    task drive_ws_weight_row;
        input integer krow;
        begin
            wait_router_ready_for("WS_WEIGHT_READY");
            w_in = {PHY_COLS*DATA_W{1'b0}};
            act_in = {PHY_ROWS*DATA_W{1'b0}};
            for (c = 0; c < ACTIVE_COLS; c = c + 1)
                w_in[c*DATA_W +: DATA_W] = pack_s8x8(ws_weight_value(krow, c));
            stat_mode = 1'b0;
            ws_direct = 1'b0;
            en = 1'b1;
            load_w = 1'b1;
            flush = 1'b0;
            tick();
            en = 1'b0;
            load_w = 1'b0;
            w_in = {PHY_COLS*DATA_W{1'b0}};
        end
    endtask

    task drive_ws_act_row;
        input integer out_row;
        integer rr;
        begin
            wait_router_ready_for("WS_ACT_READY");
            w_in = {PHY_COLS*DATA_W{1'b0}};
            act_in = {PHY_ROWS*DATA_W{1'b0}};
            for (rr = 0; rr < ACTIVE_ROWS; rr = rr + 1)
                act_in[rr*DATA_W +: DATA_W] = pack_s8x8(ws_act_value(out_row, rr));
            stat_mode = 1'b0;
            ws_direct = 1'b0;
            en = 1'b1;
            flush = 1'b0;
            tick();
            en = 1'b0;
            act_in = {PHY_ROWS*DATA_W{1'b0}};
        end
    endtask

    task expect_active_results;
        input integer scale;
        input [255:0] name;
        begin
            seen = 256'd0;
            for (wait_i = 0; wait_i < 1600; wait_i = wait_i + 1) begin
                tick();
                for (r = 0; r < ACTIVE_ROWS; r = r + 1) begin
                    for (c = 0; c < ACTIVE_COLS; c = c + 1) begin
                        idx = r * ACTIVE_COLS + c;
                        if (valid_out[idx]) begin
                            seen[idx] = 1'b1;
                            if (acc_out[idx*ACC_W +: ACC_W] !== expected_dot(r, c, scale)) begin
                                $display("[FAIL] %0s PE(%0d,%0d) got=0x%08h expected=0x%08h",
                                         name, r, c, acc_out[idx*ACC_W +: ACC_W],
                                         expected_dot(r, c, scale));
                                errors = errors + 1;
                            end
                        end
                    end
                end
            end

            if (seen[ACTIVE_RESULTS-1:0] !== {ACTIVE_RESULTS{1'b1}}) begin
                $display("[FAIL] %0s missing router PE valids seen=0x%064h", name, seen);
                errors = errors + 1;
            end
        end
    endtask

    task expect_two_active_result_waves;
        input integer first_scale;
        input integer second_scale;
        input [255:0] name;
        reg [31:0] expected;
        begin
            for (count_i = 0; count_i < 256; count_i = count_i + 1)
                seen_count[count_i] = 0;

            for (wait_i = 0; wait_i < 2400; wait_i = wait_i + 1) begin
                tick();
                for (r = 0; r < ACTIVE_ROWS; r = r + 1) begin
                    for (c = 0; c < ACTIVE_COLS; c = c + 1) begin
                        idx = r * ACTIVE_COLS + c;
                        if (valid_out[idx]) begin
                            if (seen_count[idx] == 0)
                                expected = expected_dot(r, c, first_scale);
                            else if (seen_count[idx] == 1)
                                expected = expected_dot(r, c, second_scale);
                            else begin
                                expected = 32'hxxxx_xxxx;
                                $display("[FAIL] %0s PE(%0d,%0d) extra valid count=%0d",
                                         name, r, c, seen_count[idx]);
                                errors = errors + 1;
                            end

                            if (seen_count[idx] < 2 && acc_out[idx*ACC_W +: ACC_W] !== expected) begin
                                $display("[FAIL] %0s PE(%0d,%0d) wave=%0d got=0x%08h expected=0x%08h",
                                         name, r, c, seen_count[idx],
                                         acc_out[idx*ACC_W +: ACC_W], expected);
                                errors = errors + 1;
                            end
                            seen_count[idx] = seen_count[idx] + 1;
                        end
                    end
                end
            end

            for (count_i = 0; count_i < ACTIVE_RESULTS; count_i = count_i + 1) begin
                if (seen_count[count_i] != 2) begin
                    $display("[FAIL] %0s result%0d valid_count=%0d expected=2",
                             name, count_i, seen_count[count_i]);
                    errors = errors + 1;
                end
            end
        end
    endtask

    task expect_ws_results;
        input [255:0] name;
        integer wave;
        integer bottom_base;
        reg [31:0] expected;
        begin
            wave = 0;
            bottom_base = (ACTIVE_ROWS - 1) * ACTIVE_COLS;
            for (wait_i = 0; wait_i < 8000 && wave < ACTIVE_ROWS; wait_i = wait_i + 1) begin
                tick();
                if (valid_out[bottom_base]) begin
                    for (c = 0; c < ACTIVE_COLS; c = c + 1) begin
                        if (!valid_out[bottom_base + c]) begin
                            $display("[FAIL] %0s wave=%0d col=%0d missing bottom valid",
                                     name, wave, c);
                            errors = errors + 1;
                        end
                        expected = expected_ws_dot(wave, c);
                        if (acc_out[(bottom_base + c)*ACC_W +: ACC_W] !== expected) begin
                            $display("[FAIL] %0s wave=%0d col=%0d got=0x%08h expected=0x%08h",
                                     name, wave, c,
                                     acc_out[(bottom_base + c)*ACC_W +: ACC_W], expected);
                            errors = errors + 1;
                        end
                    end
                    wave = wave + 1;
                end
            end

            if (wave != ACTIVE_ROWS) begin
                $display("[FAIL] %0s captured_waves=%0d expected=%0d", name, wave, ACTIVE_ROWS);
                errors = errors + 1;
            end
        end
    endtask

    task expect_ws_result_wave;
        input integer out_row;
        input [255:0] name;
        integer bottom_base;
        reg [31:0] expected;
        reg found;
        begin
            bottom_base = (ACTIVE_ROWS - 1) * ACTIVE_COLS;
            found = 1'b0;
            for (wait_i = 0; wait_i < 8000 && !found; wait_i = wait_i + 1) begin
                tick();
                if (valid_out[bottom_base]) begin
                    found = 1'b1;
                    for (c = 0; c < ACTIVE_COLS; c = c + 1) begin
                        if (!valid_out[bottom_base + c]) begin
                            $display("[FAIL] %0s row=%0d col=%0d missing bottom valid",
                                     name, out_row, c);
                            errors = errors + 1;
                        end
                        expected = expected_ws_dot(out_row, c);
                        if (acc_out[(bottom_base + c)*ACC_W +: ACC_W] !== expected) begin
                            $display("[FAIL] %0s row=%0d col=%0d got=0x%08h expected=0x%08h",
                                     name, out_row, c,
                                     acc_out[(bottom_base + c)*ACC_W +: ACC_W], expected);
                            errors = errors + 1;
                        end
                    end
                end
            end

            if (!found) begin
                $display("[FAIL] %0s row=%0d no bottom valid", name, out_row);
                errors = errors + 1;
            end
        end
    endtask

    task expect_router_not_ready;
        input [255:0] name;
        begin
            tick();
            if (router_ready !== 1'b0) begin
                $display("[FAIL] %0s router_ready expected 0 got %0b", name, router_ready);
                errors = errors + 1;
            end
            if (router_overflow !== 1'b0) begin
                $display("[FAIL] %0s router_overflow asserted while idle", name);
                errors = errors + 1;
            end
        end
    endtask

    initial begin
        errors = 0;
        seen = 256'd0;

        reset_dut();
        mode = 1'b1;
        stat_mode = 1'b1;
        expect_router_not_ready("UNSUPPORTED_FP16_READY");
        mode = 1'b0;
        stat_mode = 1'b0;
        tick();
        if (router_ready !== 1'b1) begin
            $display("[FAIL] SUPPORTED_WS_READY router_ready expected 1 got %0b", router_ready);
            errors = errors + 1;
        end

        reset_dut();
        drive_active_beat(1'b1);
        expect_active_results(1, "SINGLE_BEAT_FLUSH");

        reset_dut();
        drive_active_beat(1'b0);
        for (wait_i = 0; wait_i < 80; wait_i = wait_i + 1) begin
            tick();
            if (valid_out[ACTIVE_RESULTS-1:0] != {ACTIVE_RESULTS{1'b0}}) begin
                $display("[FAIL] NON_FLUSH_BEAT produced valid_out=0x%064h", valid_out);
                errors = errors + 1;
            end
        end
        drive_active_beat(1'b1);
        expect_active_results(2, "TWO_BEAT_ACCUM_FLUSH");

        reset_dut();
        drive_active_beat_scaled(1'b1, 1);
        drive_active_beat_scaled(1'b1, 2);
        expect_two_active_result_waves(1, 2, "BUSY_SKID_TWO_FLUSH_BEATS");

        if (router_overflow !== 1'b0) begin
            $display("[FAIL] router_overflow asserted during skid-buffer test");
            errors = errors + 1;
        end

        reset_dut();
        for (r = 0; r < ACTIVE_ROWS; r = r + 1)
            drive_ws_weight_row(r);
        wait_router_ready_for("WS_PRELOAD_DONE");
        swap_w = 1'b1;
        tick();
        swap_w = 1'b0;
        repeat (16) tick();
        for (r = 0; r < ACTIVE_ROWS; r = r + 1) begin
            drive_ws_act_row(r);
            expect_ws_result_wave(r, "TRUE_WS_VERTICAL_PSUM");
        end

        if (errors == 0) begin
            $display("[PASS] tb_reconfig_pe_array_router_lite rows=%0d cols=%0d shape=%0d",
                     PHY_ROWS, PHY_COLS, CFG_SHAPE);
        end else begin
            $display("[FAIL] tb_reconfig_pe_array_router_lite errors=%0d", errors);
            $fatal;
        end
        $finish;
    end
endmodule
