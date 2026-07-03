`timescale 1ns/1ps

module tb_pe_top_ws_psum;
    localparam DATA_W = 16;
    localparam ACC_W  = 32;

    reg clk;
    reg rst_n;
    reg mode;
    reg stat_mode;
    reg en;
    reg flush;
    reg load_w;
    reg swap_w;
    reg acc_init_en;
    reg [DATA_W-1:0] w_in;
    reg [DATA_W-1:0] a_in;
    reg [ACC_W-1:0]  acc_in;
    reg [ACC_W-1:0]  acc_init;
    wire [ACC_W-1:0] acc_out;
    wire valid_out;

    integer errors;

    pe_top #(
        .DATA_W(DATA_W),
        .ACC_W(ACC_W)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .mode(mode),
        .stat_mode(stat_mode),
        .en(en),
        .flush(flush),
        .load_w(load_w),
        .swap_w(swap_w),
        .acc_init_en(acc_init_en),
        .w_in(w_in),
        .a_in(a_in),
        .acc_in(acc_in),
        .acc_init(acc_init),
        .acc_out(acc_out),
        .valid_out(valid_out)
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

    task reset_dut;
        begin
            rst_n = 1'b0;
            mode = 1'b0;
            stat_mode = 1'b0;
            en = 1'b0;
            flush = 1'b0;
            load_w = 1'b0;
            swap_w = 1'b0;
            acc_init_en = 1'b0;
            w_in = {DATA_W{1'b0}};
            a_in = {DATA_W{1'b0}};
            acc_in = {ACC_W{1'b0}};
            acc_init = {ACC_W{1'b0}};
            repeat (4) @(posedge clk);
            rst_n = 1'b1;
            repeat (2) @(posedge clk);
        end
    endtask

    task prefetch_weight;
        input [DATA_W-1:0] weight;
        input do_swap;
        begin
            @(negedge clk);
            w_in = weight;
            a_in = {DATA_W{1'b0}};
            acc_in = {ACC_W{1'b0}};
            load_w = 1'b1;
            swap_w = do_swap;
            en = 1'b0;
            @(negedge clk);
            load_w = 1'b0;
            swap_w = 1'b0;
            w_in = {DATA_W{1'b0}};
        end
    endtask

    task swap_active;
        begin
            @(negedge clk);
            swap_w = 1'b1;
            load_w = 1'b0;
            en = 1'b0;
            @(negedge clk);
            swap_w = 1'b0;
        end
    endtask

    task compute_expect_ctl;
        input [DATA_W-1:0] weight;
        input [DATA_W-1:0] act;
        input [ACC_W-1:0]  psum;
        input do_load;
        input do_swap;
        input [ACC_W-1:0]  expected;
        input [255:0]      name;
        integer guard;
        reg seen;
        begin
            @(negedge clk);
            w_in = weight;
            a_in = act;
            acc_in = psum;
            load_w = do_load;
            swap_w = do_swap;
            flush = 1'b0;
            en = 1'b1;

            @(negedge clk);
            en = 1'b0;
            load_w = 1'b0;
            swap_w = 1'b0;
            w_in = {DATA_W{1'b0}};
            a_in = {DATA_W{1'b0}};
            acc_in = {ACC_W{1'b0}};

            seen = 1'b0;
            for (guard = 0; guard < 8; guard = guard + 1) begin
                @(posedge clk);
                #1;
                if (valid_out && !seen) begin
                    seen = 1'b1;
                    if (acc_out !== expected) begin
                        $display("[FAIL] %0s got=0x%08h exp=0x%08h", name, acc_out, expected);
                        errors = errors + 1;
                    end else begin
                        $display("[PASS] %0s", name);
                    end
                end
            end
            if (!seen) begin
                $display("[FAIL] %0s no valid_out", name);
                errors = errors + 1;
            end
        end
    endtask

    initial begin
        errors = 0;
        reset_dut();

        // load_w only fills prefetch; active remains reset-zero until swap_w.
        prefetch_weight(s8_scalar(3), 1'b0);
        compute_expect_ctl(16'd0, s8_scalar(5), 32'd7, 1'b0, 1'b0,
                           32'd7, "LOAD_NO_SWAP_ACTIVE_ZERO");

        swap_active();
        compute_expect_ctl(16'd0, s8_scalar(5), 32'd7, 1'b0, 1'b0,
                           32'd22, "SWAP_MAKES_WEIGHT_ACTIVE");

        // Loading a new prefetch weight while computing must not perturb active.
        compute_expect_ctl(s8_scalar(4), s8_scalar(2), 32'd1, 1'b1, 1'b0,
                           32'd7, "LOAD_DURING_COMPUTE_KEEPS_ACTIVE");

        swap_active();
        compute_expect_ctl(16'd0, s8_scalar(2), 32'd1, 1'b0, 1'b0,
                           32'd9, "SWAPPED_PREFETCH_WEIGHT_ACTIVE");

        // swap_w+load_w in one cycle: old prefetch becomes active, new w_in
        // goes into the register that just became inactive.
        prefetch_weight(s8_scalar(5), 1'b0);
        prefetch_weight(s8_scalar(6), 1'b1);
        compute_expect_ctl(16'd0, s8_scalar(1), 32'd0, 1'b0, 1'b0,
                           32'd5, "SWAP_LOAD_OLD_PREFETCH_ACTIVE");
        swap_active();
        compute_expect_ctl(16'd0, s8_scalar(1), 32'd0, 1'b0, 1'b0,
                           32'd6, "AFTER_SWAP_LOAD_NEW_PREFETCH");

        if (errors == 0) begin
            $display("[PASS] tb_pe_top_ws_psum");
        end else begin
            $display("[FAIL] tb_pe_top_ws_psum errors=%0d", errors);
            $fatal;
        end
        $finish;
    end
endmodule
