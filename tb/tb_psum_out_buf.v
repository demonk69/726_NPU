`timescale 1ns/1ps

module tb_psum_out_buf;
    reg clk;
    reg rst_n;
    reg clear;
    reg [15:0] valid_mask;

    reg tile_clear_en;
    reg tile_clear_bank;
    wire tile_clear_done;

    reg port_a_en;
    reg port_a_we;
    reg port_a_bank;
    reg [3:0] port_a_idx;
    reg [31:0] port_a_wdata;
    wire [31:0] port_a_rdata;
    wire port_a_rvalid;

    reg port_b_en;
    reg port_b_we;
    reg port_b_bank;
    reg [3:0] port_b_idx;
    reg [31:0] port_b_wdata;
    wire [31:0] port_b_rdata;
    wire port_b_rvalid;

    wire write_conflict;

    integer errors;
    integer i;
    integer partial;
    reg [31:0] tmp;
    reg [31:0] expected [0:15];

    psum_out_buf #(
        .ACC_W  (32),
        .TILE_M (4),
        .TILE_N (4),
        .BANKS  (2)
    ) dut (
        .clk             (clk),
        .rst_n           (rst_n),
        .clear           (clear),
        .valid_mask      (valid_mask),
        .tile_clear_en   (tile_clear_en),
        .tile_clear_bank (tile_clear_bank),
        .tile_clear_done (tile_clear_done),
        .port_a_en       (port_a_en),
        .port_a_we       (port_a_we),
        .port_a_bank     (port_a_bank),
        .port_a_idx      (port_a_idx),
        .port_a_wdata    (port_a_wdata),
        .port_a_rdata    (port_a_rdata),
        .port_a_rvalid   (port_a_rvalid),
        .port_b_en       (port_b_en),
        .port_b_we       (port_b_we),
        .port_b_bank     (port_b_bank),
        .port_b_idx      (port_b_idx),
        .port_b_wdata    (port_b_wdata),
        .port_b_rdata    (port_b_rdata),
        .port_b_rvalid   (port_b_rvalid),
        .write_conflict  (write_conflict)
    );

    initial clk = 1'b0;
    always #5 clk = ~clk;

    task tick;
        begin
            @(posedge clk);
            #1;
        end
    endtask

    task set_idle;
        begin
            clear = 1'b0;
            tile_clear_en = 1'b0;
            tile_clear_bank = 1'b0;
            port_a_en = 1'b0;
            port_a_we = 1'b0;
            port_a_bank = 1'b0;
            port_a_idx = 4'd0;
            port_a_wdata = 32'd0;
            port_b_en = 1'b0;
            port_b_we = 1'b0;
            port_b_bank = 1'b0;
            port_b_idx = 4'd0;
            port_b_wdata = 32'd0;
        end
    endtask

    task pulse_tile_clear;
        input bank;
        begin
            tile_clear_bank = bank;
            tile_clear_en = 1'b1;
            tick();
            if (!tile_clear_done) begin
                $display("[FAIL] tile_clear_done not asserted");
                errors = errors + 1;
            end
            tile_clear_en = 1'b0;
            tick();
        end
    endtask

    task a_write;
        input bank;
        input [3:0] idx;
        input [31:0] data;
        begin
            port_a_bank = bank;
            port_a_idx = idx;
            port_a_wdata = data;
            port_a_we = 1'b1;
            port_a_en = 1'b1;
            tick();
            port_a_en = 1'b0;
            port_a_we = 1'b0;
            port_a_wdata = 32'd0;
            tick();
        end
    endtask

    task b_write;
        input bank;
        input [3:0] idx;
        input [31:0] data;
        begin
            port_b_bank = bank;
            port_b_idx = idx;
            port_b_wdata = data;
            port_b_we = 1'b1;
            port_b_en = 1'b1;
            tick();
            port_b_en = 1'b0;
            port_b_we = 1'b0;
            port_b_wdata = 32'd0;
            tick();
        end
    endtask

    task a_read_check;
        input bank;
        input [3:0] idx;
        input [31:0] exp_data;
        input exp_valid;
        input [127:0] name;
        begin
            port_a_bank = bank;
            port_a_idx = idx;
            port_a_we = 1'b0;
            port_a_en = 1'b1;
            tick();
            if (port_a_rvalid !== exp_valid) begin
                $display("[FAIL] %0s idx=%0d rvalid=%0b expected=%0b",
                         name, idx, port_a_rvalid, exp_valid);
                errors = errors + 1;
            end
            if (port_a_rdata !== exp_data) begin
                $display("[FAIL] %0s idx=%0d data=%h expected=%h",
                         name, idx, port_a_rdata, exp_data);
                errors = errors + 1;
            end
            port_a_en = 1'b0;
            tick();
        end
    endtask

    task b_read_check;
        input bank;
        input [3:0] idx;
        input [31:0] exp_data;
        input exp_valid;
        input [127:0] name;
        begin
            port_b_bank = bank;
            port_b_idx = idx;
            port_b_we = 1'b0;
            port_b_en = 1'b1;
            tick();
            if (port_b_rvalid !== exp_valid) begin
                $display("[FAIL] %0s idx=%0d rvalid=%0b expected=%0b",
                         name, idx, port_b_rvalid, exp_valid);
                errors = errors + 1;
            end
            if (port_b_rdata !== exp_data) begin
                $display("[FAIL] %0s idx=%0d data=%h expected=%h",
                         name, idx, port_b_rdata, exp_data);
                errors = errors + 1;
            end
            port_b_en = 1'b0;
            tick();
        end
    endtask

    task b_read_value;
        input bank;
        input [3:0] idx;
        output [31:0] data;
        begin
            port_b_bank = bank;
            port_b_idx = idx;
            port_b_we = 1'b0;
            port_b_en = 1'b1;
            tick();
            if (!port_b_rvalid) begin
                $display("[FAIL] b_read_value idx=%0d rvalid low", idx);
                errors = errors + 1;
            end
            data = port_b_rdata;
            port_b_en = 1'b0;
            tick();
        end
    endtask

    initial begin
        errors = 0;
        rst_n = 1'b0;
        valid_mask = 16'hFFFF;
        set_idle();

        repeat (3) tick();
        rst_n = 1'b1;
        tick();

        pulse_tile_clear(1'b0);

        // Load the first k_tile partials through the DMA-side port.
        for (i = 0; i < 16; i = i + 1) begin
            expected[i] = 32'd100 + i[31:0];
            a_write(1'b0, i[3:0], expected[i]);
        end

        for (i = 0; i < 16; i = i + 1)
            b_read_check(1'b0, i[3:0], expected[i], 1'b1, "LOAD_CHECK");

        // K-split read-modify-write. The buffer stores bits only; the modifier
        // can be INT32 or FP32 logic outside this module.
        for (i = 0; i < 16; i = i + 1) begin
            b_read_value(1'b0, i[3:0], tmp);
            partial = (i[0] == 1'b0) ? 50 : -20;
            expected[i] = tmp + partial;
            b_write(1'b0, i[3:0], expected[i]);
        end

        for (i = 0; i < 16; i = i + 1)
            a_read_check(1'b0, i[3:0], expected[i], 1'b1, "RMW_CHECK");

        // Edge tile: active_rows=2, active_cols=3 -> indices 0,1,2,4,5,6.
        valid_mask = 16'h0077;
        pulse_tile_clear(1'b1);
        for (i = 0; i < 16; i = i + 1)
            a_write(1'b1, i[3:0], 32'hA000_0000 + i[31:0]);

        for (i = 0; i < 16; i = i + 1) begin
            if (valid_mask[i])
                b_read_check(1'b1, i[3:0], 32'hA000_0000 + i[31:0], 1'b1, "MASK_VALID");
            else
                b_read_check(1'b1, i[3:0], 32'd0, 1'b0, "MASK_INVALID");
        end

        // Invalid-lane writes were ignored, not just hidden by the mask.
        valid_mask = 16'hFFFF;
        b_read_check(1'b1, 4'd3, 32'd0, 1'b1, "MASK_IGNORED_3");
        b_read_check(1'b1, 4'd7, 32'd0, 1'b1, "MASK_IGNORED_7");

        // Bank isolation.
        a_write(1'b0, 4'd0, 32'h0000_CAFE);
        a_write(1'b1, 4'd0, 32'h0000_BEEF);
        b_read_check(1'b0, 4'd0, 32'h0000_CAFE, 1'b1, "BANK0_ISOLATION");
        b_read_check(1'b1, 4'd0, 32'h0000_BEEF, 1'b1, "BANK1_ISOLATION");

        // Same-address write conflict is flagged and port B wins.
        port_a_bank = 1'b0;
        port_a_idx = 4'd2;
        port_a_wdata = 32'h1111_1111;
        port_a_we = 1'b1;
        port_a_en = 1'b1;
        port_b_bank = 1'b0;
        port_b_idx = 4'd2;
        port_b_wdata = 32'h2222_2222;
        port_b_we = 1'b1;
        port_b_en = 1'b1;
        tick();
        if (!write_conflict) begin
            $display("[FAIL] write_conflict not asserted");
            errors = errors + 1;
        end
        set_idle();
        tick();
        a_read_check(1'b0, 4'd2, 32'h2222_2222, 1'b1, "CONFLICT_B_PRIORITY");

        if (errors == 0) begin
            $display("[PASS] tb_psum_out_buf: K-split read-modify-write accumulation passed");
        end else begin
            $display("[FAIL] tb_psum_out_buf errors=%0d", errors);
            $fatal;
        end
        $finish;
    end
endmodule
