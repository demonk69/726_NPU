`timescale 1ns/1ps

module tb_npu_tile_lane_feed;
    localparam DATA_W = 16;
    localparam ACC_W  = 32;
    localparam CLK_T  = 10;
    localparam DRAM_WORDS = 1024;

    localparam REG_CTRL      = 32'h00;
    localparam REG_M_DIM     = 32'h10;
    localparam REG_N_DIM     = 32'h14;
    localparam REG_K_DIM     = 32'h18;
    localparam REG_W_ADDR    = 32'h20;
    localparam REG_A_ADDR    = 32'h24;
    localparam REG_R_ADDR    = 32'h28;
    localparam REG_ARR_CFG   = 32'h30;
    localparam REG_CFG_SHAPE = 32'h3C;

    localparam W_ADDR = 32'h0000_0100;
    localparam A_ADDR = 32'h0000_0200;
    localparam R_ADDR = 32'h0000_0300;
    localparam CTRL_INT8_OS = 32'h0000_0011;
    localparam ARR_TILE = 32'h0000_0080;

    reg clk = 1'b0;
    always #(CLK_T/2) clk = ~clk;

    reg rst_n;

    reg  [31:0] s_awaddr, s_wdata;
    reg  [3:0]  s_wstrb;
    reg         s_awvalid, s_wvalid, s_bready;
    wire        s_awready, s_wready, s_bvalid;
    wire [1:0]  s_bresp;

    reg  [31:0] s_araddr;
    reg         s_arvalid, s_rready;
    wire        s_arready, s_rvalid;
    wire [31:0] s_rdata;
    wire [1:0]  s_rresp;

    wire [31:0] m_awaddr;
    wire [7:0]  m_awlen;
    wire [2:0]  m_awsize;
    wire [1:0]  m_awburst;
    wire        m_awvalid;
    reg         m_awready;
    wire [31:0] m_wdata;
    wire [3:0]  m_wstrb;
    wire        m_wlast;
    wire        m_wvalid;
    reg         m_wready;
    reg         m_bvalid;
    wire        m_bready;
    wire [1:0]  m_bresp = 2'b00;

    wire [31:0] m_araddr;
    wire [7:0]  m_arlen;
    wire [2:0]  m_arsize;
    wire [1:0]  m_arburst;
    wire        m_arvalid;
    reg         m_arready;
    reg  [31:0] m_rdata;
    reg         m_rvalid;
    reg         m_rlast;
    wire        m_rready;
    wire [1:0]  m_rresp = 2'b00;

    wire npu_irq;

    reg [31:0] dram [0:DRAM_WORDS-1];
    integer errors;
    integer i;

    npu_top #(
        .DATA_W(DATA_W),
        .ACC_W (ACC_W)
    ) u_npu (
        .sys_clk       (clk),
        .sys_rst_n     (rst_n),
        .s_axi_awaddr  (s_awaddr),
        .s_axi_awvalid (s_awvalid),
        .s_axi_awready (s_axi_awready),
        .s_axi_wdata   (s_wdata),
        .s_axi_wstrb   (s_wstrb),
        .s_axi_wvalid  (s_wvalid),
        .s_axi_wready  (s_axi_wready),
        .s_axi_bresp   (s_bresp),
        .s_axi_bvalid  (s_bvalid),
        .s_axi_bready  (s_bready),
        .s_axi_araddr  (s_araddr),
        .s_axi_arvalid (s_arvalid),
        .s_axi_arready (s_arready),
        .s_axi_rdata   (s_rdata),
        .s_axi_rresp   (s_rresp),
        .s_axi_rvalid  (s_rvalid),
        .s_axi_rready  (s_rready),
        .m_axi_awaddr  (m_awaddr),
        .m_axi_awlen   (m_awlen),
        .m_axi_awsize  (m_awsize),
        .m_axi_awburst (m_awburst),
        .m_axi_awvalid (m_awvalid),
        .m_axi_awready (m_awready),
        .m_axi_wdata   (m_wdata),
        .m_axi_wstrb   (m_wstrb),
        .m_axi_wlast   (m_wlast),
        .m_axi_wvalid  (m_wvalid),
        .m_axi_wready  (m_wready),
        .m_axi_bresp   (m_bresp),
        .m_axi_bvalid  (m_bvalid),
        .m_axi_bready  (m_bready),
        .m_axi_araddr  (m_araddr),
        .m_axi_arlen   (m_arlen),
        .m_axi_arsize  (m_arsize),
        .m_axi_arburst (m_arburst),
        .m_axi_arvalid (m_arvalid),
        .m_axi_arready (m_arready),
        .m_axi_rdata   (m_rdata),
        .m_axi_rresp   (m_rresp),
        .m_axi_rvalid  (m_rvalid),
        .m_axi_rready  (m_rready),
        .m_axi_rlast   (m_rlast),
        .npu_irq       (npu_irq)
    );

    reg [31:0] rd_base;
    reg [7:0]  rd_len;
    reg [7:0]  rd_cnt;
    reg        rd_active;

    always @(posedge clk) begin
        if (!rst_n) begin
            m_arready <= 1'b0;
            m_rvalid  <= 1'b0;
            m_rlast   <= 1'b0;
            m_rdata   <= 32'd0;
            rd_active <= 1'b0;
            rd_cnt    <= 8'd0;
            rd_len    <= 8'd0;
            rd_base   <= 32'd0;
        end else begin
            m_arready <= 1'b1;
            if (m_rvalid && m_rready && m_rlast) begin
                m_rvalid <= 1'b0;
                m_rlast  <= 1'b0;
            end else if (!rd_active && !m_rvalid && m_arvalid && m_arready) begin
                rd_active <= 1'b1;
                rd_base   <= m_araddr;
                rd_len    <= m_arlen;
                rd_cnt    <= 8'd0;
            end else if (rd_active && (!m_rvalid || (m_rvalid && m_rready))) begin
                m_rdata  <= dram[((rd_base >> 2) + rd_cnt) % DRAM_WORDS];
                m_rvalid <= 1'b1;
                m_rlast  <= (rd_cnt >= rd_len);
                if (rd_cnt >= rd_len)
                    rd_active <= 1'b0;
                else
                    rd_cnt <= rd_cnt + 1'b1;
            end
        end
    end

    reg [31:0] wr_base;
    reg [7:0]  wr_cnt;
    reg        wr_active;
    reg        b_pending;

    always @(posedge clk) begin
        if (!rst_n) begin
            m_awready <= 1'b0;
            m_wready  <= 1'b0;
            m_bvalid  <= 1'b0;
            wr_active <= 1'b0;
            b_pending <= 1'b0;
            wr_cnt    <= 8'd0;
            wr_base   <= 32'd0;
        end else begin
            m_awready <= 1'b1;
            if (m_awvalid && m_awready && !wr_active) begin
                wr_active <= 1'b1;
                wr_base   <= m_awaddr;
                wr_cnt    <= 8'd0;
                m_wready  <= 1'b1;
            end
            if (wr_active && m_wvalid && m_wready) begin
                dram[((wr_base >> 2) + wr_cnt) % DRAM_WORDS] <= m_wdata;
                wr_cnt <= wr_cnt + 1'b1;
                if (m_wlast) begin
                    wr_active <= 1'b0;
                    m_wready  <= 1'b0;
                    b_pending <= 1'b1;
                end
            end
            if (b_pending && !m_bvalid) begin
                m_bvalid  <= 1'b1;
                b_pending <= 1'b0;
            end else if (m_bvalid && m_bready) begin
                m_bvalid <= 1'b0;
            end
        end
    end

    function [31:0] pack4;
        input integer b0;
        input integer b1;
        input integer b2;
        input integer b3;
        begin
            pack4 = ((b0 & 32'hff) << 0)  |
                    ((b1 & 32'hff) << 8)  |
                    ((b2 & 32'hff) << 16) |
                    ((b3 & 32'hff) << 24);
        end
    endfunction

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
            s_awaddr = 32'd0; s_wdata = 32'd0; s_wstrb = 4'd0;
            s_awvalid = 1'b0; s_wvalid = 1'b0; s_bready = 1'b0;
            s_araddr = 32'd0; s_arvalid = 1'b0; s_rready = 1'b0;
            repeat (5) @(posedge clk);
            rst_n = 1'b1;
            repeat (3) @(posedge clk);
        end
    endtask

    task clear_dram;
        integer di;
        begin
            for (di = 0; di < DRAM_WORDS; di = di + 1)
                dram[di] = 32'd0;
        end
    endtask

    task load_vectors;
        input integer lanes;
        input integer a_base_value;
        integer li;
        begin
            for (li = 0; li < lanes; li = li + 4) begin
                dram[(W_ADDR >> 2) + (li >> 2)] =
                    pack4(li + 1, li + 2, li + 3, li + 4);
                dram[(A_ADDR >> 2) + (li >> 2)] =
                    pack4(a_base_value + li,
                          a_base_value + li + 1,
                          a_base_value + li + 2,
                          a_base_value + li + 3);
            end
        end
    endtask

    task axi_write;
        input [31:0] addr;
        input [31:0] data;
        integer guard;
        begin
            s_awaddr  <= addr;
            s_awvalid <= 1'b1;
            s_wdata   <= data;
            s_wstrb   <= 4'hF;
            s_wvalid  <= 1'b1;
            s_bready  <= 1'b1;
            @(posedge clk);
            while (!s_awready) @(posedge clk);
            @(posedge clk);
            s_awvalid <= 1'b0;
            s_wvalid  <= 1'b0;
            guard = 0;
            while (!s_bvalid && guard < 100) begin
                @(posedge clk);
                guard = guard + 1;
            end
            s_bready <= 1'b0;
            if (guard >= 100) begin
                $display("[FAIL] AXI-Lite write timeout at 0x%08h", addr);
                errors = errors + 1;
            end
        end
    endtask

    task check_w_lanes;
        input integer lanes;
        input [127:0] name;
        integer li;
        reg [15:0] got;
        reg [15:0] exp;
        begin
            for (li = 0; li < lanes; li = li + 1) begin
                got = u_npu.pe_w_in[li*DATA_W +: DATA_W];
                exp = s8_lane(li + 1);
                if (got !== exp) begin
                    $display("[FAIL] %0s W lane%0d got=0x%04h exp=0x%04h",
                             name, li, got, exp);
                    errors = errors + 1;
                end
            end
        end
    endtask

    task check_a_lane;
        input integer lane;
        input integer a_base_value;
        input [127:0] name;
        reg [15:0] got;
        reg [15:0] exp;
        begin
            got = u_npu.pe_a_in[lane*DATA_W +: DATA_W];
            exp = s8_lane(a_base_value + lane);
            if (got !== exp) begin
                $display("[FAIL] %0s A row%0d got=0x%04h exp=0x%04h",
                         name, lane, got, exp);
                errors = errors + 1;
            end
        end
    endtask

    task run_shape_case;
        input integer lanes;
        input [31:0] shape;
        input integer a_base_value;
        input [127:0] name;
        integer guard;
        integer feed_idx;
        reg saw_w;
        begin
            apply_reset();
            clear_dram();
            load_vectors(lanes, a_base_value);

            axi_write(REG_M_DIM, lanes);
            axi_write(REG_N_DIM, lanes);
            axi_write(REG_K_DIM, 32'd1);
            axi_write(REG_W_ADDR, W_ADDR);
            axi_write(REG_A_ADDR, A_ADDR);
            axi_write(REG_R_ADDR, R_ADDR);
            axi_write(REG_ARR_CFG, ARR_TILE);
            axi_write(REG_CFG_SHAPE, shape);
            axi_write(REG_CTRL, CTRL_INT8_OS);

            guard = 0;
            feed_idx = 0;
            saw_w = 1'b0;
            while (feed_idx < lanes && guard < 500) begin
                @(negedge clk);
                if (u_npu.tile_feed_step) begin
                    if (u_npu.tile_vec_fire) begin
                        check_w_lanes(lanes, name);
                        saw_w = 1'b1;
                    end
                    if (saw_w && feed_idx < lanes) begin
                        check_a_lane(feed_idx, a_base_value, name);
                        feed_idx = feed_idx + 1;
                    end
                end
                guard = guard + 1;
            end

            if (!saw_w) begin
                $display("[FAIL] %0s never observed tile_vec_fire", name);
                errors = errors + 1;
            end
            if (feed_idx != lanes) begin
                $display("[FAIL] %0s observed only %0d/%0d A lanes",
                         name, feed_idx, lanes);
                errors = errors + 1;
            end
            if (saw_w && feed_idx == lanes)
                $display("[PASS] %0s lane feed observed", name);
        end
    endtask

    initial begin
        errors = 0;
        clear_dram();

        run_shape_case(8, 32'd1, 17, "8x8");
        run_shape_case(16, 32'd2, 33, "16x16");

        if (errors == 0) begin
            $display("[PASS] tb_npu_tile_lane_feed");
        end else begin
            $display("[FAIL] tb_npu_tile_lane_feed errors=%0d", errors);
            $fatal;
        end
        $finish;
    end

    initial begin
        #(CLK_T * 200000);
        $display("[FAIL] tb_npu_tile_lane_feed global timeout");
        $finish;
    end
endmodule
