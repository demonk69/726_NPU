`timescale 1ns/1ps

module tb_npu_top_router_lite_error_status;
    localparam PHY_ROWS = 4;
    localparam PHY_COLS = 4;
    localparam DATA_W   = 64;
    localparam ACC_W    = 32;
    localparam [31:0] ERR_PE_ARRAY_ROUTER = 32'h0000_0800;

    reg clk;
    reg rst_n;

    reg  [31:0] s_axi_awaddr;
    reg         s_axi_awvalid;
    wire        s_axi_awready;
    reg  [31:0] s_axi_wdata;
    reg  [3:0]  s_axi_wstrb;
    reg         s_axi_wvalid;
    wire        s_axi_wready;
    wire [1:0]  s_axi_bresp;
    wire        s_axi_bvalid;
    reg         s_axi_bready;
    reg  [31:0] s_axi_araddr;
    reg         s_axi_arvalid;
    wire        s_axi_arready;
    wire [31:0] s_axi_rdata;
    wire [1:0]  s_axi_rresp;
    wire        s_axi_rvalid;
    reg         s_axi_rready;

    wire [31:0] m_axi_awaddr;
    wire [7:0]  m_axi_awlen;
    wire [2:0]  m_axi_awsize;
    wire [1:0]  m_axi_awburst;
    wire        m_axi_awvalid;
    reg         m_axi_awready;
    wire [ACC_W-1:0] m_axi_wdata;
    wire [ACC_W/8-1:0] m_axi_wstrb;
    wire        m_axi_wlast;
    wire        m_axi_wvalid;
    reg         m_axi_wready;
    reg  [1:0]  m_axi_bresp;
    reg         m_axi_bvalid;
    wire        m_axi_bready;
    wire [31:0] m_axi_araddr;
    wire [7:0]  m_axi_arlen;
    wire [2:0]  m_axi_arsize;
    wire [1:0]  m_axi_arburst;
    wire        m_axi_arvalid;
    reg         m_axi_arready;
    reg  [ACC_W-1:0] m_axi_rdata;
    reg  [1:0]  m_axi_rresp;
    reg         m_axi_rvalid;
    wire        m_axi_rready;
    reg         m_axi_rlast;
    wire        npu_irq;

    integer errors;

    npu_top #(
        .PHY_ROWS(PHY_ROWS),
        .PHY_COLS(PHY_COLS),
        .DATA_W(DATA_W),
        .ACC_W(ACC_W),
        .PPB_DEPTH(16),
        .PPB_THRESH(4),
        .INT8_SIMD_LANES(8),
        .FP16_ENABLE(1),
        .USE_ROUTER_MESH(1)
    ) dut (
        .sys_clk(clk),
        .sys_rst_n(rst_n),
        .s_axi_awaddr(s_axi_awaddr),
        .s_axi_awvalid(s_axi_awvalid),
        .s_axi_awready(s_axi_awready),
        .s_axi_wdata(s_axi_wdata),
        .s_axi_wstrb(s_axi_wstrb),
        .s_axi_wvalid(s_axi_wvalid),
        .s_axi_wready(s_axi_wready),
        .s_axi_bresp(s_axi_bresp),
        .s_axi_bvalid(s_axi_bvalid),
        .s_axi_bready(s_axi_bready),
        .s_axi_araddr(s_axi_araddr),
        .s_axi_arvalid(s_axi_arvalid),
        .s_axi_arready(s_axi_arready),
        .s_axi_rdata(s_axi_rdata),
        .s_axi_rresp(s_axi_rresp),
        .s_axi_rvalid(s_axi_rvalid),
        .s_axi_rready(s_axi_rready),
        .m_axi_awaddr(m_axi_awaddr),
        .m_axi_awlen(m_axi_awlen),
        .m_axi_awsize(m_axi_awsize),
        .m_axi_awburst(m_axi_awburst),
        .m_axi_awvalid(m_axi_awvalid),
        .m_axi_awready(m_axi_awready),
        .m_axi_wdata(m_axi_wdata),
        .m_axi_wstrb(m_axi_wstrb),
        .m_axi_wlast(m_axi_wlast),
        .m_axi_wvalid(m_axi_wvalid),
        .m_axi_wready(m_axi_wready),
        .m_axi_bresp(m_axi_bresp),
        .m_axi_bvalid(m_axi_bvalid),
        .m_axi_bready(m_axi_bready),
        .m_axi_araddr(m_axi_araddr),
        .m_axi_arlen(m_axi_arlen),
        .m_axi_arsize(m_axi_arsize),
        .m_axi_arburst(m_axi_arburst),
        .m_axi_arvalid(m_axi_arvalid),
        .m_axi_arready(m_axi_arready),
        .m_axi_rdata(m_axi_rdata),
        .m_axi_rresp(m_axi_rresp),
        .m_axi_rvalid(m_axi_rvalid),
        .m_axi_rready(m_axi_rready),
        .m_axi_rlast(m_axi_rlast),
        .npu_irq(npu_irq)
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
            release dut.pe_en;
            release dut.pe_stat;
            release dut.pe_mode;
            release dut.ctrl_cfg_shape;
            release dut.ctrl_router_enable;
            release dut.pe_array_global_ce;
            rst_n = 1'b0;
            s_axi_awaddr = 32'd0;
            s_axi_awvalid = 1'b0;
            s_axi_wdata = 32'd0;
            s_axi_wstrb = 4'd0;
            s_axi_wvalid = 1'b0;
            s_axi_bready = 1'b1;
            s_axi_araddr = 32'd0;
            s_axi_arvalid = 1'b0;
            s_axi_rready = 1'b1;
            m_axi_awready = 1'b1;
            m_axi_wready = 1'b1;
            m_axi_bresp = 2'b00;
            m_axi_bvalid = 1'b0;
            m_axi_arready = 1'b1;
            m_axi_rdata = {ACC_W{1'b0}};
            m_axi_rresp = 2'b00;
            m_axi_rvalid = 1'b0;
            m_axi_rlast = 1'b0;
            repeat (5) tick();
            rst_n = 1'b1;
            repeat (2) tick();
        end
    endtask

    task expect_router_error;
        input [1:0] shape;
        input data_mode;
        input stat_mode;
        input [255:0] name;
        begin
            reset_dut();
            force dut.ctrl_cfg_shape = shape;
            force dut.pe_mode = data_mode;
            force dut.pe_stat = stat_mode;
            force dut.ctrl_router_enable = 1'b1;
            force dut.pe_en = 1'b1;
            force dut.pe_array_global_ce = 1'b1;
            #1;

            if ((dut.core_error_status & ERR_PE_ARRAY_ROUTER) == 32'd0) begin
                $display("[FAIL] %0s core_error_status=0x%08h", name, dut.core_error_status);
                errors = errors + 1;
            end

            tick();
            tick();
            if ((dut.err_status & ERR_PE_ARRAY_ROUTER) == 32'd0) begin
                $display("[FAIL] %0s err_status=0x%08h", name, dut.err_status);
                errors = errors + 1;
            end
        end
    endtask

    task expect_no_router_error_when_disabled;
        input [1:0] shape;
        input data_mode;
        input stat_mode;
        input [255:0] name;
        begin
            reset_dut();
            force dut.ctrl_cfg_shape = shape;
            force dut.pe_mode = data_mode;
            force dut.pe_stat = stat_mode;
            force dut.ctrl_router_enable = 1'b0;
            force dut.pe_en = 1'b1;
            force dut.pe_array_global_ce = 1'b1;
            #1;

            if ((dut.core_error_status & ERR_PE_ARRAY_ROUTER) != 32'd0) begin
                $display("[FAIL] %0s core_error_status=0x%08h", name, dut.core_error_status);
                errors = errors + 1;
            end

            tick();
            tick();
            if ((dut.err_status & ERR_PE_ARRAY_ROUTER) != 32'd0) begin
                $display("[FAIL] %0s err_status=0x%08h", name, dut.err_status);
                errors = errors + 1;
            end
        end
    endtask

    task expect_no_router_error_when_enabled;
        input [1:0] shape;
        input data_mode;
        input stat_mode;
        input [255:0] name;
        begin
            reset_dut();
            force dut.ctrl_cfg_shape = shape;
            force dut.pe_mode = data_mode;
            force dut.pe_stat = stat_mode;
            force dut.ctrl_router_enable = 1'b1;
            force dut.pe_en = 1'b1;
            force dut.pe_array_global_ce = 1'b1;
            #1;

            if ((dut.core_error_status & ERR_PE_ARRAY_ROUTER) != 32'd0) begin
                $display("[FAIL] %0s core_error_status=0x%08h", name, dut.core_error_status);
                errors = errors + 1;
            end

            tick();
            tick();
            if ((dut.err_status & ERR_PE_ARRAY_ROUTER) != 32'd0) begin
                $display("[FAIL] %0s err_status=0x%08h", name, dut.err_status);
                errors = errors + 1;
            end
        end
    endtask

    initial begin
        errors = 0;
        expect_no_router_error_when_disabled(2'b00, 1'b0, 1'b0, "WS_ROUTER_DISABLED_TOP");
        expect_no_router_error_when_disabled(2'b11, 1'b0, 1'b1, "8X32_ROUTER_DISABLED_TOP");
        expect_no_router_error_when_disabled(2'b00, 1'b1, 1'b1, "FP16_ROUTER_DISABLED_TOP");
        expect_router_error(2'b00, 1'b0, 1'b0, "WS_UNSUPPORTED_TOP");
        expect_router_error(2'b00, 1'b1, 1'b1, "FP16_UNSUPPORTED_TOP");
        expect_router_error(2'b00, 1'b1, 1'b0, "FP16_WS_UNSUPPORTED_TOP");

        if (errors == 0) begin
            $display("[PASS] tb_npu_top_router_lite_error_status");
        end else begin
            $display("[FAIL] tb_npu_top_router_lite_error_status errors=%0d", errors);
            $fatal;
        end
        $finish;
    end
endmodule
