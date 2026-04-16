// =============================================================================
// File    : tb_npu_top.v
// Project : NPU_prj
// Desc    : Sanity Check (Smoke Test) for npu_top
//
//           Test case: Toy 3x3 conv with 2x2 kernel → M=4, N=1, K=4 MatMul
//
//           Input Activation A (M=4, K=4) — 4 tiles, packed as 32-bit words:
//             Tile 0 @ A_ADDR+0x00 : 0x05040201  (a[0..3])
//             Tile 1 @ A_ADDR+0x04 : 0x06050302  (a[4..7])
//             Tile 2 @ A_ADDR+0x08 : 0x08070504  (a[8..11])
//             Tile 3 @ A_ADDR+0x0C : 0x09080605  (a[12..15])
//
//           Weight B (K=4, N=1) — 1 tile:
//             Tile 0 @ W_ADDR+0x00 : 0x04030201  (b[0..3])
//
//           Expected result C (M=4, N=1) — 4 × INT32:
//             R[0] = 1*1 + 2*2 + 4*3 + 5*4 = 1+4+12+20 = 37  (0x00000025)
//             R[1] = 2*1 + 3*2 + 5*3 + 6*4 = 2+6+15+24 = 47  (0x0000002F)
//             R[2] = 4*1 + 5*2 + 7*3 + 8*4 = 4+10+21+32= 67  (0x00000043)
//             R[3] = 5*1 + 6*2 + 8*3 + 9*4 = 5+12+24+36= 77  (0x0000004D)
//
//           CTRL register for INT8 OS mode:
//             bit[3:2] = 2'b00 (INT8), bit[5:4] = 2'b01 (OS), bit[0] = 1 (start)
//             → CTRL = 32'h00000011  (0b0001_0001)
//
// =============================================================================

`timescale 1ns / 1ps

module tb_npu_top;

// ---------------------------------------------------------------------------
// Parameters
// ---------------------------------------------------------------------------
parameter CLK_PERIOD  = 10;      // 10 ns → 100 MHz
parameter TIMEOUT_CYC = 100000;  // simulation watchdog (cycles)

// DRAM address allocation (must fit in 4KB dram_mem[0:1023])
parameter A_BASE_ADDR = 32'h0000_0100;   // Activation matrix base @ 0x100
parameter W_BASE_ADDR = 32'h0000_0200;   // Weight matrix base @ 0x200
parameter R_BASE_ADDR = 32'h0000_0300;   // Result base @ 0x300

// AXI4-Lite base address (NPU register space, offset 0)
// In our TB we drive s_axi_awaddr directly with register offsets
// Register offsets
parameter REG_CTRL    = 32'h00;
parameter REG_STATUS  = 32'h04;
parameter REG_INT_EN  = 32'h08;
parameter REG_INT_CLR = 32'h0C;
parameter REG_M_DIM   = 32'h10;
parameter REG_N_DIM   = 32'h14;
parameter REG_K_DIM   = 32'h18;
parameter REG_W_ADDR  = 32'h20;
parameter REG_A_ADDR  = 32'h24;
parameter REG_R_ADDR  = 32'h28;

// CTRL value: start=1, data_mode=INT8(00), stat_mode=OS(01)
//   bit0=1(start), bit1=0, bit[3:2]=00(INT8), bit[5:4]=01(OS)
//   = 0b00_01_00_01 = 0x11
parameter CTRL_START_INT8_OS = 32'h0000_0011;

// ---------------------------------------------------------------------------
// DUT signal declarations
// ---------------------------------------------------------------------------
reg         sys_clk;
reg         sys_rst_n;

// AXI4-Lite Slave (CPU → NPU)
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

// AXI4 Master (NPU DMA → DRAM)
wire [31:0] m_axi_awaddr;
wire [7:0]  m_axi_awlen;
wire [2:0]  m_axi_awsize;
wire [1:0]  m_axi_awburst;
wire        m_axi_awvalid;
reg         m_axi_awready;
wire [31:0] m_axi_wdata;
wire [3:0]  m_axi_wstrb;
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
reg  [31:0] m_axi_rdata;
reg  [1:0]  m_axi_rresp;
reg         m_axi_rvalid;
wire        m_axi_rready;
reg         m_axi_rlast;

// Interrupt
wire        npu_irq;

// ---------------------------------------------------------------------------
// Mock DRAM: reg array (word-addressed, 32-bit per entry)
// Address range: 0x0000_0000 ~ 0x0000_3FFF (1K words = 4KB)
// ---------------------------------------------------------------------------
reg [31:0] dram_mem [0:1023];

// Word address helper (byte addr → word index)
function [9:0] word_idx;
    input [31:0] byte_addr;
    begin
        word_idx = byte_addr[11:2];  // lower 12-bit → top 10 bits = word
    end
endfunction

// ---------------------------------------------------------------------------
// Clock generation
// ---------------------------------------------------------------------------
initial sys_clk = 0;
always #(CLK_PERIOD/2) sys_clk = ~sys_clk;

// ---------------------------------------------------------------------------
// DRAM initialization
// ---------------------------------------------------------------------------
integer i;
initial begin
    // Clear entire DRAM
    for (i = 0; i < 1024; i = i + 1)
        dram_mem[i] = 32'd0;

    // ---- Activation A: 4 words @ A_BASE_ADDR ----
    // Each 32-bit word packs 4 INT8 values: [31:24]=byte3, [23:16]=byte2, [15:8]=byte1, [7:0]=byte0
    // Tile 0: a[3]=0x05, a[2]=0x04, a[1]=0x02, a[0]=0x01 → 0x05040201
    dram_mem[word_idx(A_BASE_ADDR + 32'h00)] = 32'h0504_0201;
    // Tile 1: a[7]=0x06, a[6]=0x05, a[5]=0x03, a[4]=0x02 → 0x06050302
    dram_mem[word_idx(A_BASE_ADDR + 32'h04)] = 32'h0605_0302;
    // Tile 2: a[11]=0x08, a[10]=0x07, a[9]=0x05, a[8]=0x04 → 0x08070504
    dram_mem[word_idx(A_BASE_ADDR + 32'h08)] = 32'h0807_0504;
    // Tile 3: a[15]=0x09, a[14]=0x08, a[13]=0x06, a[12]=0x05 → 0x09080605
    dram_mem[word_idx(A_BASE_ADDR + 32'h0C)] = 32'h0908_0605;

    // ---- Weight B: 1 word @ W_BASE_ADDR ----
    // b[3]=0x04, b[2]=0x03, b[1]=0x02, b[0]=0x01 → 0x04030201
    dram_mem[word_idx(W_BASE_ADDR + 32'h00)] = 32'h0403_0201;

    $display("[TB] DRAM initialized.");
    $display("[TB]   A[0] = %08h", dram_mem[word_idx(A_BASE_ADDR)]);
    $display("[TB]   A[1] = %08h", dram_mem[word_idx(A_BASE_ADDR+4)]);
    $display("[TB]   A[2] = %08h", dram_mem[word_idx(A_BASE_ADDR+8)]);
    $display("[TB]   A[3] = %08h", dram_mem[word_idx(A_BASE_ADDR+12)]);
    $display("[TB]   W[0] = %08h", dram_mem[word_idx(W_BASE_ADDR)]);
end

// ---------------------------------------------------------------------------
// DUT instantiation
// ---------------------------------------------------------------------------
npu_top #(
    .PHY_ROWS   (16),
    .PHY_COLS   (16),
    .DATA_W     (16),
    .ACC_W      (32),
    .PPB_DEPTH  (64),
    .PPB_THRESH (16)
) dut (
    .sys_clk       (sys_clk),
    .sys_rst_n     (sys_rst_n),
    // AXI4-Lite Slave
    .s_axi_awaddr  (s_axi_awaddr),
    .s_axi_awvalid (s_axi_awvalid),
    .s_axi_awready (s_axi_awready),
    .s_axi_wdata   (s_axi_wdata),
    .s_axi_wstrb   (s_axi_wstrb),
    .s_axi_wvalid  (s_axi_wvalid),
    .s_axi_wready  (s_axi_wready),
    .s_axi_bresp   (s_axi_bresp),
    .s_axi_bvalid  (s_axi_bvalid),
    .s_axi_bready  (s_axi_bready),
    .s_axi_araddr  (s_axi_araddr),
    .s_axi_arvalid (s_axi_arvalid),
    .s_axi_arready (s_axi_arready),
    .s_axi_rdata   (s_axi_rdata),
    .s_axi_rresp   (s_axi_rresp),
    .s_axi_rvalid  (s_axi_rvalid),
    .s_axi_rready  (s_axi_rready),
    // AXI4 Master
    .m_axi_awaddr  (m_axi_awaddr),
    .m_axi_awlen   (m_axi_awlen),
    .m_axi_awsize  (m_axi_awsize),
    .m_axi_awburst (m_axi_awburst),
    .m_axi_awvalid (m_axi_awvalid),
    .m_axi_awready (m_axi_awready),
    .m_axi_wdata   (m_axi_wdata),
    .m_axi_wstrb   (m_axi_wstrb),
    .m_axi_wlast   (m_axi_wlast),
    .m_axi_wvalid  (m_axi_wvalid),
    .m_axi_wready  (m_axi_wready),
    .m_axi_bresp   (m_axi_bresp),
    .m_axi_bvalid  (m_axi_bvalid),
    .m_axi_bready  (m_axi_bready),
    .m_axi_araddr  (m_axi_araddr),
    .m_axi_arlen   (m_axi_arlen),
    .m_axi_arsize  (m_axi_arsize),
    .m_axi_arburst (m_axi_arburst),
    .m_axi_arvalid (m_axi_arvalid),
    .m_axi_arready (m_axi_arready),
    .m_axi_rdata   (m_axi_rdata),
    .m_axi_rresp   (m_axi_rresp),
    .m_axi_rvalid  (m_axi_rvalid),
    .m_axi_rready  (m_axi_rready),
    .m_axi_rlast   (m_axi_rlast),
    // Interrupt
    .npu_irq       (npu_irq)
);

// ---------------------------------------------------------------------------
// AXI-Lite default tie-offs (read channel not used in main flow)
// ---------------------------------------------------------------------------
initial begin
    s_axi_awaddr  = 0;
    s_axi_awvalid = 0;
    s_axi_wdata   = 0;
    s_axi_wstrb   = 4'hF;
    s_axi_wvalid  = 0;
    s_axi_bready  = 1;   // always accept write response
    s_axi_araddr  = 0;
    s_axi_arvalid = 0;
    s_axi_rready  = 1;
end

// ---------------------------------------------------------------------------
// Task: AXI4-Lite single write  (address + data phase)
//
// npu_axi_lite timing (per spec):
//   T0: awvalid=1, awready=1 → AW captured, aw_q<=1
//   T1: wvalid=1, wready=1 (aw_q=1), bvalid=1 → W captured, B issued
//   T2: bvalid=0 (since wvalid=0 now)
//
// So B is valid during the same cycle as W handshake. We accept B at T1.
// ---------------------------------------------------------------------------
task axi_lite_write;
    input [31:0] addr;
    input [31:0] data;
    begin
        // T0: Drive AW
        @(posedge sys_clk); #1;
        s_axi_awaddr  = addr;
        s_axi_awvalid = 1'b1;
        
        // Wait for AW handshake
        wait (s_axi_awready);
        @(posedge sys_clk); #1;
        s_axi_awvalid = 1'b0;
        
        // T1: Drive W, accept B in same cycle
        s_axi_wdata   = data;
        s_axi_wstrb   = 4'hF;
        s_axi_wvalid  = 1'b1;
        
        // Wait for W handshake (bvalid is also high this cycle)
        wait (s_axi_wready && s_axi_bvalid);
        @(posedge sys_clk); #1;
        s_axi_wvalid  = 1'b0;
        // B is accepted automatically since bready is always 1
        
        $display("[CFG] Write addr=0x%08h data=0x%08h  (time=%0t)", addr, data, $time);
    end
endtask

// ---------------------------------------------------------------------------
// Mock AXI4 Slave — Read channel (AR/R): serve NPU DMA burst reads
//
// Protocol: simple round-trip
//   1. Accept AR when arvalid && arready
//   2. Return (arlen+1) R beats with data from dram_mem[]
//      rlast on the final beat
// ---------------------------------------------------------------------------
// AR capture registers
reg [31:0] ar_addr_cap;
reg [7:0]  ar_len_cap;
reg        ar_pending;   // AR has been captured, waiting to issue R beats

// AXI4 Read state machine
localparam AR_IDLE  = 2'd0;
localparam AR_ACCEPT= 2'd1;
localparam AR_DATA  = 2'd2;

reg [1:0]  ar_state;
reg [7:0]  ar_beat_cnt;
reg [31:0] ar_cur_addr;

always @(posedge sys_clk or negedge sys_rst_n) begin
    if (!sys_rst_n) begin
        ar_state    <= AR_IDLE;
        m_axi_arready <= 1'b0;
        m_axi_rvalid  <= 1'b0;
        m_axi_rdata   <= 32'd0;
        m_axi_rlast   <= 1'b0;
        m_axi_rresp   <= 2'b00;
        ar_beat_cnt   <= 8'd0;
        ar_cur_addr   <= 32'd0;
    end else begin
        case (ar_state)
            AR_IDLE: begin
                m_axi_arready <= 1'b1;   // always ready to accept AR
                m_axi_rvalid  <= 1'b0;
                m_axi_rlast   <= 1'b0;
                if (m_axi_arvalid && m_axi_arready) begin
                    // Latch address and burst length
                    ar_cur_addr   <= m_axi_araddr;
                    ar_beat_cnt   <= m_axi_arlen;   // 0-based: (arlen+1) beats
                    ar_state      <= AR_DATA;
                    m_axi_arready <= 1'b0;
                    $display("[DRAM-R] AR: addr=0x%08h len=%0d  (time=%0t)",
                             m_axi_araddr, m_axi_arlen, $time);
                end
            end

            AR_DATA: begin
                // Present read data
                m_axi_rvalid <= 1'b1;
                m_axi_rdata  <= dram_mem[ar_cur_addr[11:2]];
                m_axi_rresp  <= 2'b00;
                m_axi_rlast  <= (ar_beat_cnt == 8'd0);

                if (m_axi_rvalid && m_axi_rready) begin
                    $display("[DRAM-R]   beat: addr=0x%08h data=0x%08h last=%b",
                             ar_cur_addr, dram_mem[ar_cur_addr[11:2]], m_axi_rlast);
                    if (ar_beat_cnt == 8'd0) begin
                        // Last beat done
                        m_axi_rvalid  <= 1'b0;
                        m_axi_rlast   <= 1'b0;
                        ar_state      <= AR_IDLE;
                    end else begin
                        ar_beat_cnt <= ar_beat_cnt - 8'd1;
                        ar_cur_addr <= ar_cur_addr + 32'd4;
                    end
                end
            end

            default: ar_state <= AR_IDLE;
        endcase
    end
end

// ---------------------------------------------------------------------------
// Mock AXI4 Slave — Write channel (AW/W/B): receive NPU DMA write-back
//
// Protocol:
//   1. Accept AW → latch addr + len
//   2. Accept W beats → write into dram_mem[]
//   3. Issue B response
// ---------------------------------------------------------------------------
localparam AW_IDLE   = 3'd0;
localparam AW_ACCEPT = 3'd1;
localparam AW_DATA   = 3'd2;
localparam AW_RESP   = 3'd3;

reg [2:0]  aw_state;
reg [31:0] aw_addr_cap;
reg [7:0]  aw_len_cap;
reg [7:0]  aw_beat_cnt;
reg [31:0] aw_cur_addr;

always @(posedge sys_clk or negedge sys_rst_n) begin
    if (!sys_rst_n) begin
        aw_state      <= AW_IDLE;
        m_axi_awready <= 1'b0;
        m_axi_wready  <= 1'b0;
        m_axi_bvalid  <= 1'b0;
        m_axi_bresp   <= 2'b00;
        aw_beat_cnt   <= 8'd0;
        aw_cur_addr   <= 32'd0;
    end else begin
        case (aw_state)
            AW_IDLE: begin
                m_axi_awready <= 1'b1;
                m_axi_bvalid  <= 1'b0;
                m_axi_wready  <= 1'b0;
                if (m_axi_awvalid && m_axi_awready) begin
                    aw_cur_addr   <= m_axi_awaddr;
                    aw_beat_cnt   <= m_axi_awlen;
                    m_axi_awready <= 1'b0;
                    aw_state      <= AW_DATA;
                    $display("[DRAM-W] AW: addr=0x%08h len=%0d  (time=%0t)",
                             m_axi_awaddr, m_axi_awlen, $time);
                end
            end

            AW_DATA: begin
                m_axi_wready <= 1'b1;
                if (m_axi_wvalid && m_axi_wready) begin
                    // Write data into DRAM model
                    dram_mem[aw_cur_addr[11:2]] <= m_axi_wdata;
                    $display("[DRAM-W]   beat: addr=0x%08h data=0x%08h last=%b",
                             aw_cur_addr, m_axi_wdata, m_axi_wlast);
                    if (m_axi_wlast) begin
                        m_axi_wready <= 1'b0;
                        aw_state     <= AW_RESP;
                    end else begin
                        aw_cur_addr <= aw_cur_addr + 32'd4;
                        aw_beat_cnt <= aw_beat_cnt - 8'd1;
                    end
                end
            end

            AW_RESP: begin
                m_axi_bvalid <= 1'b1;
                m_axi_bresp  <= 2'b00;
                if (m_axi_bvalid && m_axi_bready) begin
                    m_axi_bvalid <= 1'b0;
                    aw_state     <= AW_IDLE;
                end
            end

            default: aw_state <= AW_IDLE;
        endcase
    end
end

// ---------------------------------------------------------------------------
// Watchdog timer
// ---------------------------------------------------------------------------
integer watchdog_cnt;
initial begin
    watchdog_cnt = 0;
end
always @(posedge sys_clk) begin
    if (sys_rst_n) begin
        watchdog_cnt = watchdog_cnt + 1;
        if (watchdog_cnt >= TIMEOUT_CYC) begin
            $display("[ERROR] TIMEOUT: simulation exceeded %0d cycles. NPU did not raise IRQ.",
                     TIMEOUT_CYC);
            $finish;
        end
    end
end

// ---------------------------------------------------------------------------
// Main test sequence
// ---------------------------------------------------------------------------
integer err_cnt;
reg [31:0] result [0:3];

initial begin
    // -----------------------------------------------------------------------
    // Step 1: System Reset
    // -----------------------------------------------------------------------
    err_cnt   = 0;
    sys_rst_n = 1'b0;
    repeat (10) @(posedge sys_clk);
    sys_rst_n = 1'b1;
    repeat (5)  @(posedge sys_clk);
    $display("[TB] Reset deasserted at time=%0t", $time);

    // -----------------------------------------------------------------------
    // Step 2: Configure dimension registers
    // -----------------------------------------------------------------------
    axi_lite_write(REG_M_DIM, 32'd4);   // M = 4
    axi_lite_write(REG_N_DIM, 32'd1);   // N = 1
    axi_lite_write(REG_K_DIM, 32'd4);   // K = 4

    // -----------------------------------------------------------------------
    // Step 3: Configure DRAM addresses
    // -----------------------------------------------------------------------
    axi_lite_write(REG_W_ADDR, W_BASE_ADDR);   // Weight base
    axi_lite_write(REG_A_ADDR, A_BASE_ADDR);   // Activation base
    axi_lite_write(REG_R_ADDR, R_BASE_ADDR);   // Result base

    // -----------------------------------------------------------------------
    // Step 3.5: Enable interrupt (INT_EN bit0 = 1)
    // -----------------------------------------------------------------------
    axi_lite_write(REG_INT_EN, 32'h0000_0001);

    // -----------------------------------------------------------------------
    // Step 4: Trigger start
    //   CTRL = 0x11:
    //     bit[0]   = 1  → start
    //     bit[3:2] = 00 → INT8 data mode
    //     bit[5:4] = 01 → OS stationary mode
    // -----------------------------------------------------------------------
    $display("[TB] Triggering NPU start: CTRL=0x%08h", CTRL_START_INT8_OS);
    axi_lite_write(REG_CTRL, CTRL_START_INT8_OS);
    
    // Small delay to ensure config is latched
    @(posedge sys_clk); #1;

    // -----------------------------------------------------------------------
    // Step 5: Wait for IRQ
    // -----------------------------------------------------------------------
    $display("[TB] Waiting for npu_irq ...");
    wait (npu_irq == 1'b1);
    $display("[TB] IRQ received at time=%0t", $time);

    // Allow any in-flight AXI write to complete
    repeat (20) @(posedge sys_clk);

    // -----------------------------------------------------------------------
    // Step 6: Read results from DRAM model
    // -----------------------------------------------------------------------
    result[0] = dram_mem[word_idx(R_BASE_ADDR + 32'h00)];
    result[1] = dram_mem[word_idx(R_BASE_ADDR + 32'h04)];
    result[2] = dram_mem[word_idx(R_BASE_ADDR + 32'h08)];
    result[3] = dram_mem[word_idx(R_BASE_ADDR + 32'h0C)];

    $display("[TB] Result readback:");
    $display("[TB]   R[0] = 0x%08h (%0d)", result[0], result[0]);
    $display("[TB]   R[1] = 0x%08h (%0d)", result[1], result[1]);
    $display("[TB]   R[2] = 0x%08h (%0d)", result[2], result[2]);
    $display("[TB]   R[3] = 0x%08h (%0d)", result[3], result[3]);

    // -----------------------------------------------------------------------
    // Step 7: Self-checking
    //   Golden: R[0]=37, R[1]=47, R[2]=67, R[3]=77
    // -----------------------------------------------------------------------
    if (result[0] !== 32'd37) begin
        $display("[ERROR] R[0] mismatch: got 0x%08h (%0d), expected 37 (0x00000025)",
                 result[0], result[0]);
        err_cnt = err_cnt + 1;
    end
    if (result[1] !== 32'd47) begin
        $display("[ERROR] R[1] mismatch: got 0x%08h (%0d), expected 47 (0x0000002F)",
                 result[1], result[1]);
        err_cnt = err_cnt + 1;
    end
    if (result[2] !== 32'd67) begin
        $display("[ERROR] R[2] mismatch: got 0x%08h (%0d), expected 67 (0x00000043)",
                 result[2], result[2]);
        err_cnt = err_cnt + 1;
    end
    if (result[3] !== 32'd77) begin
        $display("[ERROR] R[3] mismatch: got 0x%08h (%0d), expected 77 (0x0000004D)",
                 result[3], result[3]);
        err_cnt = err_cnt + 1;
    end

    // -----------------------------------------------------------------------
    // Final verdict
    // -----------------------------------------------------------------------
    if (err_cnt == 0)
        $display("[SUCCESS] NPU Sanity Check Passed! All 4 results match golden.");
    else
        $display("[FAIL] NPU Sanity Check FAILED: %0d error(s).", err_cnt);

    // -----------------------------------------------------------------------
    // Step 8: Clear interrupt (Path A via INT_CLR)
    // -----------------------------------------------------------------------
    axi_lite_write(REG_INT_CLR, 32'h0000_0001);
    $display("[TB] IRQ cleared.");

    repeat (10) @(posedge sys_clk);
    $finish;
end

// ---------------------------------------------------------------------------
// Optional: waveform dump
// ---------------------------------------------------------------------------
initial begin
    if ($test$plusargs("DUMP")) begin
        $dumpfile("tb_npu_top.vcd");
        $dumpvars(0, tb_npu_top);
        $display("[TB] VCD dump enabled.");
    end
end

// ---------------------------------------------------------------------------
// AXI bus monitor — catch X/Z propagation on key signals
// ---------------------------------------------------------------------------
always @(posedge sys_clk) begin
    if (sys_rst_n) begin
        if (m_axi_arvalid && m_axi_arready) begin
            if (^m_axi_araddr === 1'bx)
                $display("[WARN] X detected on m_axi_araddr at time=%0t", $time);
        end
        if (m_axi_awvalid && m_axi_awready) begin
            if (^m_axi_awaddr === 1'bx)
                $display("[WARN] X detected on m_axi_awaddr at time=%0t", $time);
        end
        if (m_axi_wvalid && m_axi_wready) begin
            if (^m_axi_wdata === 1'bx)
                $display("[WARN] X detected on m_axi_wdata at time=%0t", $time);
        end
    end
end

// (Optional) Debug monitors - uncomment for detailed tracing
// always @(posedge sys_clk) begin
//     if (sys_rst_n && (s_axi_awvalid || s_axi_wvalid || s_axi_bvalid)) begin
//         $display("[AXI-LITE] time=%0t awvalid=%b awready=%b wvalid=%b wready=%b bvalid=%b",
//                  $time, s_axi_awvalid, s_axi_awready, s_axi_wvalid, s_axi_wready, s_axi_bvalid);
//     end
// end

endmodule
// =============================================================================
// End of tb_npu_top.v
// =============================================================================
