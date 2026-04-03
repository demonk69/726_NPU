// =============================================================================
// Module  : soc_mem
// Project : NPU_prj
// Desc    : Simple single-port SRAM for PicoRV32 instruction + data storage.
//           Compatible with picosoc_mem interface.
//           Supports byte-lane write strobes (wen[3:0]).
//
//           Based on picosoc_mem from PicoRV32 project with enhancements.
// =============================================================================

`timescale 1ns/1ps

module soc_mem #(
    parameter WORDS  = 1024,    // number of 32-bit words (default 4KB)
    parameter INIT_HEX = ""      // optional hex file for initialization
)(
    input  wire              clk,
    input  wire [3:0]        wen,    // byte-lane write enable
    input  wire [21:0]       addr,   // word address (byte_addr >> 2)
    input  wire [31:0]       wdata,
    output reg  [31:0]       rdata
);

localparam ADDR_W = $clog2(WORDS);

reg [31:0] mem [0:WORDS-1];

// Optional initialization from hex file
`ifdef INIT_FILE
initial begin
    $readmemh(INIT_FILE, mem);
end
`endif

always @(posedge clk) begin
    rdata <= mem[addr[ADDR_W-1:0]];
    if (wen[0]) mem[addr[ADDR_W-1:0]][ 7: 0] <= wdata[ 7: 0];
    if (wen[1]) mem[addr[ADDR_W-1:0]][15: 8] <= wdata[15: 8];
    if (wen[2]) mem[addr[ADDR_W-1:0]][23:16] <= wdata[23:16];
    if (wen[3]) mem[addr[ADDR_W-1:0]][31:24] <= wdata[31:24];
end

endmodule
