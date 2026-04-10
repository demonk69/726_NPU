#!/usr/bin/env pwsh
# Debug WS K=1 test - verify PE pipeline delay

cd d:\NPU_prj

# Create a minimal testbench for WS K=1
$testbench = @'
module tb_debug_ws_k1;
    reg clk = 0;
    reg rst_n = 0;
    
    // PE signals
    reg en, load_w, mode, stat_mode, flush;
    reg [15:0] w_in, a_in;
    reg [31:0] acc_in;
    wire [31:0] acc_out;
    wire valid_out;
    
    // DUT
    pe_top #(
        .DATA_W(16),
        .ACC_W(32)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .en(en),
        .load_w(load_w),
        .mode(mode),
        .stat_mode(stat_mode),
        .flush(flush),
        .w_in(w_in),
        .a_in(a_in),
        .acc_in(acc_in),
        .acc_out(acc_out),
        .valid_out(valid_out)
    );
    
    always #5 clk = ~clk;
    
    initial begin
        $display("=== WS K=1 Pipeline Debug ===");
        $display("Time | en | load_w | w_in   | a_in   | valid_out | acc_out");
        $display("-----------------------------------------------------------");
        $monitor("%0t | %b  | %b      | 0x%04X | 0x%04X | %b         | 0x%08X", 
                 $time, en, load_w, w_in, a_in, valid_out, acc_out);
        
        // Reset
        rst_n = 0; en = 0; load_w = 0; mode = 1; stat_mode = 0; flush = 0;
        w_in = 16'h3E00; a_in = 16'h4000; acc_in = 32'h00000000;
        repeat(3) @(posedge clk);
        rst_n = 1;
        repeat(2) @(posedge clk);
        
        // WS K=1: one cycle with en=1, load_w=1
        en = 1; load_w = 1;
        @(posedge clk);
        
        // Followed by cycles with en=1, load_w=0
        load_w = 0;
        repeat(5) @(posedge clk);
        
        en = 0;
        repeat(3) @(posedge clk);
        
        $display("=== Done ===");
        $finish;
    end
endmodule
'@

$testbench | Out-File -Encoding ASCII tb\tb_debug_ws_k1.v

# Compile and run
Write-Host "Compiling..."
icarus\iverilog.exe -g2012 -o sim\tb_debug_ws_k1.vvp tb\tb_debug_ws_k1.v rtl\pe\pe_top.v rtl\pe\pe_fp16_mul.v rtl\fp16\fp16_mul.v rtl\fp16\fp16_add.v 2>&1 | Out-Null

if ($LASTEXITCODE -eq 0) {
    Write-Host "Running simulation..."
    iverilog\vvp.exe sim\tb_debug_ws_k1.vvp 2>&1
} else {
    Write-Host "Compilation failed"
}
