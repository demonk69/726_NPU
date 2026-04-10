@echo off
setlocal

set RTL=D:\NPU_prj\rtl
set TB=D:\NPU_prj\tb

cd /d D:\NPU_prj\tb
iverilog -o tb_pe_top.vvp %RTL%\pe\fp16_mul.v %RTL%\pe\fp16_add.v %RTL%\pe\fp32_add.v %RTL%\pe\pe_top.v tb_pe_top.v 2>nul
vvp tb_pe_top.vvp 2>&1
