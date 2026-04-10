@echo off
setlocal

set TB=D:\NPU_prj\tb\matmul\tb_matmul_os.v
set RTL=D:\NPU_prj\rtl

set RTL_FILES=%RTL%\pe\fp16_mul.v %RTL%\pe\fp16_add.v %RTL%\pe\fp32_add.v %RTL%\pe\pe_top.v %RTL%\common\fifo.v %RTL%\common\axi_monitor.v %RTL%\common\op_counter.v %RTL%\array\pe_array.v %RTL%\buf\pingpong_buf.v %RTL%\power\npu_power.v %RTL%\ctrl\npu_ctrl.v %RTL%\axi\npu_axi_lite.v %RTL%\axi\npu_dma.v %RTL%\top\npu_top.v

echo ===============================
echo OS INT8 2x3x2
echo ===============================
cd /d D:\NPU_prj\tb\matmul\os_int8_2x3x2
iverilog -o tb_matmul_os.vvp -I. %RTL_FILES% %TB% 2>nul
if errorlevel 1 (echo COMPILE FAILED & goto next1)
vvp tb_matmul_os.vvp 2>&1 | findstr /C:"PASSED" /C:"FAILED"
:next1

echo ===============================
echo OS INT8 3x4x3
echo ===============================
cd /d D:\NPU_prj\tb\matmul\os_int8_3x4x3
iverilog -o tb_matmul_os.vvp -I. %RTL_FILES% %TB% 2>nul
if errorlevel 1 (echo COMPILE FAILED & goto next2)
vvp tb_matmul_os.vvp 2>&1 | findstr /C:"PASSED" /C:"FAILED"
:next2

echo ===============================
echo OS INT8 2x4x3
echo ===============================
cd /d D:\NPU_prj\tb\matmul\os_int8_2x4x3
iverilog -o tb_matmul_os.vvp -I. %RTL_FILES% %TB% 2>nul
if errorlevel 1 (echo COMPILE FAILED & goto next3)
vvp tb_matmul_os.vvp 2>&1 | findstr /C:"PASSED" /C:"FAILED"
:next3

echo ===============================
echo OS FP16 2x3x2
echo ===============================
cd /d D:\NPU_prj\tb\matmul\os_fp16_2x3x2
iverilog -o tb_matmul_os.vvp -I. %RTL_FILES% %TB% 2>nul
if errorlevel 1 (echo COMPILE FAILED & goto next4)
vvp tb_matmul_os.vvp 2>&1 | findstr /C:"PASSED" /C:"FAILED"
:next4

echo ===============================
echo OS FP16 3x4x3
echo ===============================
cd /d D:\NPU_prj\tb\matmul\os_fp16_3x4x3
iverilog -o tb_matmul_os.vvp -I. %RTL_FILES% %TB% 2>nul
if errorlevel 1 (echo COMPILE FAILED & goto next5)
vvp tb_matmul_os.vvp 2>&1 | findstr /C:"PASSED" /C:"FAILED"
:next5

echo ===============================
echo ALL OS BASE TESTS DONE
echo ===============================
