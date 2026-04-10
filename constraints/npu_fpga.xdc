# =============================================================================
# FPGA Constraints for NPU (Neural Processing Unit)
# Project: NPU_prj
# Target: Xilinx 7-Series (Artix-7/Kintex-7) or Zynq
# Clock: 100 MHz (adjust based on target device)
# =============================================================================

# -----------------------------------------------------------------------------
# Clock Constraints
# -----------------------------------------------------------------------------
# Primary system clock (adjust period for target frequency)
# 100 MHz = 10.000 ns period
# 50 MHz = 20.000 ns period (safer for initial synthesis)
create_clock -name sys_clk -period 20.000 [get_ports sys_clk]

# Optional: Clock uncertainty for setup/hold analysis
set_clock_uncertainty -setup 0.200 [get_clocks sys_clk]
set_clock_uncertainty -hold 0.050 [get_clocks sys_clk]

# -----------------------------------------------------------------------------
# I/O Constraints
# -----------------------------------------------------------------------------
# System reset (active low)
set_property IOSTANDARD LVCMOS33 [get_ports sys_rst_n]
set_property PACKAGE_PIN <PIN> [get_ports sys_rst_n]  ;# Assign appropriate pin

# System clock
set_property IOSTANDARD LVCMOS33 [get_ports sys_clk]
set_property PACKAGE_PIN <PIN> [get_ports sys_clk]    ;# Assign appropriate pin

# -----------------------------------------------------------------------------
# AXI4-Lite Slave Interface (CPU Config Port)
# -----------------------------------------------------------------------------
# Address Write Channel
set_property IOSTANDARD LVCMOS33 [get_ports s_axi_awaddr[*]]
set_property IOSTANDARD LVCMOS33 [get_ports s_axi_awvalid]
set_property IOSTANDARD LVCMOS33 [get_ports s_axi_awready]

# Write Data Channel
set_property IOSTANDARD LVCMOS33 [get_ports s_axi_wdata[*]]
set_property IOSTANDARD LVCMOS33 [get_ports s_axi_wstrb[*]]
set_property IOSTANDARD LVCMOS33 [get_ports s_axi_wvalid]
set_property IOSTANDARD LVCMOS33 [get_ports s_axi_wready]

# Write Response Channel
set_property IOSTANDARD LVCMOS33 [get_ports s_axi_bresp[*]]
set_property IOSTANDARD LVCMOS33 [get_ports s_axi_bvalid]
set_property IOSTANDARD LVCMOS33 [get_ports s_axi_bready]

# Address Read Channel
set_property IOSTANDARD LVCMOS33 [get_ports s_axi_araddr[*]]
set_property IOSTANDARD LVCMOS33 [get_ports s_axi_arvalid]
set_property IOSTANDARD LVCMOS33 [get_ports s_axi_arready]

# Read Data Channel
set_property IOSTANDARD LVCMOS33 [get_ports s_axi_rdata[*]]
set_property IOSTANDARD LVCMOS33 [get_ports s_axi_rresp[*]]
set_property IOSTANDARD LVCMOS33 [get_ports s_axi_rvalid]
set_property IOSTANDARD LVCMOS33 [get_ports s_axi_rready]

# -----------------------------------------------------------------------------
# AXI4 Master Interface (DMA to DRAM)
# -----------------------------------------------------------------------------
# Note: These may be internal connections in SoC mode
# If connected to external DDR, adjust IOSTANDARD accordingly

# Write Address Channel
set_property IOSTANDARD LVCMOS33 [get_ports m_axi_awaddr[*]]
set_property IOSTANDARD LVCMOS33 [get_ports m_axi_awlen[*]]
set_property IOSTANDARD LVCMOS33 [get_ports m_axi_awsize[*]]
set_property IOSTANDARD LVCMOS33 [get_ports m_axi_awburst[*]]
set_property IOSTANDARD LVCMOS33 [get_ports m_axi_awvalid]
set_property IOSTANDARD LVCMOS33 [get_ports m_axi_awready]

# Write Data Channel
set_property IOSTANDARD LVCMOS33 [get_ports m_axi_wdata[*]]
set_property IOSTANDARD LVCMOS33 [get_ports m_axi_wstrb[*]]
set_property IOSTANDARD LVCMOS33 [get_ports m_axi_wlast]
set_property IOSTANDARD LVCMOS33 [get_ports m_axi_wvalid]
set_property IOSTANDARD LVCMOS33 [get_ports m_axi_wready]

# Write Response Channel
set_property IOSTANDARD LVCMOS33 [get_ports m_axi_bresp[*]]
set_property IOSTANDARD LVCMOS33 [get_ports m_axi_bvalid]
set_property IOSTANDARD LVCMOS33 [get_ports m_axi_bready]

# Read Address Channel
set_property IOSTANDARD LVCMOS33 [get_ports m_axi_araddr[*]]
set_property IOSTANDARD LVCMOS33 [get_ports m_axi_arlen[*]]
set_property IOSTANDARD LVCMOS33 [get_ports m_axi_arsize[*]]
set_property IOSTANDARD LVCMOS33 [get_ports m_axi_arburst[*]]
set_property IOSTANDARD LVCMOS33 [get_ports m_axi_arvalid]
set_property IOSTANDARD LVCMOS33 [get_ports m_axi_arready]

# Read Data Channel
set_property IOSTANDARD LVCMOS33 [get_ports m_axi_rdata[*]]
set_property IOSTANDARD LVCMOS33 [get_ports m_axi_rresp[*]]
set_property IOSTANDARD LVCMOS33 [get_ports m_axi_rlast]
set_property IOSTANDARD LVCMOS33 [get_ports m_axi_rvalid]
set_property IOSTANDARD LVCMOS33 [get_ports m_axi_rready]

# -----------------------------------------------------------------------------
# Input/Output Delay Constraints
# -----------------------------------------------------------------------------
# AXI4-Lite slave interface (assume 5ns input/output delay for CPU interface)
set_input_delay -clock sys_clk -max 5.000 [get_ports s_axi_awaddr[*]]
set_input_delay -clock sys_clk -max 5.000 [get_ports s_axi_awvalid]
set_input_delay -clock sys_clk -max 5.000 [get_ports s_axi_wdata[*]]
set_input_delay -clock sys_clk -max 5.000 [get_ports s_axi_wstrb[*]]
set_input_delay -clock sys_clk -max 5.000 [get_ports s_axi_wvalid]
set_input_delay -clock sys_clk -max 5.000 [get_ports s_axi_araddr[*]]
set_input_delay -clock sys_clk -max 5.000 [get_ports s_axi_arvalid]
set_input_delay -clock sys_clk -max 5.000 [get_ports s_axi_bready]
set_input_delay -clock sys_clk -max 5.000 [get_ports s_axi_rready]

set_output_delay -clock sys_clk -max 5.000 [get_ports s_axi_awready]
set_output_delay -clock sys_clk -max 5.000 [get_ports s_axi_wready]
set_output_delay -clock sys_clk -max 5.000 [get_ports s_axi_bresp[*]]
set_output_delay -clock sys_clk -max 5.000 [get_ports s_axi_bvalid]
set_output_delay -clock sys_clk -max 5.000 [get_ports s_axi_arready]
set_output_delay -clock sys_clk -max 5.000 [get_ports s_axi_rdata[*]]
set_output_delay -clock sys_clk -max 5.000 [get_ports s_axi_rresp[*]]
set_output_delay -clock sys_clk -max 5.000 [get_ports s_axi_rvalid]

# -----------------------------------------------------------------------------
# False Paths / Asynchronous Signals
# -----------------------------------------------------------------------------
# Reset synchronization (if using async reset sync)
set_false_path -from [get_ports sys_rst_n]

# -----------------------------------------------------------------------------
# Timing Exceptions
# -----------------------------------------------------------------------------
# Multi-cycle paths for control signals (if needed)
# Example: Control signals that change infrequently
# set_multicycle_path -setup 2 -from [get_cells <control_reg>] -to [get_cells <target_reg>]

# -----------------------------------------------------------------------------
# Area/Timing Optimization Directives
# -----------------------------------------------------------------------------
# Enable DSP inference for multipliers (FP16/INT8 MAC operations)
set_property USE_DSP48 AUTO [get_cells -hierarchical -filter {NAME =~ */*mul*}]

# Enable block RAM inference for ping-pong buffers
set_property RAM_STYLE block [get_cells -hierarchical -filter {NAME =~ */pingpong_buf*}]
set_property RAM_STYLE block [get_cells -hierarchical -filter {NAME =~ */fifo*}]

# PE array registers - use distributed RAM for small memories
set_property RAM_STYLE distributed [get_cells -hierarchical -filter {NAME =~ */pe_array* && NAME =~ */act_reg*}]

# -----------------------------------------------------------------------------
# Power Optimization
# -----------------------------------------------------------------------------
# Enable clock gating (if supported by synthesis tool)
# set_property POWER_OPT true [get_designs]

# -----------------------------------------------------------------------------
# Placement Constraints (Optional)
# -----------------------------------------------------------------------------
# Keep PE array logic together for better timing
# create_pblock pe_array_pblock
# add_cells_to_pblock pe_array_pblock [get_cells -hierarchical -filter {NAME =~ */pe_array*}]
# resize_pblock pe_array_pblock -add {SLICE_X0Y0:SLICE_X10Y10}

# =============================================================================
# End of Constraints
# =============================================================================
