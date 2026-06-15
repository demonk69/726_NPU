# Vivado source file list for the synthesizable NPU RTL.
#
# Usage from a Vivado Tcl script:
#   source scripts/vivado_npu_filelist.tcl
#   npu_vivado_add_sources sources_1
#
# Or manually:
#   source scripts/vivado_npu_filelist.tcl
#   add_files -fileset sources_1 $npu_vivado_project_rtl_files
#   set_property top npu_pynq_wrapper [get_filesets sources_1]
#   update_compile_order -fileset sources_1
#
# Top-level notes:
#   - Use npu_pynq_wrapper as the RTL module top before wrapping it in a BD.
#   - If a block design wrapper is generated, the final Vivado top is usually
#     the generated <bd_name>_wrapper, not npu_pynq_wrapper.
#
# FP16 status:
#   - FP16 is controlled by the synthesizable parameter FP16_ENABLE.
#   - FP16_ENABLE=0 is the default resource-saving INT8-only build. pe_top does
#     not elaborate fp16_mul/fp32_add in this mode.
#   - FP16_ENABLE=1 restores the FP16 PE datapath. Use
#     $npu_vivado_fp16_project_rtl_files and set the top/BD parameter to 1.
#   - fp16_add.v is not referenced from npu_pynq_wrapper today; it is listed only
#     as optional extra RTL for compatibility with the old glob-based flow.

set npu_vivado_repo_root [file normalize [file join [file dirname [info script]] ..]]

# Exact INT8-only dependency closure for rtl/top/npu_pynq_wrapper.v.
set npu_vivado_int8_rtl_files [list \
    [file join $npu_vivado_repo_root rtl/common/fifo.v] \
    [file join $npu_vivado_repo_root rtl/common/op_counter.v] \
    [file join $npu_vivado_repo_root rtl/common/axi_monitor.v] \
    [file join $npu_vivado_repo_root rtl/pe/pe_top.v] \
    [file join $npu_vivado_repo_root rtl/buf/pingpong_buf.v] \
    [file join $npu_vivado_repo_root rtl/array/reconfig_pe_array.v] \
    [file join $npu_vivado_repo_root rtl/axi/npu_axi_lite.v] \
    [file join $npu_vivado_repo_root rtl/axi/npu_dma.v] \
    [file join $npu_vivado_repo_root rtl/ctrl/npu_ctrl.v] \
    [file join $npu_vivado_repo_root rtl/power/npu_power.v] \
    [file join $npu_vivado_repo_root rtl/top/npu_top.v] \
    [file join $npu_vivado_repo_root rtl/top/npu_pynq_wrapper.v] \
]

set npu_vivado_fp16_datapath_rtl_files [list \
    [file join $npu_vivado_repo_root rtl/pe/fp16_mul.v] \
    [file join $npu_vivado_repo_root rtl/pe/fp32_add.v] \
]

set npu_vivado_fp16_rtl_files \
    [concat $npu_vivado_int8_rtl_files $npu_vivado_fp16_datapath_rtl_files]

# Backward-compatible alias: required files now mean the default INT8-only build.
set npu_vivado_required_rtl_files $npu_vivado_int8_rtl_files

# Synthesizable RTL currently added by the existing directory-glob flow, but not
# referenced from npu_pynq_wrapper today. Adding these files is harmless; Vivado
# will keep them as unused modules unless another top instantiates them.
set npu_vivado_extra_rtl_files [list \
    [file join $npu_vivado_repo_root rtl/pe/fp16_add.v] \
    [file join $npu_vivado_repo_root rtl/buf/psum_out_buf.v] \
]

# Default project list for resource-saving Vivado builds.
set npu_vivado_project_rtl_files $npu_vivado_int8_rtl_files

# Full source list for an explicit FP16-enabled build.
set npu_vivado_fp16_project_rtl_files \
    [concat $npu_vivado_fp16_rtl_files $npu_vivado_extra_rtl_files]

# These files are for simulation/SoC testbenches and should not be added as
# board-level Vivado design sources for the npu_pynq_wrapper project.
set npu_vivado_excluded_sim_rtl_files [list \
    [file join $npu_vivado_repo_root rtl/soc/axi_lite_bridge.v] \
    [file join $npu_vivado_repo_root rtl/soc/dram_model.v] \
    [file join $npu_vivado_repo_root rtl/soc/soc_mem.v] \
    [file join $npu_vivado_repo_root rtl/soc/soc_top.v] \
    [file join $npu_vivado_repo_root sim/picorv32.v] \
]

proc npu_vivado_add_sources {{fileset sources_1} {fp16_enable 0} {include_extra 0}} {
    global npu_vivado_int8_rtl_files
    global npu_vivado_fp16_datapath_rtl_files
    global npu_vivado_extra_rtl_files

    set rtl_files $npu_vivado_int8_rtl_files
    if {$fp16_enable} {
        set rtl_files [concat $rtl_files $npu_vivado_fp16_datapath_rtl_files]
    }
    if {$include_extra} {
        set rtl_files [concat $rtl_files $npu_vivado_extra_rtl_files]
    }

    add_files -fileset $fileset $rtl_files
    set_property top npu_pynq_wrapper [get_filesets $fileset]
    update_compile_order -fileset $fileset
}
