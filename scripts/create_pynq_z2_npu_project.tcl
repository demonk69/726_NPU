# Create a PYNQ-Z2 Vivado project and a first-pass NPU block design.
#
# Usage:
#   vivado -mode batch -source scripts/create_pynq_z2_npu_project.tcl
#   vivado -mode batch -source scripts/create_pynq_z2_npu_project.tcl -tclargs --build-bitstream
#   vivado -mode batch -source scripts/create_pynq_z2_npu_project.tcl -tclargs --project-dir build/vivado/pynq_z2_npu
#
# This script intentionally creates a minimal PS + PL design:
#   Zynq PS M_AXI_GP0 -> NPU AXI-Lite register interface
#   NPU AXI master    -> Zynq PS S_AXI_HP0 DDR port
#   FCLK_CLK0         -> NPU and AXI clocks
#   proc_sys_reset    -> active-low peripheral resets

if {![llength [info commands create_project]]} {
    puts "This script must be run from Vivado Tcl. Syntax-only load complete."
    exit 0
}

proc usage {} {
    puts "Usage: vivado -mode batch -source scripts/create_pynq_z2_npu_project.tcl -tclargs ?options?"
    puts ""
    puts "Options:"
    puts "  --origin-dir <path>       Repository root. Default: parent of this script."
    puts "  --project-dir <path>      Vivado project directory. Default: build/vivado/pynq_z2_npu."
    puts "  --project-name <name>     Vivado project name. Default: npu_pynq_z2."
    puts "  --bd-name <name>          Block design name. Default: system."
    puts "  --board-part <name>       PYNQ-Z2 board part. Default: auto-detect *pynq-z2*."
    puts "  --jobs <n>                Parallel Vivado jobs for build. Default: 4."
    puts "  --build-bitstream         Run synthesis, implementation, and bitstream generation."
    puts "  --help                    Show this help."
}

proc parse_args {argv} {
    array set opts {
        origin_dir ""
        project_dir ""
        project_name npu_pynq_z2
        bd_name system
        board_part ""
        jobs 4
        build_bitstream 0
    }

    set i 0
    while {$i < [llength $argv]} {
        set arg [lindex $argv $i]
        switch -- $arg {
            --origin-dir {
                incr i
                set opts(origin_dir) [lindex $argv $i]
            }
            --project-dir {
                incr i
                set opts(project_dir) [lindex $argv $i]
            }
            --project-name {
                incr i
                set opts(project_name) [lindex $argv $i]
            }
            --bd-name {
                incr i
                set opts(bd_name) [lindex $argv $i]
            }
            --board-part {
                incr i
                set opts(board_part) [lindex $argv $i]
            }
            --jobs {
                incr i
                set opts(jobs) [lindex $argv $i]
            }
            --build-bitstream {
                set opts(build_bitstream) 1
            }
            --help - -h {
                usage
                exit 0
            }
            default {
                error "Unknown option: $arg"
            }
        }
        incr i
    }
    return [array get opts]
}

proc first_ipdef {pattern} {
    set defs [get_ipdefs -all -quiet $pattern]
    if {[llength $defs] == 0} {
        error "Could not find Vivado IP definition matching $pattern"
    }
    return [lindex $defs end]
}

proc bd_intf_pin_any {cell names} {
    foreach name $names {
        set pins [get_bd_intf_pins -quiet "$cell/$name"]
        if {[llength $pins] > 0} {
            return [lindex $pins 0]
        }
    }
    error "Could not find any interface pin on $cell matching: $names. Check Vivado interface inference for npu_pynq_wrapper."
}

proc bd_pin_if_exists {pin_name} {
    set pins [get_bd_pins -quiet $pin_name]
    if {[llength $pins] > 0} {
        return [lindex $pins 0]
    }
    return ""
}

proc connect_if_pin_exists {net_name pin_name} {
    set pin [bd_pin_if_exists $pin_name]
    if {$pin ne ""} {
        connect_bd_net $net_name $pin
    }
}

proc add_npu_rtl {origin_dir} {
    set rtl_files [list]
    foreach rel_dir {rtl/pe rtl/common rtl/buf rtl/array rtl/axi rtl/ctrl rtl/power rtl/top} {
        foreach file_name [lsort [glob -nocomplain [file join $origin_dir $rel_dir *.v]]] {
            lappend rtl_files $file_name
        }
    }

    if {[llength $rtl_files] == 0} {
        error "No RTL files found under $origin_dir/rtl"
    }

    add_files -fileset sources_1 $rtl_files
    set_property top npu_pynq_wrapper [get_filesets sources_1]
    update_compile_order -fileset sources_1
}

proc set_pynq_z2_board_part {requested_board_part} {
    if {$requested_board_part ne ""} {
        set board_parts [get_board_parts -quiet $requested_board_part]
    } else {
        set board_parts [get_board_parts -quiet *pynq-z2*]
        if {[llength $board_parts] == 0} {
            set board_parts [get_board_parts -quiet *PYNQ*Z2*]
        }
        if {[llength $board_parts] == 0} {
            set board_parts [get_board_parts -quiet *pynq*z2*]
        }
    }

    if {[llength $board_parts] == 0} {
        error "PYNQ-Z2 board part not found. Install PYNQ-Z2 board files or pass --board-part <name>."
    }

    set board_part [lindex $board_parts 0]
    set_property board_part $board_part [current_project]
    puts "Using board_part: $board_part"
}

proc create_npu_bd {bd_name} {
    create_bd_design $bd_name
    current_bd_design $bd_name

    set ps7 [create_bd_cell -type ip -vlnv [first_ipdef xilinx.com:ip:processing_system7:*] processing_system7_0]
    set rst [create_bd_cell -type ip -vlnv [first_ipdef xilinx.com:ip:proc_sys_reset:*] proc_sys_reset_0]
    set axi_lite [create_bd_cell -type ip -vlnv [first_ipdef xilinx.com:ip:axi_interconnect:*] axi_lite_interconnect]
    set axi_ddr [create_bd_cell -type ip -vlnv [first_ipdef xilinx.com:ip:axi_interconnect:*] axi_ddr_interconnect]
    set npu [create_bd_cell -type module -reference npu_pynq_wrapper npu_0]

    set_property -dict [list \
        CONFIG.PCW_USE_M_AXI_GP0 {1} \
        CONFIG.PCW_USE_S_AXI_HP0 {1} \
        CONFIG.PCW_EN_CLK0_PORT {1} \
        CONFIG.PCW_EN_RST0_PORT {1} \
        CONFIG.PCW_FPGA0_PERIPHERAL_FREQMHZ {100.000000} \
    ] $ps7

    set_property -dict [list CONFIG.C_EXT_RESET_HIGH {0}] $rst
    set_property -dict [list CONFIG.NUM_SI {1} CONFIG.NUM_MI {1}] $axi_lite
    set_property -dict [list CONFIG.NUM_SI {1} CONFIG.NUM_MI {1}] $axi_ddr
    set_property -dict [list \
        CONFIG.PHY_ROWS {16} \
        CONFIG.PHY_COLS {16} \
        CONFIG.DATA_W {32} \
        CONFIG.ACC_W {32} \
        CONFIG.PPB_DEPTH {64} \
        CONFIG.PPB_THRESH {16} \
        CONFIG.INT8_SIMD_LANES {4} \
        CONFIG.PERF_ENABLE_DERIVED {0} \
        CONFIG.S_AXI_OFFSET_BITS {16} \
        CONFIG.M_AXI_ID_WIDTH {1} \
    ] $npu

    apply_bd_automation -rule xilinx.com:bd_rule:processing_system7 \
        -config {make_external "FIXED_IO, DDR" apply_board_preset "1" Master "Disable" Slave "Disable"} \
        $ps7

    set_property -dict [list \
        CONFIG.PCW_USE_M_AXI_GP0 {1} \
        CONFIG.PCW_USE_S_AXI_HP0 {1} \
        CONFIG.PCW_EN_CLK0_PORT {1} \
        CONFIG.PCW_EN_RST0_PORT {1} \
        CONFIG.PCW_FPGA0_PERIPHERAL_FREQMHZ {100.000000} \
    ] $ps7

    set fclk [get_bd_pins processing_system7_0/FCLK_CLK0]
    set fresetn [get_bd_pins processing_system7_0/FCLK_RESET0_N]
    set periph_aresetn [get_bd_pins proc_sys_reset_0/peripheral_aresetn]

    connect_bd_net $fclk [get_bd_pins proc_sys_reset_0/slowest_sync_clk]
    connect_bd_net $fresetn [get_bd_pins proc_sys_reset_0/ext_reset_in]
    connect_bd_net $fclk [get_bd_pins npu_0/aclk]
    connect_bd_net $periph_aresetn [get_bd_pins npu_0/aresetn]
    connect_bd_net $fclk [get_bd_pins processing_system7_0/M_AXI_GP0_ACLK]
    connect_bd_net $fclk [get_bd_pins processing_system7_0/S_AXI_HP0_ACLK]

    foreach cell {axi_lite_interconnect axi_ddr_interconnect} {
        connect_if_pin_exists $fclk "$cell/ACLK"
        connect_if_pin_exists $fclk "$cell/S00_AXI_ACLK"
        connect_if_pin_exists $fclk "$cell/M00_AXI_ACLK"
        connect_if_pin_exists $periph_aresetn "$cell/ARESETN"
        connect_if_pin_exists $periph_aresetn "$cell/S00_AXI_ARESETN"
        connect_if_pin_exists $periph_aresetn "$cell/M00_AXI_ARESETN"
    }

    set npu_s_axi [bd_intf_pin_any npu_0 {S_AXI s_axi}]
    set npu_m_axi [bd_intf_pin_any npu_0 {M_AXI m_axi}]

    connect_bd_intf_net [get_bd_intf_pins processing_system7_0/M_AXI_GP0] [get_bd_intf_pins axi_lite_interconnect/S00_AXI]
    connect_bd_intf_net [get_bd_intf_pins axi_lite_interconnect/M00_AXI] $npu_s_axi
    connect_bd_intf_net $npu_m_axi [get_bd_intf_pins axi_ddr_interconnect/S00_AXI]
    connect_bd_intf_net [get_bd_intf_pins axi_ddr_interconnect/M00_AXI] [get_bd_intf_pins processing_system7_0/S_AXI_HP0]

    assign_bd_address
    foreach seg [get_bd_addr_segs -quiet -filter {NAME =~ "*npu_0*"}] {
        catch {set_property range 64K $seg}
    }

    validate_bd_design
    save_bd_design

    set bd_files [get_files -quiet ${bd_name}.bd]
    if {[llength $bd_files] == 0} {
        error "Could not locate generated BD file for $bd_name"
    }

    generate_target all [lindex $bd_files 0]
    set wrapper_files [make_wrapper -files [lindex $bd_files 0] -top]
    add_files -norecurse $wrapper_files
    set_property top ${bd_name}_wrapper [get_filesets sources_1]
    update_compile_order -fileset sources_1
}

array set opts [parse_args $argv]

if {$opts(origin_dir) eq ""} {
    set opts(origin_dir) [file normalize [file join [file dirname [info script]] ..]]
} else {
    set opts(origin_dir) [file normalize $opts(origin_dir)]
}

if {$opts(project_dir) eq ""} {
    set opts(project_dir) [file join $opts(origin_dir) build vivado pynq_z2_npu]
} elseif {[file pathtype $opts(project_dir)] ne "absolute"} {
    set opts(project_dir) [file join $opts(origin_dir) $opts(project_dir)]
}
set opts(project_dir) [file normalize $opts(project_dir)]

puts "Origin directory:  $opts(origin_dir)"
puts "Project directory: $opts(project_dir)"
puts "Project name:      $opts(project_name)"
puts "BD name:           $opts(bd_name)"

create_project -force $opts(project_name) $opts(project_dir) -part xc7z020clg400-1
set_pynq_z2_board_part $opts(board_part)
add_npu_rtl $opts(origin_dir)
create_npu_bd $opts(bd_name)

if {$opts(build_bitstream)} {
    launch_runs synth_1 -jobs $opts(jobs)
    wait_on_run synth_1
    launch_runs impl_1 -to_step write_bitstream -jobs $opts(jobs)
    wait_on_run impl_1
} else {
    puts "Project and BD created. Re-run with --build-bitstream to build .bit/.hwh outputs."
}

puts "Done. Open project: [file join $opts(project_dir) $opts(project_name).xpr]"
