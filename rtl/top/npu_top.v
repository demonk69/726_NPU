`ifdef VERILATOR_TRACE
always @(posedge sys_clk) begin
    if (ctrl_vec_consume)
        $display("[V_TRACE] fire: w_lane0=0x%08h a_lane0=0x%08h",
                 w_ppb_rd_vec[0*DATA_W +: DATA_W],
                 a_ppb_rd_vec[0*DATA_W +: DATA_W]);
end
`endif