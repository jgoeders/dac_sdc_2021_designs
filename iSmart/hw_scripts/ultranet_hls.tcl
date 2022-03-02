############################################################
## This file is generated automatically by Vivado HLS.
## Please DO NOT edit it.
## Copyright (C) 1986-2019 Xilinx, Inc. All Rights Reserved.
############################################################
open_project ultra_core
set_top ultra_net
add_files ../src/ultranet.cpp -cflags "-std=c++11" -csimflags "-std=c++11"

open_solution "ultracore_125"
set_part {xczu3eg-sbva484-1-e}
create_clock -period 8 -name default
#source "./HLS_SINGLE/solution1/directives.tcl"
#csim_design
csynth_design


export_design -format ip_catalog
