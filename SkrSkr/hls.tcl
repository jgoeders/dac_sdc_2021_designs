############################################################
## This file is generated automatically by Vitis HLS.
## Please DO NOT edit it.
## Copyright 1986-2020 Xilinx, Inc. All Rights Reserved.
############################################################
open_project VHLS
set_top skynet_flow
add_files bypass_unit.h -cflags "--std=c++14"
add_files dwconv.h -cflags "--std=c++14"
add_files findMax.h -cflags "--std=c++14"
add_files maxPool.h -cflags "--std=c++14"
add_files norm_actv.h -cflags "--std=c++14"
add_files pwconv.h -cflags "--std=c++14"
add_files resize.h -cflags "--std=c++14"
add_files skynet_flow.cpp -cflags "--std=c++14"
add_files skynet_flow.h -cflags "--std=c++14"
add_files skynet_para_256.h -cflags "--std=c++14"
add_files stream_tools.h -cflags "--std=c++14"
add_files -tb test.cpp -cflags "-O2 --std=c++14 -Wno-unknown-pragmas -lcnpy -lz"
add_files -tb zq_tools.h -cflags "-Wno-unknown-pragmas"
open_solution "pwc2" -flow_target vivado
set_part {xczu3eg-sbva484-1-i}
create_clock -period 4 -name default
#csim_design
csynth_design
#cosim_design
export_design -rtl verilog -format ip_catalog -description "pwc2"