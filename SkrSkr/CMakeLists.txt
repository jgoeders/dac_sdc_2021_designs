cmake_minimum_required(VERSION 3.2.2)
project(SkyNet)

set(CMAKE_CXX_STANDARD 14)
include_directories(/tools/Xilinx/Vivado/2020.2/include/)
add_executable(SkyNet 
bypass_unit.h findMax.h maxPool.h norm_actv.h resize.h dwconv.h pwconv.h skynet_flow.cpp tb_skynet.cpp)