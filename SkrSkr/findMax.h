#pragma once
#ifndef FINDMAX_H_
#define FINDMAX_H_

#include "stream_tools.h"
#include "skynet_flow.h"

constexpr unsigned FINDMAX_NLINE = 14;

void findMax(data_stream<BIT_CONV>& in, data_stream<BIT_CONV>& out)
{
    assert(in.size() == N_BATCH * ROW3 * COL3 * L6_PW_NOCH);
    assert(out.empty());

    constexpr int32_t MIN_16 = -(1 << (BIT_CONV - 1));

    #pragma HLS DATAFLOW

    ap_int<BIT_CONV> data_m[2][4];
    ap_int<BIT_CONV> conf_m[2];
    ap_uint<BIT_CONV> pos_m[2][2];

    #pragma HLS ARRAY_PARTITION variable=data_m complete dim=0
    #pragma HLS ARRAY_PARTITION variable=conf_m complete dim=0
    #pragma HLS ARRAY_PARTITION variable=pos_m complete dim=0

    for (unsigned b = 0; b < N_BATCH; ++b)
    {
        conf_m[0] = MIN_16;
        conf_m[1] = MIN_16;
        for (unsigned r = 0; r < ROW3; ++r)
        {
            for (unsigned c = 0; c < COL3; ++c)
            {
                #pragma HLS PIPELINE II=10
                for (unsigned t = 0; t < 2; ++t)
                {
                    #pragma HLS UNROLL
                    ap_int<BIT_CONV> d0 = in.read();
                    ap_int<BIT_CONV> d1 = in.read();
                    ap_int<BIT_CONV> d2 = in.read();
                    ap_int<BIT_CONV> d3 = in.read();
                    ap_int<BIT_CONV> conf = in.read();
                    if (conf > conf_m[t])
                    {
                        data_m[t][0] = d0;
                        data_m[t][1] = d1;
                        data_m[t][2] = d2;
                        data_m[t][3] = d3;
                        conf_m[t] = conf;
                        pos_m[t][0] = c;
                        pos_m[t][1] = r;
                    }
                }
            }
        }
        for (unsigned t = 0; t < 2; ++t)
        {
            #pragma HLS UNROLL
            out.write(data_m[t][0]);
            out.write(data_m[t][1]);
            out.write(data_m[t][2]);
            out.write(data_m[t][3]);
            out.write(conf_m[t]);
            out.write(pos_m[t][0]);
            out.write(pos_m[t][1]);
        }
    }

    assert(in.empty());
    assert(out.size() == N_BATCH * FINDMAX_NLINE);
    return;
};

#endif
