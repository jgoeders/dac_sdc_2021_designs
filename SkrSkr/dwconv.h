#pragma once
#ifndef DWCONV_H_
#define DWCONV_H_

#include "stream_tools.h"

// depthwise conv3x3
template<
    unsigned N_IO,     // # of IO
    unsigned N_CH,     // # of CHannel
    unsigned BIT_IN,    // nBit of INput
    unsigned BIT_WT,    // nBit of WeighT
    unsigned BIT_OUT,   // nBit of OUTput
    unsigned VEC_LEN    // ROW * COL
>
void dwconv_3x3(data_stream<N_IO * BIT_IN>& in, data_stream<N_IO * BIT_OUT>& out, const ap_uint<N_IO * BIT_WT> weight[N_CH / N_IO][9])
// origin:    ap_int<BIT_WT> wt[9][N_CH]
// reshape:   ap_int<BIT_WT> wt[9][N_CH / N_IO][N_IO]
// transpose: ap_int<BIT_WT> wt[N_CH / N_IO][9][N_IO]
// reshape:   ap_uint<N_IO * BIT_WT> wt[N_CH / N_IO][9]
{
    static_assert(N_CH >= N_IO, "dwconv_3x3");
    static_assert(N_CH % N_IO == 0, "dwconv_3x3");
    constexpr unsigned FOLD = N_CH / N_IO;
    constexpr unsigned ITERS = VEC_LEN;

    assert(in.size() == VEC_LEN * FOLD * 9);
    assert(out.empty());

    #pragma HLS DATAFLOW

    ap_int<BIT_OUT> acc[N_IO];
    #pragma HLS ARRAY_PARTITION variable=acc complete dim=1

    for (unsigned i = 0; i < N_IO; ++i)
    {
        #pragma HLS UNROLL
        acc[i] = 0;
    }

    for (unsigned it = 0; it < ITERS; ++it)
    {
        for (unsigned f = 0; f < FOLD; ++f)
        {
            for (unsigned k = 0; k < 9; ++k)
            {
                #pragma HLS PIPELINE II=1
                // load
                ap_uint<N_IO * BIT_IN> in_buf = in.read();
                ap_uint<N_IO * BIT_WT> wt_buf = weight[f][k];

                // calc
                for (unsigned i = 0; i < N_IO; ++i)
                {
                    #pragma HLS UNROLL
                    ap_uint<BIT_IN> x = in_buf(SLICE(BIT_IN, i));
                    ap_int<BIT_WT>  y = wt_buf(SLICE(BIT_WT, i));
                    acc[i] += x * y;
                }

                // output
                if (k == 8)
                {
                    ap_uint<N_IO * BIT_OUT> out_buf;
                    for (unsigned i = 0; i < N_IO; ++i)
                    {
                        #pragma HLS UNROLL
                        out_buf(SLICE(BIT_OUT, i)) = acc[i];
                        acc[i] = 0;
                    }
                    out.write(out_buf);
                }
            }
        }
    }

    assert(in.empty());
    assert(out.size() == VEC_LEN * FOLD);
    return;
}

#endif
