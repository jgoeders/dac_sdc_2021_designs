#pragma once
#ifndef NORM_ACTV_H_
#define NORM_ACTV_H_

#include "stream_tools.h"

template<
    unsigned N_IO,     // # of IO
    unsigned N_CH,     // # of CHannel
    unsigned BIT_IN,    // nBit of INput
    unsigned BIT_B,     // nBit of Bias
    unsigned BIT_M,     // nBit of Mul
    unsigned BIT_R,     // nBit of middle Result
    unsigned BIT_OUT,   // nBit of OUTput
    unsigned R_SHIFT,
    unsigned VEC_LEN    // ROW * COL
>
void norm_actv(
    data_stream<N_IO * BIT_IN>& in, 
    data_stream<N_IO * BIT_OUT>& out, 
    const ap_uint<N_IO * BIT_B> bias[N_CH / N_IO],
    const ap_uint<N_IO * BIT_M> mult[N_CH / N_IO]
)
{
    static_assert(N_CH >= N_IO, "norm_actv");
    static_assert(N_CH % N_IO == 0, "norm_actv");
    constexpr unsigned FOLD = N_CH / N_IO;
    constexpr unsigned ITERS = VEC_LEN * FOLD;
    constexpr unsigned MAXOUT = (1 << BIT_OUT) - 1;

    assert(in.size() == VEC_LEN * FOLD);
    assert(out.empty());

    #pragma HLS DATAFLOW

    unsigned f = 0;
    for (unsigned it = 0; it < ITERS; ++it)
    {

        #pragma HLS PIPELINE II=1
        ap_uint<N_IO * BIT_IN> in_buf = in.read();
        ap_uint<N_IO * BIT_B> b_buf = bias[f];
        ap_uint<N_IO * BIT_M> m_buf = mult[f];
        ap_uint<N_IO * BIT_OUT> out_buf;

        for (unsigned i = 0; i < N_IO; ++i)
        {
            #pragma HLS UNROLL
            // UnRoll
            ap_int<BIT_IN> a = in_buf(SLICE(BIT_IN, i));

            ap_int<BIT_B> b =  b_buf(SLICE(BIT_B , i));
            ap_int<BIT_M>  m =  m_buf(SLICE(BIT_M , i));
            //ap_int<20> tmp = (a + b);
			//#pragma HLS BIND_OP variable=tmp op=add
            //ap_int<BIT_R>  x = (tmp * m) >> R_SHIFT;
            ap_int<BIT_R>  x = ((a + b) * m) >> R_SHIFT;

            ap_uint<BIT_OUT> y = 0;
            if (x > 0)
            {
                y = x < MAXOUT ? ap_uint<BIT_OUT>(x) : ap_uint<BIT_OUT>(MAXOUT);
            }
            else
            {
                y = 0;
            }
            out_buf(SLICE(BIT_OUT, i)) = y;
        }
        out.write(out_buf);
        f = (f != FOLD - 1) ? f + 1 : 0;
    }

    assert(in.empty());
    assert(out.size() == VEC_LEN * FOLD);
    return;
}

#endif
