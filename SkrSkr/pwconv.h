#pragma once
#ifndef PWCONV_H_
#define PWCONV_H_

#include "stream_tools.h"

void MAC(ap_int<5> mul_a1, ap_int<5> mul_a2, ap_uint<8> mul_b, ap_int<13>& rel1, ap_int<13>& rel2){
#pragma HLS INLINE
	ap_int<20> concatnum = ((ap_int<20>)mul_a1 << 14) + mul_a2;
	ap_int<30> result = concatnum* mul_b;
	rel1 = result(26,14) + result(13,13);
	rel2 = result(12,0);
}

// matrix multiplication
// # MACs = N_IN * N_OUT
template<
    unsigned N_IN,      // # of INput
    unsigned N_OUT,     // # of OUTput
    unsigned N_ICH,     // # of Input CHannel
    unsigned N_OCH,     // # of Output CHannel
    unsigned BIT_IN,    // nBit of INput
    unsigned BIT_WT,    // nBit of WeighT
    unsigned BIT_OUT,   // nBit of OUTput
    unsigned VEC_LEN
>
void pwconv(
    data_stream<N_IN * BIT_IN>& in, 
    data_stream<N_OUT * BIT_OUT>& out, 
    const ap_uint<N_OUT * N_IN * BIT_WT> weight[N_OCH / N_OUT][N_ICH / N_IN]
)
{
    static_assert(N_ICH >= N_IN,  "pwconv");
    static_assert(N_OCH >= N_OUT, "pwconv");
    static_assert(N_ICH % N_IN  == 0, "pwconv");
    static_assert(N_OCH % N_OUT == 0, "pwconv");
    static_assert(N_OUT % 2 == 0, "pwconv");

    constexpr unsigned FOLD_I = N_ICH / N_IN;
    constexpr unsigned FOLD_O = N_OCH / N_OUT;
    constexpr unsigned ITERS = VEC_LEN;

    assert(in.size() == VEC_LEN * FOLD_I);
    assert(out.empty());

    #pragma HLS DATAFLOW

    ap_uint<N_IN * BIT_IN> line[FOLD_I];
    ap_int<BIT_OUT> acc[N_OUT];
    #pragma HLS ARRAY_PARTITION variable=acc complete dim=1

    for (unsigned o = 0; o < N_OUT; ++o)
    {
        #pragma HLS UNROLL
        acc[o] = 0;
    }

    for (unsigned it = 0; it < ITERS; ++it)
    {
        for (unsigned fo = 0; fo < FOLD_O; ++fo)
        {
            for (unsigned fi = 0; fi < FOLD_I; ++fi)
            {
                #pragma HLS PIPELINE II=1
                // load
                ap_uint<N_IN * BIT_IN> in_buf;
                if (fo == 0)
                {
                    in_buf = in.read();
                    line[fi] = in_buf;
                }
                else
                {
                    in_buf = line[fi];
                }
                ap_uint<N_OUT * N_IN * BIT_WT> wt_buf = weight[fo][fi];

                PE_loop:for (unsigned i = 0; i < N_IN; ++i)
                {
                    #pragma HLS UNROLL
                    ap_uint<BIT_IN> x = in_buf(SLICE(BIT_IN, i));
//                    for (unsigned o = 0; o < N_OUT; ++o)
//                    {
//                        ap_int<BIT_WT> y = wt_buf(SLICE(BIT_WT, N_IN * o + i));
//                        acc[o] += x * y;
//                    }
					for (unsigned o = 0; o < N_OUT / 2; ++o)
					{

						ap_int<BIT_WT> y1 = wt_buf(SLICE(BIT_WT, N_IN * o * 2 + i));
						ap_int<BIT_WT> y2 = wt_buf(SLICE(BIT_WT, N_IN * o * 2 + N_IN + i));
						ap_int<13> tem1,tem2;
						MAC(y1, y2, x, tem1, tem2);
						acc[2 * o] += tem1;
						acc[2 * o + 1] += tem2;
					}
                }
            }
            ap_uint<N_OUT * BIT_OUT> out_buf;
            for (unsigned o = 0; o < N_OUT; ++o)
            {
                #pragma HLS UNROLL
                out_buf(SLICE(BIT_OUT, o)) = acc[o];
                acc[o] = 0;
            }
            out.write(out_buf);
        }
    }

    assert(in.empty());
    assert(out.size() == VEC_LEN * FOLD_O);
    return;
};


template<
    unsigned N_IN,      // # of INput
    unsigned N_OUT,     // # of OUTput
    unsigned N_ICH,     // # of Input CHannel
    unsigned N_OCH,     // # of Output CHannel
    unsigned BIT_IN,    // nBit of INput
    unsigned BIT_WT,    // nBit of WeighT
    unsigned BIT_OUT,   // nBit of OUTput
    unsigned VEC_LEN
>
void pwconv_single(
    data_stream<N_IN * BIT_IN>& in,
    data_stream<N_OUT * BIT_OUT>& out,
    const ap_uint<N_OUT * N_IN * BIT_WT> weight[N_OCH / N_OUT][N_ICH / N_IN]
)
{
    static_assert(N_ICH >= N_IN,  "pwconv");
    static_assert(N_OCH >= N_OUT, "pwconv");
    static_assert(N_ICH % N_IN  == 0, "pwconv");
    static_assert(N_OCH % N_OUT == 0, "pwconv");

    constexpr unsigned FOLD_I = N_ICH / N_IN;
    constexpr unsigned FOLD_O = N_OCH / N_OUT;
    constexpr unsigned ITERS = VEC_LEN;

    assert(in.size() == VEC_LEN * FOLD_I);
    assert(out.empty());

    #pragma HLS DATAFLOW

    ap_uint<N_IN * BIT_IN> line[FOLD_I];
    ap_int<BIT_OUT> acc[N_OUT];
    #pragma HLS ARRAY_PARTITION variable=acc complete dim=1

    for (unsigned o = 0; o < N_OUT; ++o)
    {
        #pragma HLS UNROLL
        acc[o] = 0;
    }

    for (unsigned it = 0; it < ITERS; ++it)
    {
        for (unsigned fo = 0; fo < FOLD_O; ++fo)
        {
            for (unsigned fi = 0; fi < FOLD_I; ++fi)
            {
                #pragma HLS PIPELINE II=1
                // load
                ap_uint<N_IN * BIT_IN> in_buf;
                if (fo == 0)
                {
                    in_buf = in.read();
                    line[fi] = in_buf;
                }
                else
                {
                    in_buf = line[fi];
                }
                ap_uint<N_OUT * N_IN * BIT_WT> wt_buf = weight[fo][fi];

                for (unsigned i = 0; i < N_IN; ++i)
                {
                    #pragma HLS UNROLL
                    ap_uint<BIT_IN> x = in_buf(SLICE(BIT_IN, i));
                    for (unsigned o = 0; o < N_OUT; ++o)
                    {
                        ap_int<BIT_WT> y = wt_buf(SLICE(BIT_WT, N_IN * o + i));
                        acc[o] += x * y;
                    }
                }
            }
            ap_uint<N_OUT * BIT_OUT> out_buf;
            for (unsigned o = 0; o < N_OUT; ++o)
            {
                #pragma HLS UNROLL
                out_buf(SLICE(BIT_OUT, o)) = acc[o];
                acc[o] = 0;
            }
            out.write(out_buf);
        }
    }

    assert(in.empty());
    assert(out.size() == VEC_LEN * FOLD_O);
    return;
};

// matrix multiplication
// # MACs = N_IN * N_OUT
template<
    unsigned N_IN,      // # of INput
    unsigned N_OUT,     // # of OUTput
    unsigned N_ICH,     // # of Input CHannel
    unsigned N_OCH,     // # of Output CHannel
    unsigned BIT_IN,    // nBit of INput
    unsigned BIT_WT,    // nBit of WeighT
    unsigned BIT_OUT,   // nBit of OUTput
    unsigned VEC_LEN
>
void pwconv_old(
    data_stream<N_IN * BIT_IN>& in, 
    data_stream<N_OUT * BIT_OUT>& out, 
    const ap_uint<N_OUT * N_IN * BIT_WT> weight[N_OCH / N_OUT][N_ICH / N_IN]
)
{
    static_assert(N_ICH >= N_IN,  "pwconv");
    static_assert(N_OCH >= N_OUT, "pwconv");
    static_assert(N_ICH % N_IN  == 0, "pwconv");
    static_assert(N_OCH % N_OUT == 0, "pwconv");

    constexpr unsigned FOLD_I = N_ICH / N_IN;
    constexpr unsigned FOLD_O = N_OCH / N_OUT;
    constexpr unsigned ITERS = VEC_LEN * FOLD_I * FOLD_O;

    assert(in.size() == VEC_LEN * FOLD_I);
    assert(out.empty());

    ap_uint<N_IN * BIT_IN> line[FOLD_I];
    ap_int<BIT_OUT> acc[N_OUT];
    #pragma HLS ARRAY_PARTITION variable=acc complete dim=1

    unsigned fi = 0;
    unsigned fo = 0;

    for (unsigned o = 0; o < N_OUT; ++o)
    {
        #pragma HLS UNROLL
        acc[o] = 0;
    }

    for (unsigned it = 0; it < ITERS; ++it)
    {
        #pragma HLS PIPELINE II=1
        ap_uint<N_IN * BIT_IN> in_buf;
        if (fo == 0)
        {
            in_buf = in.read();
            line[fi] = in_buf;
        }
        else
        {
            in_buf = line[fi];
        }
        ap_uint<N_OUT * N_IN * BIT_WT> wt_buf = weight[fo][fi];

        for (unsigned i = 0; i < N_IN; ++i)
        {
            #pragma HLS UNROLL
            ap_uint<BIT_IN> x = in_buf(SLICE(BIT_IN, i));
            for (unsigned o = 0; o < N_OUT; ++o)
            {
                ap_int<BIT_WT> y = wt_buf(SLICE(BIT_WT, N_IN * o + i));
                acc[o] += x * y;
            }
        }

        if (fi == FOLD_I - 1)
        {
            ap_uint<N_OUT * BIT_OUT> out_buf;
            for (unsigned o = 0; o < N_OUT; ++o)
            {
                #pragma HLS UNROLL
                out_buf(SLICE(BIT_OUT, o)) = acc[o];
                acc[o] = 0;
            }
            out.write(out_buf);
            fi = 0;
            fo = (fo != FOLD_O - 1) ? fo + 1 : 0;
        }
        else
        {
            ++fi;
        }
    }

    assert(in.empty());
    assert(out.size() == VEC_LEN * FOLD_O);
    return;
};

#endif
