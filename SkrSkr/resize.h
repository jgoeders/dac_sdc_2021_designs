#pragma once
#ifndef RESIZE_H_
#define RESIZE_H_

#include "stream_tools.h"
#include "skynet_flow.h"

const unsigned RESIZE_NIN = 6;
const unsigned RESIZE_NOUT = 3;
constexpr unsigned RS_STREAM_DEPTH = COL0;

void resize(data_stream<RESIZE_NIN * BIT_ACTV>& in, data_stream<RESIZE_NOUT * BIT_ACTV>& out)
{
    static_assert(COLIN == COL0 * 2, "resize");
    static const ap_uint<1> row_mask[ROWIN] = {
        1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 
        0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 
        0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 
        1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 
        1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 
        0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 
        0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 
        1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 
        0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 
        0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 
        1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 
        1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 
    };

    assert(in.size() == N_BATCH * ROWIN * COLIN / 2);
    assert(out.empty());

    #pragma HLS DATAFLOW

    for (unsigned b = 0; b < N_BATCH; ++b)
    {
        for (unsigned r = 0; r < ROWIN; ++r)
        {
            for (unsigned c = 0; c < COL0; ++c)
            {
                #pragma HLS PIPELINE II=1
                ap_uint<RESIZE_NIN * BIT_ACTV> in_buf = in.read();
                if (row_mask[r] == 0x1)
                {
                    ap_uint<RESIZE_NOUT * BIT_ACTV> out_buf = in_buf(SLICE(RESIZE_NOUT * BIT_ACTV, 1));
                    out.write(out_buf);
                }
            }
        }
    }

    assert(in.empty());
    assert(out.size() == N_BATCH * ROW0 * COL0);
    return;
};

#endif
