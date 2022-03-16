#pragma once
#ifndef BYPASS_UNIT_H_
#define BYPASS_UNIT_H_

#include "stream_tools.h"
#include "skynet_flow.h"

using signal_fifo = data_stream<1>;

constexpr unsigned BP_IO = 2;
constexpr unsigned BP_BIT = BIT_ACTV;
constexpr unsigned BP_ROW = 20;
constexpr unsigned BP_COL = 40;
constexpr unsigned BP_CH = 768;
constexpr unsigned BP_BLK = BP_CH / 2 / BP_IO;

constexpr unsigned BP_FIFO_0_DEPTH = 4 * BP_COL * BP_BLK;
constexpr unsigned BP_FIFO_1_DEPTH = 3 * BP_COL * BP_BLK;
constexpr unsigned BP_STREAM_DEPTH = 32;

void bypass_send_reOrg(data_stream<BP_IO * BP_BIT>& in, data_stream<BP_IO * BP_BIT>& bp_fifo0, data_stream<BP_IO * BP_BIT>& bp_fifo1)
{
    // input:     (40, 80, 192)
    // reshape:   (20, 2, 40, 384)
    // transpose: (20, 40, 2, 384)
    // reshape:   (20, 40, 768)

    constexpr unsigned ITERS = N_BATCH * BP_ROW;

    assert(in.size() == ITERS * BP_COL * BP_CH / BP_IO);
    assert(bp_fifo0.empty());
    assert(bp_fifo1.empty());

    #pragma HLS DATAFLOW

    for (unsigned it = 0; it < ITERS; ++it)
    {
        for (unsigned t = 0; t < 2; ++t)
        {
            for (unsigned i = 0; i < BP_COL * BP_BLK; ++i)
            {
                #pragma HLS PIPELINE II=1
                ap_uint<BP_IO * BP_BIT> buf;
                buf = in.read();
                if (t == 0)
                {
                    bp_fifo0.write(buf);
                }
                else
                {
                    bp_fifo1.write(buf);
                }
            }
        }
    }

    assert(in.empty());
    assert(bp_fifo0.size() == ITERS * BP_COL * BP_BLK);
    assert(bp_fifo1.size() == ITERS * BP_COL * BP_BLK);
    return;
};

void bypass_recv(data_stream<BP_IO * BP_BIT>& bp_fifo0, data_stream<BP_IO * BP_BIT>& bp_fifo1, data_stream<BP_IO * BP_BIT>& out)
{
    constexpr unsigned ITERS = N_BATCH * BP_ROW * BP_COL;

    assert(bp_fifo0.size() == ITERS * BP_BLK);
    assert(bp_fifo1.size() == ITERS * BP_BLK);
    assert(out.empty());

    #pragma HLS DATAFLOW

    for (unsigned it = 0; it < ITERS; ++it)
    {
        for (unsigned t = 0; t < 2; ++t)
        {
            for (unsigned i = 0; i < BP_BLK; ++i)
            {
                #pragma HLS PIPELINE II=1
                ap_uint<BP_IO * BP_BIT> buf;
                if (t == 0)
                {
                    buf = bp_fifo0.read();
                }
                else
                {
                    buf = bp_fifo1.read();
                }
                out.write(buf);
            }
        }
    }

    assert(bp_fifo0.empty());
    assert(bp_fifo1.empty());
    assert(out.size() == ITERS * BP_CH / BP_IO);
    return;
};

#endif
