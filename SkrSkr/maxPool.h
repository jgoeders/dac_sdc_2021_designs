#ifndef MAXPOOL_H_
#define MAXPOOL_H_

#include "stream_tools.h"

template<unsigned N_IO, unsigned BIT>
ap_uint<N_IO * BIT> max(const ap_uint<N_IO * BIT>& x, const ap_uint<N_IO * BIT>& y)
{
	#pragma HLS INLINE
    ap_uint<N_IO * BIT> z;
    for (unsigned i = 0; i < N_IO; ++i)
    {
        #pragma HLS UNROLL
        ap_uint<BIT> a = x(SLICE(BIT, i));
        ap_uint<BIT> b = y(SLICE(BIT, i));
        ap_uint<BIT> c = a > b ? a : b;
        z(SLICE(BIT, i)) = c;
    }
    return z;
};

template<
    unsigned N_IO, 
    unsigned N_CH, 
    unsigned BIT, 
    unsigned ROW, 
    unsigned COL,
    unsigned N_BATCH
>
void maxPool2x2(data_stream<N_IO * BIT>& in, data_stream<N_IO * BIT>& out)
{
    static_assert(N_CH >= N_IO, "maxPool2x2");
    static_assert(N_CH % N_IO == 0, "maxPool2x2");
    static_assert(ROW % 2 == 0, "maxPool2x2");
    static_assert(COL % 2 == 0, "maxPool2x2");

    constexpr unsigned FOLD = N_CH / N_IO;
    constexpr unsigned ITER = N_BATCH * ROW * COL * FOLD;
    assert(in.size() == ITER);
    assert(out.empty());

    #pragma HLS DATAFLOW

    ap_uint<N_IO * BIT> line[COL / 2][FOLD];

    for (unsigned r = 0; r < N_BATCH * ROW; ++r)
    {
        for (unsigned c = 0; c < COL; ++c)
        {
            for (unsigned f = 0; f < FOLD; ++f)
            {
                #pragma HLS PIPELINE II=1
                const unsigned idx = c >> 1;
                ap_uint<N_IO * BIT> in_buf = in.read();
                ap_uint<N_IO * BIT> out_buf;
                if ((r & 0x1) == 0)
                {
                    if ((c & 0x1) == 0)
                    {
                        // 0x0
                        line[idx][f] = in_buf;
                    }
                    else
                    {
                        // 0x1
                        out_buf = max<N_IO, BIT>(in_buf, line[idx][f]);
                        line[idx][f] = out_buf;
                    }
                }
                else
                {
                    if ((c & 0x1) == 0)
                    {
                        // 0x2
                        out_buf = max<N_IO, BIT>(in_buf, line[idx][f]);
                        line[idx][f] = out_buf;
                    }
                    else
                    {
                        // 0x3
                        out_buf = max<N_IO, BIT>(in_buf, line[idx][f]);
                        out.write(out_buf);
                    }
                }
                // const unsigned state = ((r & 0x1) << 1) | (c & 0x1);
                // switch (state)
                // {
                // case 0x0:
                //     line[idx][f] = in_buf;
                //     break;
                // case 0x1:
                //     out_buf = max<N_IO, BIT>(in_buf, line[idx][f]);
                //     line[idx][f] = out_buf;
                //     break;
                // case 0x2:
                //     out_buf = max<N_IO, BIT>(in_buf, line[idx][f]);
                //     line[idx][f] = out_buf;
                //     break;
                // case 0x3:
                //     out_buf = max<N_IO, BIT>(in_buf, line[idx][f]);
                //     out.write(out_buf);
                //     break;
                // default:
                //     assert(false);
                //     break;
                // }
            }
        }
    }

    assert(in.empty());
    assert(out.size() == ITER / 4);
    return;
};

#endif
