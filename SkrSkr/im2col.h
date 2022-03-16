#include "skynet_flow.h"

template<
	unsigned W_IO,  // Width of IO	  N_IO * BIT
	unsigned W_CH,  // Width of CHannel N_CH * BIT
	unsigned ROW,
	unsigned COL,
	unsigned N_BATCH,
	unsigned PADVAL = 0x00
>
void im2col_3x3(data_stream<W_IO>& in, data_stream<W_IO>& out)
{
	static_assert(W_CH >= W_IO, "im2col");
	static_assert(W_CH % W_IO == 0, "im2col");
	constexpr unsigned FOLD = W_CH / W_IO;
	constexpr unsigned COL_OFFSET = 2;
	const ap_uint<W_IO> PAD_AP = PADVAL;

	assert(in.size() == N_BATCH * ROW * COL * FOLD);
	assert(out.empty());

	//#pragma HLS DATAFLOW

	static ap_uint<2> idx[3] = {0, 1, 2};
	#pragma HLS ARRAY_PARTITION variable=idx complete dim=1
	ap_uint<W_IO> line[3][COL][FOLD];
	
	for (unsigned c = 0; c < COL + COL_OFFSET; ++c)
	{
		for (unsigned f = 0; f < FOLD; ++f)
		{
			#pragma HLS PIPELINE II=1
			if (c < COL)
			{
				line[idx[2]][c][f] = in.read();
			}
			else
			{
				line[idx[0]][c - COL][f] = in.read();
			}
		}
	}

	// main loop
	for (unsigned b = 0; b < N_BATCH; ++b)
	{
		for (unsigned r = 0; r < ROW; ++r)
		{
			for (unsigned c = 0; c < COL; ++c)
			{
				for (unsigned f = 0; f < FOLD; ++f)
				{
					#pragma HLS PIPELINE II=9
					// idx shift
					if (c == 0 && f == 0)
					{
						ap_uint<2> it = idx[0];
						idx[0] = idx[1];
						idx[1] = idx[2];
						idx[2] = it;
					}
					// load data
					if (b < N_BATCH - 1 || r < ROW - 2 || (r == ROW - 2 && c < COL - COL_OFFSET))
					{
						ap_uint<W_IO> in_buf = in.read();
						if (c < COL - COL_OFFSET)
						{
							line[idx[2]][c + COL_OFFSET][f] = in_buf;
						}
						else
						{
							line[idx[0]][c + COL_OFFSET - COL][f] = in_buf;
						}
					}

					// output, II = 9
					// (r == 0 || r == ROW - 1 || c == 0 || c == COL - 1) --> padding
					out.write((r != 0 && c != 0) ? 
						line[idx[0]][c - 1][f] : PAD_AP);
					out.write((r != 0) ? 
						line[idx[0]][c + 0][f] : PAD_AP);
					out.write((r != 0 && c != COL - 1)? 
						line[idx[0]][c + 1][f] : PAD_AP);
					out.write((c != 0) ? 
						line[idx[1]][c - 1][f] : PAD_AP);
					out.write(
						line[idx[1]][c + 0][f]);
					out.write((c != COL - 1) ? 
						line[idx[1]][c + 1][f] : PAD_AP);
					out.write((r != ROW - 1 && c != 0) ? 
						line[idx[2]][c - 1][f] : PAD_AP);
					out.write((r != ROW - 1) ? 
						line[idx[2]][c + 0][f] : PAD_AP);
					out.write((r != ROW - 1 && c != COL - 1) ? 
						line[idx[2]][c + 1][f] : PAD_AP);
				}
			}
		}
	}

	assert(in.empty());
	assert(out.size() == N_BATCH * ROW * COL * FOLD * 9);
	return;
}