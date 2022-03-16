#include "skynet_flow.h"
#include "dwconv.h"
#include "pwconv.h"
#include "norm_actv.h"
#include "maxPool.h"
#include "bypass_unit.h"
#include "resize.h"
#include "findMax.h"
#include "im2col.h"

void skynet_flow(data_stream<N_IN * BIT_ACTV>& in, axiu_stream<N_OUT * BIT_CONV>& out)
{
	#pragma HLS INTERFACE ap_ctrl_none port=return
	#pragma HLS INTERFACE axis register both port=in
	#pragma HLS INTERFACE axis register both port=out

	#pragma HLS DATAFLOW

	// input reshape
	data_stream<2 * BIT_ACTV> s_in_reshape("s_in_reshape");
	#pragma HLS STREAM variable=s_in_reshape depth=DEFAULT_DEPTH dim=1
	#pragma HLS RESOURCE variable=s_in_reshape core=FIFO_SRL
	reduceWidth<
		N_IN * BIT_ACTV, 
		2 * BIT_ACTV, 
		ROWIN * COLIN * L0_DW_NCH / N_IN * N_BATCH
	>(in, s_in_reshape);

	data_stream<RESIZE_NIN * BIT_ACTV> s_resize_in("resize_in");
	#pragma HLS STREAM variable=s_resize_in depth=DEFAULT_DEPTH dim=1
	#pragma HLS RESOURCE variable=s_resize_in core=FIFO_SRL
	expandWidth<
		2 * BIT_ACTV, 
		RESIZE_NIN * BIT_ACTV, 
		ROWIN * COLIN * L0_DW_NCH / 2 * N_BATCH
	>(s_in_reshape, s_resize_in);

	// Resize
	data_stream<RESIZE_NOUT * BIT_ACTV> s_resize("resize");
	#pragma HLS STREAM variable=s_resize depth=RS_STREAM_DEPTH dim=1
	#pragma HLS RESOURCE variable=s_resize core=FIFO_LUTRAM
	resize(s_resize_in, s_resize);

	// Bundle #1
	static_assert(3 == L0_DW_NCH, "Bundle 1");
	static_assert(RESIZE_NOUT == L0_DW_NCH, "Bundle 1");
	static_assert(L0_DW_NCH == L0_PW_NICH, "Bundle 1");

	data_stream<L0_DW_NIO * BIT_ACTV> s_l0_im2co("L0_im2col");
	#pragma HLS STREAM variable=s_l0_im2co depth=DEFAULT_DEPTH dim=1
	#pragma HLS RESOURCE variable=s_l0_im2co core=FIFO_SRL
	im2col_3x3<
		L0_DW_NIO * BIT_ACTV, 
		L0_DW_NCH * BIT_ACTV, 
		ROW0, COL0, 
		N_BATCH,
		0x80808080
	>(s_resize, s_l0_im2co);

	data_stream<L0_DW_NIO * BIT_CONV> s_l0_dwconv("L0_dwconv");
	#pragma HLS STREAM variable=s_l0_dwconv depth=DEFAULT_DEPTH dim=1
	#pragma HLS RESOURCE variable=s_l0_dwconv core=FIFO_SRL
	dwconv_3x3<
		L0_DW_NIO, 
		L0_DW_NCH, 
		BIT_ACTV, BIT_WT, BIT_CONV, 
		ROW0 * COL0 * N_BATCH
	>(s_l0_im2co, s_l0_dwconv, L0_DW);

	data_stream<L0_DW_NACT * BIT_CONV> s_l0_dwactv_in("L0_dwactv_in");
	#pragma HLS STREAM variable=s_l0_dwactv_in depth=DEFAULT_DEPTH dim=1
	#pragma HLS RESOURCE variable=s_l0_dwactv_in core=FIFO_SRL
	reduceWidth<
		L0_DW_NIO * BIT_CONV, 
		L0_DW_NACT * BIT_CONV, 
		ROW0 * COL0 * L0_DW_NCH / L0_DW_NIO * N_BATCH
	>(s_l0_dwconv, s_l0_dwactv_in);

	data_stream<L0_DW_NACT * BIT_ACTV> s_l0_dwactv("L0_dwactv");
	#pragma HLS STREAM variable=s_l0_dwactv depth=DEFAULT_DEPTH dim=1
	#pragma HLS RESOURCE variable=s_l0_dwactv core=FIFO_SRL
	norm_actv<
		L0_DW_NACT, L0_DW_NCH, 
		BIT_CONV, 
		BIT_BIAS, 
		BIT_MULT, 
		BIT_NORM, 
		BIT_ACTV, 
		R_SHIFT, 
		ROW0 * COL0 * N_BATCH
	>(s_l0_dwactv_in, s_l0_dwactv, L0_DB, L0_DM);

	data_stream<L0_PW_NIN * BIT_ACTV> s_l0_pwconv_in("L0_pwconv_in");
	// L0_PW_NICH/L0_PW_NIN < DEFAULT_DEPTH
	// #pragma HLS STREAM variable=s_l0_pwconv_in depth=L0_PW_NICH/L0_PW_NIN dim=1
	#pragma HLS STREAM variable=s_l0_pwconv_in depth=DEFAULT_DEPTH dim=1
	#pragma HLS RESOURCE variable=s_l0_pwconv_in core=FIFO_SRL
	// #pragma HLS RESOURCE variable=s_l0_pwconv_in core=FIFO_LUTRAM
	expandWidth<
		L0_DW_NACT * BIT_ACTV,
		L0_PW_NIN * BIT_ACTV,
		ROW0 * COL0 * L0_DW_NCH / L0_DW_NACT * N_BATCH
	>(s_l0_dwactv, s_l0_pwconv_in);

	data_stream<L0_PW_NOUT * BIT_CONV> s_l0_pwconv("L0_pwconv");
	#pragma HLS STREAM variable=s_l0_pwconv depth=DEFAULT_DEPTH dim=1
	#pragma HLS RESOURCE variable=s_l0_pwconv core=FIFO_SRL
	pwconv<
		L0_PW_NIN, L0_PW_NOUT,
		L0_PW_NICH, L0_PW_NOCH,
		BIT_ACTV, BIT_WT, BIT_CONV,
		ROW0 * COL0 * N_BATCH
	>(s_l0_pwconv_in, s_l0_pwconv, L0_PW);

	data_stream<L0_PW_NACT * BIT_ACTV> s_l0_pwactv("L0_pwactv");
	#pragma HLS STREAM variable=s_l0_pwactv depth=DEFAULT_DEPTH dim=1
	#pragma HLS RESOURCE variable=s_l0_pwactv core=FIFO_SRL
	norm_actv<
		L0_PW_NACT, L0_PW_NOCH,
		BIT_CONV,
		BIT_BIAS,
		BIT_MULT,
		BIT_NORM,
		BIT_ACTV,
		R_SHIFT,
		ROW0 * COL0 * N_BATCH
	>(s_l0_pwconv, s_l0_pwactv, L0_PB, L0_PM);

	data_stream<L0_PW_NACT * BIT_ACTV> s_l0_pool("L0_pool");
	constexpr unsigned L0_POOL_DEPTH = (COL1 / 2 + 2) * L0_PW_NOCH / L0_PW_NACT;
	#pragma HLS STREAM variable=s_l0_pool depth=L0_POOL_DEPTH dim=1
	#pragma HLS RESOURCE variable=s_l0_pool core=FIFO_BRAM
	maxPool2x2<
		L0_PW_NACT,
		L0_PW_NOCH,
		BIT_ACTV,
		ROW0, COL0,
		N_BATCH
	>(s_l0_pwactv, s_l0_pool);

	// Bundle #2
	static_assert(L0_PW_NOCH == L1_DW_NCH, "Bundle 2");
	static_assert(L1_DW_NCH == L1_PW_NICH, "Bundle 2");

	data_stream<L1_DW_NIO * BIT_ACTV> s_l1_im2col_in("L1_im2col_in");
	#pragma HLS STREAM variable=s_l1_im2col_in depth=DEFAULT_DEPTH dim=1
	#pragma HLS RESOURCE variable=s_l1_im2col_in core=FIFO_SRL
	expandWidth<
		L0_PW_NACT * BIT_ACTV,
		L1_DW_NIO * BIT_ACTV,
		ROW1 * COL1 * L1_DW_NCH / L0_PW_NACT * N_BATCH
	>(s_l0_pool, s_l1_im2col_in);

	data_stream<L1_DW_NIO * BIT_ACTV> s_l1_im2co("L1_im2col");
	#pragma HLS STREAM variable=s_l1_im2co depth=DEFAULT_DEPTH dim=1
	#pragma HLS RESOURCE variable=s_l1_im2co core=FIFO_SRL
	im2col_3x3<
		L1_DW_NIO * BIT_ACTV,
		L1_DW_NCH * BIT_ACTV,
		ROW1, COL1, N_BATCH
	>(s_l1_im2col_in, s_l1_im2co);

	data_stream<L1_DW_NIO * BIT_CONV> s_l1_dwconv("L1_dwconv");
	#pragma HLS STREAM variable=s_l1_dwconv depth=DEFAULT_DEPTH dim=1
	#pragma HLS RESOURCE variable=s_l1_dwconv core=FIFO_SRL
	dwconv_3x3<
		L1_DW_NIO,
		L1_DW_NCH,
		BIT_ACTV, BIT_WT, BIT_CONV,
		ROW1 * COL1 * N_BATCH
	>(s_l1_im2co, s_l1_dwconv, L1_DW);

	data_stream<L1_DW_NACT * BIT_CONV> s_l1_dwactv_in("L1_dwactv_in");
	#pragma HLS STREAM variable=s_l1_dwactv_in depth=DEFAULT_DEPTH dim=1
	#pragma HLS RESOURCE variable=s_l1_dwactv_in core=FIFO_SRL
	reduceWidth<
		L1_DW_NIO * BIT_CONV,
		L1_DW_NACT * BIT_CONV,
		ROW1 * COL1 * L1_DW_NCH / L1_DW_NIO * N_BATCH
	>(s_l1_dwconv, s_l1_dwactv_in);

	data_stream<L1_DW_NACT * BIT_ACTV> s_l1_dwactv("L1_dwactv");
	#pragma HLS STREAM variable=s_l1_dwactv depth=DEFAULT_DEPTH dim=1
	#pragma HLS RESOURCE variable=s_l1_dwactv core=FIFO_SRL
	norm_actv<
		L1_DW_NACT, L1_DW_NCH,
		BIT_CONV,
		BIT_BIAS,
		BIT_MULT,
		BIT_NORM,
		BIT_ACTV,
		R_SHIFT,
		ROW1 * COL1 * N_BATCH
	>(s_l1_dwactv_in, s_l1_dwactv, L1_DB, L1_DM);

	data_stream<L1_PW_NIN * BIT_ACTV> s_l1_pwconv_in("L1_pwconv_in");
	#pragma HLS STREAM variable=s_l1_pwconv_in depth=L1_PW_NICH/L1_PW_NIN dim=1
	#pragma HLS RESOURCE variable=s_l1_pwconv_in core=FIFO_SRL
	// #pragma HLS RESOURCE variable=s_l1_pwconv_in core=FIFO_LUTRAM
	expandWidth<
		L1_DW_NACT * BIT_ACTV,
		L1_PW_NIN * BIT_ACTV,
		ROW1 * COL1 * L1_DW_NCH / L1_DW_NACT * N_BATCH
	>(s_l1_dwactv, s_l1_pwconv_in);

	data_stream<L1_PW_NOUT * BIT_CONV> s_l1_pwconv("L1_pwconv");
	#pragma HLS STREAM variable=s_l1_pwconv depth=DEFAULT_DEPTH dim=1
	#pragma HLS RESOURCE variable=s_l1_pwconv core=FIFO_SRL

	pwconv<
		L1_PW_NIN, L1_PW_NOUT,
		L1_PW_NICH, L1_PW_NOCH,
		BIT_ACTV, BIT_WT, BIT_CONV,
		ROW1 * COL1 * N_BATCH
	>(s_l1_pwconv_in, s_l1_pwconv, L1_PW);

	data_stream<L1_PW_NACT * BIT_CONV> s_l1_pwactv_in("L1_pwactv_in");
	#pragma HLS STREAM variable=s_l1_pwactv_in depth=DEFAULT_DEPTH dim=1
	#pragma HLS RESOURCE variable=s_l1_pwactv_in core=FIFO_SRL
	reduceWidth<
		L1_PW_NOUT * BIT_CONV,
		L1_PW_NACT * BIT_CONV,
		ROW1 * COL1 * L1_PW_NOCH / L1_PW_NOUT * N_BATCH
	>(s_l1_pwconv, s_l1_pwactv_in);

	data_stream<L1_PW_NACT * BIT_ACTV> s_l1_pwactv("L1_pwactv");
	#pragma HLS STREAM variable=s_l1_pwactv depth=DEFAULT_DEPTH dim=1
	#pragma HLS RESOURCE variable=s_l1_pwactv core=FIFO_SRL
	norm_actv<
		L1_PW_NACT, L1_PW_NOCH,
		BIT_CONV,
		BIT_BIAS,
		BIT_MULT,
		BIT_NORM,
		BIT_ACTV,
		R_SHIFT,
		ROW1 * COL1 * N_BATCH
	>(s_l1_pwactv_in, s_l1_pwactv, L1_PB, L1_PM);

	data_stream<L1_PW_NACT * BIT_ACTV> s_l1_pool("L1_pool");
	constexpr unsigned L1_POOL_DEPTH = (COL2 / 2 + 2) * L1_PW_NOCH / L1_PW_NACT;
	#pragma HLS STREAM variable=s_l1_pool depth=L1_POOL_DEPTH dim=1
	#pragma HLS RESOURCE variable=s_l1_pool core=FIFO_BRAM
	maxPool2x2<
		L1_PW_NACT,
		L1_PW_NOCH,
		BIT_ACTV,
		ROW1, COL1, N_BATCH
	>(s_l1_pwactv, s_l1_pool);

	// Bundle #3
	static_assert(L1_PW_NOCH == L2_DW_NCH, "Bundle 3");
	static_assert(L2_DW_NCH == L2_PW_NICH, "Bundle 3");

	data_stream<L2_DW_NIO * BIT_ACTV> s_l2_im2col_in("L2_im2col_in");
	#pragma HLS STREAM variable=s_l2_im2col_in depth=DEFAULT_DEPTH dim=1
	#pragma HLS RESOURCE variable=s_l2_im2col_in core=FIFO_SRL
	expandWidth<
		L1_PW_NACT * BIT_ACTV,
		L2_DW_NIO * BIT_ACTV,
		ROW2 * COL2 * L2_DW_NCH / L1_PW_NACT * N_BATCH
	>(s_l1_pool, s_l2_im2col_in);

	data_stream<L2_DW_NIO * BIT_ACTV> s_l2_im2co("L2_im2col");
	#pragma HLS STREAM variable=s_l2_im2co depth=DEFAULT_DEPTH dim=1
	#pragma HLS RESOURCE variable=s_l2_im2co core=FIFO_SRL
	im2col_3x3<
		L2_DW_NIO * BIT_ACTV,
		L2_DW_NCH * BIT_ACTV,
		ROW2, COL2, N_BATCH
	>(s_l2_im2col_in, s_l2_im2co);

	data_stream<L2_DW_NIO * BIT_CONV> s_l2_dwconv("L2_dwconv");
	#pragma HLS STREAM variable=s_l2_dwconv depth=DEFAULT_DEPTH dim=1
	#pragma HLS RESOURCE variable=s_l2_dwconv core=FIFO_SRL
	dwconv_3x3<
		L2_DW_NIO,
		L2_DW_NCH,
		BIT_ACTV, BIT_WT, BIT_CONV,
		ROW2 * COL2 * N_BATCH
	>(s_l2_im2co, s_l2_dwconv, L2_DW);

	data_stream<L2_DW_NACT * BIT_CONV> s_l2_dwactv_in("L2_dwactv_in");
	#pragma HLS STREAM variable=s_l2_dwactv_in depth=DEFAULT_DEPTH dim=1
	#pragma HLS RESOURCE variable=s_l2_dwactv_in core=FIFO_SRL
	reduceWidth<
		L2_DW_NIO * BIT_CONV,
		L2_DW_NACT * BIT_CONV,
		ROW2 * COL2 * L2_DW_NCH / L2_DW_NIO * N_BATCH
	>(s_l2_dwconv, s_l2_dwactv_in);

	data_stream<L2_DW_NACT * BIT_ACTV> s_l2_dwactv("L2_dwactv");
	#pragma HLS STREAM variable=s_l2_dwactv depth=DEFAULT_DEPTH dim=1
	#pragma HLS RESOURCE variable=s_l2_dwactv core=FIFO_SRL
	norm_actv<
		L2_DW_NACT, L2_DW_NCH,
		BIT_CONV,
		BIT_BIAS,
		BIT_MULT,
		BIT_NORM,
		BIT_ACTV,
		R_SHIFT,
		ROW2 * COL2 * N_BATCH
	>(s_l2_dwactv_in, s_l2_dwactv, L2_DB, L2_DM);

	data_stream<L2_PW_NIN * BIT_ACTV> s_l2_pwconv_in("L2_pwconv_in");
	#pragma HLS STREAM variable=s_l2_pwconv_in depth=L2_PW_NICH/L2_PW_NIN dim=1
	#pragma HLS RESOURCE variable=s_l2_pwconv_in core=FIFO_SRL
	// #pragma HLS RESOURCE variable=s_l2_pwconv_in core=FIFO_LUTRAM
	expandWidth<
		L2_DW_NACT * BIT_ACTV,
		L2_PW_NIN * BIT_ACTV,
		ROW2 * COL2 * L2_DW_NCH / L2_DW_NACT * N_BATCH
	>(s_l2_dwactv, s_l2_pwconv_in);

	data_stream<L2_PW_NOUT * BIT_CONV> s_l2_pwconv("L2_pwconv");
	#pragma HLS STREAM variable=s_l2_pwconv depth=DEFAULT_DEPTH dim=1
	#pragma HLS RESOURCE variable=s_l2_pwconv core=FIFO_SRL
	pwconv<
		L2_PW_NIN, L2_PW_NOUT,
		L2_PW_NICH, L2_PW_NOCH,
		BIT_ACTV, BIT_WT, BIT_CONV,
		ROW2 * COL2 * N_BATCH
	>(s_l2_pwconv_in, s_l2_pwconv, L2_PW);

	data_stream<L2_PW_NACT * BIT_CONV> s_l2_pwactv_in("L2_pwactv_in");
	#pragma HLS STREAM variable=s_l2_pwactv_in depth=DEFAULT_DEPTH dim=1
	#pragma HLS RESOURCE variable=s_l2_pwactv_in core=FIFO_SRL
	reduceWidth<
		L2_PW_NOUT * BIT_CONV,
		L2_PW_NACT * BIT_CONV,
		ROW2 * COL2 * L2_PW_NOCH / L2_PW_NOUT * N_BATCH
	>(s_l2_pwconv, s_l2_pwactv_in);

	data_stream<L2_PW_NACT * BIT_ACTV> s_l2_pwactv("L2_pwactv");
	#pragma HLS STREAM variable=s_l2_pwactv depth=DEFAULT_DEPTH dim=1
	#pragma HLS RESOURCE variable=s_l2_pwactv core=FIFO_SRL
	norm_actv<
		L2_PW_NACT, L2_PW_NOCH,
		BIT_CONV,
		BIT_BIAS,
		BIT_MULT,
		BIT_NORM,
		BIT_ACTV,
		R_SHIFT,
		ROW2 * COL2 * N_BATCH
	>(s_l2_pwactv_in, s_l2_pwactv, L2_PB, L2_PM);

	// bypass send
	// static_assert(L2_PW_NACT == BP_IO, "bypass");
	data_stream<L2_PW_NACT * BIT_ACTV> s_l2_pool_in("s_l2_pool_in");
	data_stream<L2_PW_NACT * BIT_ACTV> s_bptx_in("s_bypss_send_in");
	data_stream<BP_IO * BIT_ACTV> s_bptx("s_bypss_send");
	data_stream<BP_IO * BP_BIT> bp_fifo_0("bp_fifo_0");
	data_stream<BP_IO * BP_BIT> bp_fifo_1("bp_fifo_1");
	#pragma HLS STREAM variable=s_l2_pool_in depth=DEFAULT_DEPTH dim=1
	#pragma HLS STREAM variable=s_bptx_in depth=DEFAULT_DEPTH dim=1
	#pragma HLS STREAM variable=s_bptx depth=BP_STREAM_DEPTH dim=1
	#pragma HLS STREAM variable=bp_fifo_0 depth=BP_FIFO_0_DEPTH dim=1
	#pragma HLS STREAM variable=bp_fifo_1 depth=BP_FIFO_1_DEPTH dim=1
	#pragma HLS RESOURCE variable=s_l2_pool_in core=FIFO_SRL
	#pragma HLS RESOURCE variable=s_bptx_in core=FIFO_SRL
	#pragma HLS RESOURCE variable=s_bptx core=FIFO_SRL
	#pragma HLS RESOURCE variable=bp_fifo_0 core=FIFO_BRAM
	#pragma HLS RESOURCE variable=bp_fifo_1 core=FIFO_BRAM

	copy_stream<
		L2_PW_NACT * BIT_ACTV,
		ROW2 * COL2 * L2_PW_NOCH / L2_PW_NACT * N_BATCH
	>(s_l2_pwactv, s_l2_pool_in, s_bptx_in);

	expandWidth<
		L2_PW_NACT * BIT_ACTV,
		BP_IO * BIT_ACTV,
		ROW2 * COL2 * L2_PW_NOCH / L2_PW_NACT * N_BATCH
	>(s_bptx_in, s_bptx);

	bypass_send_reOrg(s_bptx, bp_fifo_0, bp_fifo_1);

	data_stream<L2_PW_NACT * BIT_ACTV> s_l2_pool("L2_pool");
	constexpr unsigned L2_POOL_DEPTH = (COL3 / 2 + 1) * L2_PW_NOCH / L2_PW_NACT;
	#pragma HLS STREAM variable=s_l2_pool depth=L2_POOL_DEPTH dim=1
	#pragma HLS RESOURCE variable=s_l2_pool core=FIFO_BRAM
	maxPool2x2<
		L2_PW_NACT,
		L2_PW_NOCH,
		BIT_ACTV,
		ROW2, COL2, N_BATCH
	>(s_l2_pool_in, s_l2_pool);

	// Bundle #4
	static_assert(L2_PW_NOCH == L3_DW_NCH, "Bundle 4");
	static_assert(L3_DW_NCH == L3_PW_NICH, "Bundle 4");

	data_stream<L3_DW_NIO * BIT_ACTV> s_l3_im2col_in("L3_im2col_in");
	#pragma HLS STREAM variable=s_l3_im2col_in depth=DEFAULT_DEPTH dim=1
	#pragma HLS RESOURCE variable=s_l3_im2col_in core=FIFO_SRL
	expandWidth<
		L2_PW_NACT * BIT_ACTV,
		L3_DW_NIO * BIT_ACTV,
		ROW3 * COL3 * L3_DW_NCH / L2_PW_NACT * N_BATCH
	>(s_l2_pool, s_l3_im2col_in);

	data_stream<L3_DW_NIO * BIT_ACTV> s_l3_im2co("L3_im2col");
	#pragma HLS STREAM variable=s_l3_im2co depth=DEFAULT_DEPTH dim=1
	#pragma HLS RESOURCE variable=s_l3_im2co core=FIFO_SRL
	im2col_3x3<
		L3_DW_NIO * BIT_ACTV,
		L3_DW_NCH * BIT_ACTV,
		ROW3, COL3, N_BATCH
	>(s_l3_im2col_in, s_l3_im2co);

	data_stream<L3_DW_NIO * BIT_CONV> s_l3_dwconv("L3_dwconv");
	#pragma HLS STREAM variable=s_l3_dwconv depth=DEFAULT_DEPTH dim=1
	#pragma HLS RESOURCE variable=s_l3_dwconv core=FIFO_SRL
	dwconv_3x3<
		L3_DW_NIO,
		L3_DW_NCH,
		BIT_ACTV, BIT_WT, BIT_CONV,
		ROW3 * COL3 * N_BATCH
	>(s_l3_im2co, s_l3_dwconv, L3_DW);

	data_stream<L3_DW_NACT * BIT_CONV> s_l3_dwactv_in("L3_dwactv_in");
	#pragma HLS STREAM variable=s_l3_dwactv_in depth=DEFAULT_DEPTH dim=1
	#pragma HLS RESOURCE variable=s_l3_dwactv_in core=FIFO_SRL
	reduceWidth<
		L3_DW_NIO * BIT_CONV,
		L3_DW_NACT * BIT_CONV,
		ROW3 * COL3 * L3_DW_NCH / L3_DW_NIO * N_BATCH
	>(s_l3_dwconv, s_l3_dwactv_in);

	data_stream<L3_DW_NACT * BIT_ACTV> s_l3_dwactv("L3_dwactv");
	#pragma HLS STREAM variable=s_l3_dwactv depth=DEFAULT_DEPTH dim=1
	#pragma HLS RESOURCE variable=s_l3_dwactv core=FIFO_SRL
	norm_actv<
		L3_DW_NACT, L3_DW_NCH,
		BIT_CONV,
		BIT_BIAS,
		BIT_MULT,
		BIT_NORM,
		BIT_ACTV,
		R_SHIFT,
		ROW3 * COL3 * N_BATCH
	>(s_l3_dwactv_in, s_l3_dwactv, L3_DB, L3_DM);

	data_stream<L3_PW_NIN * BIT_ACTV> s_l3_pwconv_in("L3_pwconv_in");
	#pragma HLS STREAM variable=s_l3_pwconv_in depth=L3_PW_NICH/L3_PW_NIN dim=1
	#pragma HLS RESOURCE variable=s_l3_pwconv_in core=FIFO_SRL
	// #pragma HLS RESOURCE variable=s_l3_pwconv_in core=FIFO_LUTRAM
	expandWidth<
		L3_DW_NACT * BIT_ACTV,
		L3_PW_NIN * BIT_ACTV,
		ROW3 * COL3 * L3_DW_NCH / L3_DW_NACT * N_BATCH
	>(s_l3_dwactv, s_l3_pwconv_in);

	data_stream<L3_PW_NOUT * BIT_CONV> s_l3_pwconv("L3_pwconv");
	#pragma HLS STREAM variable=s_l3_pwconv depth=DEFAULT_DEPTH dim=1
	#pragma HLS RESOURCE variable=s_l3_pwconv core=FIFO_SRL
	pwconv<
		L3_PW_NIN, L3_PW_NOUT,
		L3_PW_NICH, L3_PW_NOCH,
		BIT_ACTV, BIT_WT, BIT_CONV,
		ROW3 * COL3 * N_BATCH
	>(s_l3_pwconv_in, s_l3_pwconv, L3_PW);

	data_stream<L3_PW_NACT * BIT_CONV> s_l3_pwactv_in("L3_pwactv_in");
	#pragma HLS STREAM variable=s_l3_pwactv_in depth=DEFAULT_DEPTH dim=1
	#pragma HLS RESOURCE variable=s_l3_pwactv_in core=FIFO_SRL
	reduceWidth<
		L3_PW_NOUT * BIT_CONV,
		L3_PW_NACT * BIT_CONV,
		ROW3 * COL3 * L3_PW_NOCH / L3_PW_NOUT * N_BATCH
	>(s_l3_pwconv, s_l3_pwactv_in);

	data_stream<L3_PW_NACT * BIT_ACTV> s_l3_pwactv("L3_pwactv");
	#pragma HLS STREAM variable=s_l3_pwactv depth=DEFAULT_DEPTH dim=1
	#pragma HLS RESOURCE variable=s_l3_pwactv core=FIFO_SRL
	norm_actv<
		L3_PW_NACT, L3_PW_NOCH,
		BIT_CONV,
		BIT_BIAS,
		BIT_MULT,
		BIT_NORM,
		BIT_ACTV,
		R_SHIFT,
		ROW3 * COL3 * N_BATCH
	>(s_l3_pwactv_in, s_l3_pwactv, L3_PB, L3_PM);

	// Bundle 5
	static_assert(L3_PW_NOCH == L4_DW_NCH, "Bundle 5");
	static_assert(L4_DW_NCH == L4_PW_NICH, "Bundle 5");

	data_stream<L4_DW_NIO * BIT_ACTV> s_l4_im2col_in("L4_im2col_in");
	#pragma HLS STREAM variable=s_l4_im2col_in depth=DEFAULT_DEPTH dim=1
	#pragma HLS RESOURCE variable=s_l4_im2col_in core=FIFO_SRL
	expandWidth<
		L3_PW_NACT * BIT_ACTV,
		L4_DW_NIO * BIT_ACTV,
		ROW3 * COL3 * L4_DW_NCH / L3_PW_NACT * N_BATCH
	>(s_l3_pwactv, s_l4_im2col_in);

	data_stream<L4_DW_NIO * BIT_ACTV> s_l4_im2co("L4_im2col");
	#pragma HLS STREAM variable=s_l4_im2co depth=DEFAULT_DEPTH dim=1
	#pragma HLS RESOURCE variable=s_l4_im2co core=FIFO_SRL
	im2col_3x3<
		L4_DW_NIO * BIT_ACTV,
		L4_DW_NCH * BIT_ACTV,
		ROW3, COL3, N_BATCH
	>(s_l4_im2col_in, s_l4_im2co);

	data_stream<L4_DW_NIO * BIT_CONV> s_l4_dwconv("L4_dwconv");
	#pragma HLS STREAM variable=s_l4_dwconv depth=DEFAULT_DEPTH dim=1
	#pragma HLS RESOURCE variable=s_l4_dwconv core=FIFO_SRL
	dwconv_3x3<
		L4_DW_NIO,
		L4_DW_NCH,
		BIT_ACTV, BIT_WT, BIT_CONV,
		ROW3 * COL3 * N_BATCH
	>(s_l4_im2co, s_l4_dwconv, L4_DW);

	data_stream<L4_DW_NACT * BIT_CONV> s_l4_dwactv_in("L4_dwactv_in");
	#pragma HLS STREAM variable=s_l4_dwactv_in depth=DEFAULT_DEPTH dim=1
	#pragma HLS RESOURCE variable=s_l4_dwactv_in core=FIFO_SRL
	reduceWidth<
		L4_DW_NIO * BIT_CONV,
		L4_DW_NACT * BIT_CONV,
		ROW3 * COL3 * L4_DW_NCH / L4_DW_NIO * N_BATCH
	>(s_l4_dwconv, s_l4_dwactv_in);

	data_stream<L4_DW_NACT * BIT_ACTV> s_l4_dwactv("L4_dwactv");
	#pragma HLS STREAM variable=s_l4_dwactv depth=DEFAULT_DEPTH dim=1
	#pragma HLS RESOURCE variable=s_l4_dwactv core=FIFO_SRL
	norm_actv<
		L4_DW_NACT, L4_DW_NCH,
		BIT_CONV,
		BIT_BIAS,
		BIT_MULT,
		BIT_NORM,
		BIT_ACTV,
		R_SHIFT,
		ROW3 * COL3 * N_BATCH
	>(s_l4_dwactv_in, s_l4_dwactv, L4_DB, L4_DM);

	data_stream<L4_PW_NIN * BIT_ACTV> s_l4_pwconv_in("L4_pwconv_in");
	#pragma HLS STREAM variable=s_l4_pwconv_in depth=L4_PW_NICH/L4_PW_NIN dim=1
	#pragma HLS RESOURCE variable=s_l4_pwconv_in core=FIFO_SRL
	// #pragma HLS RESOURCE variable=s_l4_pwconv_in core=FIFO_LUTRAM
	expandWidth<
		L4_DW_NACT * BIT_ACTV,
		L4_PW_NIN * BIT_ACTV,
		ROW3 * COL3 * L4_DW_NCH / L4_DW_NACT * N_BATCH
	>(s_l4_dwactv, s_l4_pwconv_in);

	data_stream<L4_PW_NOUT * BIT_CONV> s_l4_pwconv("L4_pwconv");
	#pragma HLS STREAM variable=s_l4_pwconv depth=DEFAULT_DEPTH dim=1
	#pragma HLS RESOURCE variable=s_l4_pwconv core=FIFO_SRL
	pwconv<
		L4_PW_NIN, L4_PW_NOUT,
		L4_PW_NICH, L4_PW_NOCH,
		BIT_ACTV, BIT_WT, BIT_CONV,
		ROW3 * COL3 * N_BATCH
	>(s_l4_pwconv_in, s_l4_pwconv, L4_PW);

	data_stream<L4_PW_NACT * BIT_CONV> s_l4_pwactv_in("L4_pwactv_in");
	#pragma HLS STREAM variable=s_l4_pwactv_in depth=DEFAULT_DEPTH dim=1
	#pragma HLS RESOURCE variable=s_l4_pwactv_in core=FIFO_SRL
	reduceWidth<
		L4_PW_NOUT * BIT_CONV,
		L4_PW_NACT * BIT_CONV,
		ROW3 * COL3 * L4_PW_NOCH / L4_PW_NOUT * N_BATCH
	>(s_l4_pwconv, s_l4_pwactv_in);

	data_stream<L4_PW_NACT * BIT_ACTV> s_l4_pwactv("L4_pwactv");
	#pragma HLS STREAM variable=s_l4_pwactv depth=DEFAULT_DEPTH dim=1
	#pragma HLS RESOURCE variable=s_l4_pwactv core=FIFO_SRL
	norm_actv<
		L4_PW_NACT, L4_PW_NOCH,
		BIT_CONV,
		BIT_BIAS,
		BIT_MULT,
		BIT_NORM,
		BIT_ACTV,
		R_SHIFT,
		ROW3 * COL3 * N_BATCH
	>(s_l4_pwactv_in, s_l4_pwactv, L4_PB, L4_PM);

	// bypass recv
	data_stream<BP_IO * BIT_ACTV> s_bprx("s_bypass_recv");
	#pragma HLS STREAM variable=s_bprx depth=BP_STREAM_DEPTH dim=1
	#pragma HLS RESOURCE variable=s_bprx core=FIFO_SRL
	bypass_recv(bp_fifo_0, bp_fifo_1, s_bprx);

	data_stream<BP_IO * BIT_ACTV> s_comb_in("s_comb_in1");
	// BP_CH / (L4_PW_NICH/L4_PW_NIN) * L4_PW_NOUT == 256
	#pragma HLS STREAM variable=s_comb_in depth=192 dim=1
	#pragma HLS RESOURCE variable=s_comb_in core=FIFO_LUTRAM
	expandWidth<
		L4_PW_NACT * BIT_ACTV,
		BP_IO * BIT_ACTV,
		ROW3 * COL3 * L4_PW_NOCH / L4_PW_NACT * N_BATCH
	>(s_l4_pwactv, s_comb_in);

	data_stream<BP_IO * BIT_ACTV> s_reorg("s_reorg");
	#pragma HLS STREAM variable=s_reorg depth=DEFAULT_DEPTH dim=1
	#pragma HLS RESOURCE variable=s_reorg core=FIFO_SRL
	comb_stream<
		BP_IO * BIT_ACTV,
		BP_CH / BP_IO,
		L4_PW_NOCH / BP_IO,
		ROW3 * COL3 * N_BATCH
	>(s_bprx, s_comb_in, s_reorg);

	// Bundle #6
	static_assert(BP_CH + L4_PW_NOCH == L5_DW_NCH, "Bundle 6");
	static_assert(L5_DW_NCH == L5_PW_NICH, "Bundle 6");

	data_stream<L5_DW_NIO * BIT_ACTV> s_l5_im2col_in("L5_im2col_in");
	#pragma HLS STREAM variable=s_l5_im2col_in depth=DEFAULT_DEPTH dim=1
	#pragma HLS RESOURCE variable=s_l5_im2col_in core=FIFO_SRL
	expandWidth<
		BP_IO * BIT_ACTV,
		L5_DW_NIO * BIT_ACTV,
		ROW3 * COL3 * L5_DW_NCH / BP_IO * N_BATCH
	>(s_reorg, s_l5_im2col_in);

	data_stream<L5_DW_NIO * BIT_ACTV> s_l5_im2co("L5_im2col");
	#pragma HLS STREAM variable=s_l5_im2co depth=DEFAULT_DEPTH dim=1
	#pragma HLS RESOURCE variable=s_l5_im2co core=FIFO_SRL
	im2col_3x3<
		L5_DW_NIO * BIT_ACTV,
		L5_DW_NCH * BIT_ACTV,
		ROW3, COL3, N_BATCH
	>(s_l5_im2col_in, s_l5_im2co);

	data_stream<L5_DW_NIO * BIT_CONV> s_l5_dwconv("L5_dwconv");
	#pragma HLS STREAM variable=s_l5_dwconv depth=DEFAULT_DEPTH dim=1
	#pragma HLS RESOURCE variable=s_l5_dwconv core=FIFO_SRL
	dwconv_3x3<
		L5_DW_NIO,
		L5_DW_NCH,
		BIT_ACTV, BIT_WT, BIT_CONV,
		ROW3 * COL3 * N_BATCH
	>(s_l5_im2co, s_l5_dwconv, L5_DW);

	data_stream<L5_DW_NACT * BIT_CONV> s_l5_dwactv_in("L5_dwactv_in");
	#pragma HLS STREAM variable=s_l5_dwactv_in depth=DEFAULT_DEPTH dim=1
	#pragma HLS RESOURCE variable=s_l5_dwactv_in core=FIFO_SRL
	reduceWidth<
		L5_DW_NIO * BIT_CONV,
		L5_DW_NACT * BIT_CONV,
		ROW3 * COL3 * L5_DW_NCH / L5_DW_NIO * N_BATCH
	>(s_l5_dwconv, s_l5_dwactv_in);

	data_stream<L5_DW_NACT * BIT_ACTV> s_l5_dwactv("L5_dwactv");
	#pragma HLS STREAM variable=s_l5_dwactv depth=DEFAULT_DEPTH dim=1
	#pragma HLS RESOURCE variable=s_l5_dwactv core=FIFO_SRL
	norm_actv<
		L5_DW_NACT, L5_DW_NCH,
		BIT_CONV,
		BIT_BIAS,
		BIT_MULT,
		BIT_NORM,
		BIT_ACTV,
		R_SHIFT,
		ROW3 * COL3 * N_BATCH
	>(s_l5_dwactv_in, s_l5_dwactv, L5_DB, L5_DM);

	data_stream<L5_PW_NIN * BIT_ACTV> s_l5_pwconv_in("L5_pwconv_in");
	#pragma HLS STREAM variable=s_l5_pwconv_in depth=L5_PW_NICH/L5_PW_NIN dim=1
	#pragma HLS RESOURCE variable=s_l5_pwconv_in core=FIFO_SRL
	// #pragma HLS RESOURCE variable=s_l5_pwconv_in core=FIFO_LUTRAM
	expandWidth<
		L5_DW_NACT * BIT_ACTV,
		L5_PW_NIN * BIT_ACTV,
		ROW3 * COL3 * L5_DW_NCH / L5_DW_NACT * N_BATCH
	>(s_l5_dwactv, s_l5_pwconv_in);

	data_stream<L5_PW_NOUT * BIT_CONV> s_l5_pwconv("L5_pwconv");
	#pragma HLS STREAM variable=s_l5_pwconv depth=DEFAULT_DEPTH dim=1
	#pragma HLS RESOURCE variable=s_l5_pwconv core=FIFO_SRL
	pwconv<
		L5_PW_NIN, L5_PW_NOUT,
		L5_PW_NICH, L5_PW_NOCH,
		BIT_ACTV, BIT_WT, BIT_CONV,
		ROW3 * COL3 * N_BATCH
	>(s_l5_pwconv_in, s_l5_pwconv, L5_PW);

	data_stream<L5_PW_NACT * BIT_CONV> s_l5_pwactv_in("L5_pwactv_in");
	#pragma HLS STREAM variable=s_l5_pwactv_in depth=DEFAULT_DEPTH dim=1
	#pragma HLS RESOURCE variable=s_l5_pwactv_in core=FIFO_SRL
	reduceWidth<
		L5_PW_NOUT * BIT_CONV,
		L5_PW_NACT * BIT_CONV,
		ROW3 * COL3 * L5_PW_NOCH / L5_PW_NOUT * N_BATCH
	>(s_l5_pwconv, s_l5_pwactv_in);

	data_stream<L5_PW_NACT * BIT_ACTV> s_l5_pwactv("L5_pwactv");
	// input of L6_pwconv for PE128
	#pragma HLS STREAM variable=s_l5_pwactv depth=DEFAULT_DEPTH dim=1
	// #pragma HLS STREAM variable=s_l5_pwactv depth=L6_PW_NICH/L6_PW_NIN dim=1
	#pragma HLS RESOURCE variable=s_l5_pwactv core=FIFO_SRL
	// #pragma HLS RESOURCE variable=s_l5_pwactv core=FIFO_LUTRAM
	norm_actv<
		L5_PW_NACT, L5_PW_NOCH,
		BIT_CONV,
		BIT_BIAS,
		BIT_MULT,
		BIT_NORM,
		BIT_ACTV,
		R_SHIFT,
		ROW3 * COL3 * N_BATCH
	>(s_l5_pwactv_in, s_l5_pwactv, L5_PB, L5_PM);

	// Bundle 7
	static_assert(L5_PW_NOCH == L6_PW_NICH, "Bundle 7");
	// static_assert(L5_PW_NACT == L6_PW_NIN, "Bundle 7");
	static_assert(L6_PW_NOUT == N_OUT, "Bundle 7");

	data_stream<L6_PW_NIN * BIT_ACTV> s_l6_pwconv_in("L6_pwconv_in");
	#pragma HLS STREAM variable=s_l6_pwconv_in depth=L6_PW_NICH/L6_PW_NIN dim=1
	#pragma HLS RESOURCE variable=s_l6_pwconv_in core=FIFO_SRL
	expandWidth<
		L5_PW_NACT * BIT_ACTV,
		L6_PW_NIN * BIT_ACTV,
		ROW3 * COL3 * L5_PW_NOCH / L5_PW_NACT * N_BATCH
	>(s_l5_pwactv, s_l6_pwconv_in);

	data_stream<L6_PW_NOUT * BIT_CONV> s_l6_pwconv("L6_pwconv");
	#pragma HLS STREAM variable=s_l6_pwconv depth=DEFAULT_DEPTH dim=1
	#pragma HLS RESOURCE variable=s_l6_pwconv core=FIFO_SRL
	pwconv_single<
		L6_PW_NIN, L6_PW_NOUT,
		L6_PW_NICH, L6_PW_NOCH,
		BIT_ACTV, BIT_WT, BIT_CONV,
		ROW3 * COL3 * N_BATCH
	>(s_l6_pwconv_in, s_l6_pwconv, L6_PW);

	// Find Max
	static_assert(L6_PW_NOUT == 1, "Find Max");
	data_stream<BIT_CONV> s_findmax("s_findmax");
	#pragma HLS STREAM variable=s_findmax depth=DEFAULT_DEPTH dim=1
	#pragma HLS RESOURCE variable=s_findmax core=FIFO_SRL
	findMax(s_l6_pwconv, s_findmax);

	add_last<
		BIT_CONV,
		FINDMAX_NLINE * N_BATCH
	>(s_findmax, out);

	return;
}
