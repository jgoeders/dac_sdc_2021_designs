#ifndef __CONV2D_L1_HPP__
#define __CONV2D_L1_HPP__

#include <ap_int.h>
#include <hls_stream.h>
using namespace hls;

// #include "debug.hpp"
#include "function.h"
#include "matrix_vector_unit.h"
#include "sliding_window_unit.h"
#include "stream_tools.h"

template <unsigned IN_W, unsigned IN_BIT>
void stream_in_row_l0(stream<ap_uint<3 * IN_BIT>> &in,
                      ap_uint<3 * IN_BIT> row_buffer[4][IN_W + 2],
                      bool skip_flag, ap_uint<2> rowBufferIdx) {

  if (skip_flag)
    return;

  for (unsigned w = 0; w < IN_W + 2; w++) {
#pragma HLS pipeline
    ap_uint<3 * IN_BIT> data;
    if (w != 0 && w != IN_W + 1) {
      data = in.read();
    } else {
      data = 0;
    }
    row_buffer[rowBufferIdx][w] = data;
  }
}

template <unsigned IN_H, unsigned IN_W, unsigned IN_BIT, unsigned OUTPENUM>
void stream_out_data_l0(stream<ap_uint<3 * IN_BIT * 3>> &out,
                        ap_uint<3 * IN_BIT> row_buffer[4][IN_W + 2],
                        bool skip_flag, ap_int<12> outRowIdx,
                        ap_uint<2> centerRowBufferIdx) {
#pragma HLS array_partition variable = row_buffer dim = 1 complete

  if (skip_flag)
    return;

  for (unsigned peIdx = 0; peIdx < OUTPENUM; peIdx++)
    for (unsigned c = 0; c < IN_W; c++)
      for (unsigned kc = 0; kc < 3; kc++) {
#pragma HLS pipeline
        ap_uint<3 * IN_BIT> data[4];
#pragma HLS array_partition variable = data dim = 1 complete
        for (unsigned i = 0; i < 4; i++) {
          data[i] = row_buffer[i][c + kc];
        }
        ap_uint<2> row_sel0, row_sel1, row_sel2;
        row_sel0 = centerRowBufferIdx - 1;
        row_sel1 = centerRowBufferIdx;
        row_sel2 = centerRowBufferIdx + 1;
        ap_uint<3 * IN_BIT> data0, data1, data2;

        if (outRowIdx - 1 < 0)
          data0 = 0;
        else
          data0 = data[row_sel0];
        data1 = data[row_sel1];
        if (outRowIdx + 1 == IN_H)
          data2 = 0;
        else
          data2 = data[row_sel2];
        out.write((data2, data1, data0));
      }
}

template <unsigned K, unsigned IN_H, unsigned IN_W, unsigned IN_CH,
          unsigned IN_BIT, unsigned OUTPENUM>
void conv3padding_l0(stream<ap_uint<3 * IN_BIT>> &in,
                     stream<ap_uint<3 * IN_BIT * 3>> &out,
                     const unsigned reps = 1) {
  static_assert(K == 3, "K!=3");
  ap_uint<IN_CH * IN_BIT> row_buffer[4][IN_W + 2];
#pragma HLS ARRAY_PARTITION variable = row_buffer dim = 1 complete
#pragma HLS RESOURCE variable = row_buffer core = RAM_S2P_BRAM
  ap_uint<2> storeBufferIdx = 0;
  ap_uint<2> loadBufferIdx = -2;
  ap_int<10> rowIdx = -2;

  for (unsigned rep = 0; rep < reps * IN_H + 2; rep++) {
#pragma HLS dependence intra false variable = row_buffer
    stream_in_row_l0<IN_W, IN_BIT>(in, row_buffer, (rep >= reps * IN_H),
                                   storeBufferIdx);
    stream_out_data_l0<IN_H, IN_W, IN_BIT, OUTPENUM>(out, row_buffer, (rep < 2),
                                                     rowIdx, loadBufferIdx);
    loadBufferIdx++;
    storeBufferIdx++;
    if (rowIdx == IN_H - 1) {
      rowIdx = 0;
    } else {
      rowIdx++;
    }
  }
}

template <unsigned IN_BIT, unsigned W_BIT, unsigned PROD_BIT>
void simd_mac9_DSP2(ap_uint<IN_BIT> invec[9], ap_int<W_BIT> w0vec[9],
                    ap_int<W_BIT> w1vec[9], ap_int<PROD_BIT> &out0,
                    ap_int<PROD_BIT> &out1) {
// #pragma HLS pipeline II = 1
#pragma HLS array_partition variable = invec
#pragma HLS array_partition variable = w1vec
#pragma HLS array_partition variable = w0vec

  ap_int<PROD_BIT * 2> acc = 0;

  // cout << "ivec" << endl;
  // for (int i = 0; i < 9; i++) {
  //   cout << invec[i] << endl;
  // }
  // getchar();

  // cout << "wvec0" << endl;
  // for (int i = 0; i < 9; i++) {
  //   cout << w0vec[i] << endl;
  // }
  // getchar();

  // cout << "wvec1" << endl;
  // for (int i = 0; i < 9; i++) {
  //   cout << w1vec[i] << endl;
  // }
  // getchar();
  // cout << "rst,m" << endl;
  for (int i = 0; i < 9; i++) {
    ap_int<PROD_BIT + W_BIT> rst = w1vec[i] * (1 << PROD_BIT) + w0vec[i];
    ap_int<PROD_BIT * 2> m = invec[i] * rst;

    acc += m;
    // cout << rst.to_string(10) << "," << invec[i].to_string(10) << ","
    //      << m.to_string(10) << endl;
  }
  // getchar();

  out0 = acc(PROD_BIT - 1, 0);
  out1 = acc(PROD_BIT * 2 - 1, PROD_BIT) + acc[PROD_BIT - 1];
}

template <unsigned IN_BIT>
void loadInReg9(ap_uint<IN_BIT * 9> inData, ap_uint<IN_BIT> ivec[9]) {
#pragma HLS pipeline II = 1
#pragma HLS inline off
#pragma HLS ARRAY_PARTITION variable = ivec complete dim = 1

  for (unsigned s = 0; s < 9; s++) {
    ivec[s] = inData((s + 1) * IN_BIT - 1, s * IN_BIT);
  }
}

template <unsigned OUT_ROW, unsigned OUT_COL, unsigned OUT_CH, unsigned PE,
          unsigned IN_BIT, unsigned W_BIT, unsigned M_BIT,
          // unsigned BIAS_BIT,
          // unsigned INC_BIT,
          unsigned OUT_BIT>
void convDSPOpt_l0(stream<ap_uint<IN_BIT * 9>> &in,
                   const ap_uint<3 * W_BIT> weights[PE][3][3 * (OUT_CH / PE)],
                   stream<ap_uint<M_BIT * PE>> &out, const unsigned reps = 1) {
#pragma HLS ARRAY_PARTITION variable = weights complete dim = 1
#pragma HLS ARRAY_PARTITION variable = weights complete dim = 2

  //   ap_int<PROD_BIT * 2> m_hi[INFOLD][PE][SIMD];
  // #pragma HLS ARRAY_PARTITION variable = m_hi complete dim = 2
  //   ap_int<PROD_BIT * 2> m_lo[PE][SIMD];
  // #pragma HLS ARRAY_PARTITION variable = m_lo complete dim = 1

  const unsigned PROD_BIT = IN_BIT + W_BIT + 4;

  ap_int<M_BIT> outPartialArr[PE];
#pragma HLS ARRAY_PARTITION variable = outPartialArr complete dim = 1

  for (unsigned int h = 0; h < OUT_ROW * reps; h++) {
    for (unsigned peIdx = 0; peIdx < OUT_CH / PE; peIdx++)
      for (unsigned int w = 0; w < OUT_COL; w++) {
        for (unsigned int kc = 0; kc < 3; kc++) {
#pragma HLS pipeline II = 1

          ap_uint<IN_BIT> ivec[9];
#pragma HLS ARRAY_PARTITION variable = ivec complete dim = 1
          ap_int<W_BIT> wvec[PE][9];
#pragma HLS ARRAY_PARTITION variable = wvec complete dim = 1
#pragma HLS ARRAY_PARTITION variable = wvec complete dim = 2

          ap_uint<IN_BIT * 9> inData;
          in >> inData;
          loadInReg9<IN_BIT>(inData, ivec);
          for (int i = 0; i < PE; i++) {
            for (int s = 0; s < 9; s++) {
              wvec[i][s] = weights[i][s / 3][peIdx * 3 + kc](
                  (s % 3 + 1) * W_BIT - 1, s % 3 * W_BIT);
            }
          }

          // cout << "w,kc:" << w << "," << kc << endl;

          for (int p = 0; p < PE; p += 2) {
            ap_int<PROD_BIT> outPartial0;
            ap_int<PROD_BIT> outPartial1;
            simd_mac9_DSP2<IN_BIT, W_BIT, PROD_BIT>(ivec, wvec[p], wvec[p + 1],
                                                    outPartial0, outPartial1);

            // cout << outPartial0.to_string(10) << endl;
            // cout << outPartial1.to_string(10) << endl;

            if (kc == 0) {
              outPartialArr[p] = outPartial0;
              outPartialArr[p + 1] = outPartial1;
            } else {
              outPartialArr[p] += outPartial0;
              outPartialArr[p + 1] += outPartial1;
            }
          }
          // getchar();
          ap_uint<M_BIT * PE> odata;
          if (kc == 2) {
            // cout << outPartialArr[0].to_string(16) << endl;

            for (int i = 0; i < PE; i++) {
              // cout << outPartialArr[0].to_string(16) << ",";
              odata((i + 1) * M_BIT - 1, i * M_BIT) = outPartialArr[i];
            }

            out.write(odata);
          }
        }
      }
  }
}

template <unsigned OUT_ROW, unsigned OUT_COL, unsigned OUT_CH, unsigned M_BIT,
          unsigned OUT_BIT, unsigned INC_BIT, unsigned BIAS_BIT,
          unsigned IN_BIT, unsigned W_BIT, unsigned L_SHIFT, unsigned PE>
void streamBnRelu_l0(stream<ap_uint<PE * M_BIT>> &in,
                     const ap_int<INC_BIT> inc[PE][OUT_CH / PE],
                     const ap_int<BIAS_BIT> bias[PE][OUT_CH / PE],
                     stream<ap_uint<PE * OUT_BIT * 2>> &out,
                     const unsigned rep = 1) {
#pragma HLS ARRAY_PARTITION variable = inc complete dim = 1
#pragma HLS ARRAY_PARTITION variable = bias complete dim = 1
  for (unsigned r = 0; r < OUT_ROW * rep; r++)
    for (unsigned peIdx = 0; peIdx < OUT_CH / PE; peIdx++)
      for (unsigned w = 0; w < OUT_COL; w += 2) {

#pragma HLS pipeline II = 4
        ap_uint<M_BIT * PE> data;
        ap_uint<OUT_BIT * PE> data0, data1;
        ap_int<M_BIT> invec[PE];
#pragma HLS array_partition variable = invec dim = 1 complete
        data = in.read();
        for (int i = 0; i < PE; i++) {
          invec[i] = data((i + 1) * M_BIT - 1, i * M_BIT);
        }
        for (int i = 0; i < PE; i++) {
          data0((i + 1) * OUT_BIT - 1, i * OUT_BIT) =
              bn_qurelu_fixed<M_BIT, OUT_BIT, INC_BIT, BIAS_BIT, IN_BIT, W_BIT,
                              L_SHIFT>(invec[i], inc[i][peIdx], bias[i][peIdx]);
        }

        data = in.read();
        for (int i = 0; i < PE; i++) {
          invec[i] = data((i + 1) * M_BIT - 1, i * M_BIT);
        }
        for (int i = 0; i < PE; i++) {
          data1((i + 1) * OUT_BIT - 1, i * OUT_BIT) =
              bn_qurelu_fixed<M_BIT, OUT_BIT, INC_BIT, BIAS_BIT, IN_BIT, W_BIT,
                              L_SHIFT>(invec[i], inc[i][peIdx], bias[i][peIdx]);
        }
        out.write((data1, data0));
      }
}

template <unsigned IN_ROW, unsigned IN_COL, unsigned IN_CH, unsigned IN_BIT,

          unsigned OUT_CH,
          unsigned OUT_BIT, // 量化激活后的位宽

          unsigned W_BIT, unsigned M_BIT, unsigned INC_BIT, unsigned BIAS_BIT,

          unsigned SIMD, unsigned CASCADE, unsigned IN_PE, unsigned PE,
          unsigned L_SHIFT>
void conv3x3_l0_bn_act_DSPopt(
    stream<ap_uint<IN_BIT * IN_CH>> &in,
    const ap_uint<IN_CH * W_BIT> weights[PE][3][3 * (OUT_CH / PE)],
    const ap_int<INC_BIT> inc[PE][OUT_CH / PE],
    const ap_int<BIAS_BIT> bias[PE][OUT_CH / PE],

    stream<ap_uint<OUT_BIT * PE * 2>> &out, const unsigned reps = 1) {
#pragma HLS DATAFLOW

  // 暂时认为输入 输出维度不变
  const unsigned OUT_ROW = IN_ROW;
  const unsigned OUT_COL = IN_COL;

  stream<ap_uint<SIMD * IN_BIT * 3>> padding_out("pad_l0_out");
  conv3padding_l0<3, IN_ROW, IN_COL, IN_CH, IN_BIT, OUT_CH / PE>(
      in, padding_out, reps);

  stream<ap_uint<M_BIT * PE>> conv_l0_out("conv_l0_out");
  convDSPOpt_l0<OUT_ROW, OUT_COL, OUT_CH, PE, IN_BIT, W_BIT, M_BIT, OUT_BIT>(
      padding_out, weights, conv_l0_out, reps);
  streamBnRelu_l0<OUT_ROW, OUT_COL, OUT_CH, M_BIT, OUT_BIT, INC_BIT, BIAS_BIT,
                  L_SHIFT, IN_BIT, W_BIT, PE>(conv_l0_out, inc, bias, out,
                                              reps);
}

#endif
