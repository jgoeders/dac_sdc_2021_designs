#ifndef __CONV2D_DSPOPT_HPP__
#define __CONV2D_DSPOPT_HPP__

#include <ap_int.h>
#include <hls_stream.h>
using namespace hls;

// #include "debug.hpp"
#include "function.h"
#include "matrix_vector_unit.h"
#include "sliding_window_unit.h"
#include "stream_tools.h"

#define CEILDIV(x, y) (((x) + (y)-1) / (y))

template <unsigned IN_W, unsigned IN_CH, unsigned IN_BIT, unsigned IN_PE,
          unsigned SIMD>
void stream_in_row(
    stream<ap_uint<IN_PE * IN_BIT * 2>> &in,
    ap_uint<IN_PE * IN_BIT * 2> row_buffer[SIMD / IN_PE][4]
                                          [(IN_W / 2 + 1) * IN_CH / SIMD],
    bool skip_flag, ap_uint<2> rowBufferIdx) {
#pragma HLS inline off
  if (skip_flag)
    return;
  ap_uint<IN_PE *IN_BIT> reg = 0;

  for (unsigned peIdx = 0; peIdx < IN_CH / IN_PE; peIdx++)
    for (unsigned w = 0; w < IN_W / 2 + 1; w++) {
#pragma HLS pipeline
      ap_uint<IN_PE * IN_BIT * 2> data;
      ap_uint<IN_PE * IN_BIT> data0, data1;
      if (w != (IN_W / 2)) {
        (data1, data0) = in.read();
      } else {
        data1 = 0;
        data0 = 0;
      }
      data = (data0, reg);
      reg = data1;

      row_buffer[peIdx % (SIMD / IN_PE)][rowBufferIdx]
                [w * IN_CH / SIMD + peIdx / (SIMD / IN_PE)] = data;
    }
}

template <unsigned K, unsigned IN_H, unsigned IN_W, unsigned IN_CH,
          unsigned IN_BIT, unsigned IN_PE, unsigned SIMD, unsigned OUTPENUM>
void stream_out_data(
    stream<ap_uint<SIMD * IN_BIT * 2>> &out,
    ap_uint<IN_PE * IN_BIT * 2> row_buffer[SIMD / IN_PE][4]
                                          [(IN_W / 2 + 1) * IN_CH / SIMD],
    bool skip_flag, ap_int<12> outRowIdx, ap_uint<2> startRowBufferIdx) {
#pragma HLS array_partition variable = row_buffer dim = 1 complete

  const unsigned IN_PE_BIT = IN_PE * IN_BIT;
  const unsigned SIMDNUM = IN_CH / SIMD;
  const unsigned WLEN = IN_W / 2 + 1;
  if (skip_flag)
    return;

  ap_uint<8> infoldIdx = 0;
  ap_uint<8> w = 0;

  for (unsigned peIdx = 0; peIdx < OUTPENUM; peIdx++) {
    for (unsigned cycle = 0; cycle < WLEN * K * SIMDNUM; cycle++) {
      // for (unsigned w = 0; w < WLEN; w++) {
      //   for (unsigned wr = 0; wr < K; wr++) {
      //     for (unsigned simdIdx = 0; simdIdx < SIMDNUM; simdIdx++) {
      ap_uint<2> wr = infoldIdx / SIMDNUM;
      ap_uint<4> simdIdx = infoldIdx % SIMDNUM;
#pragma HLS pipeline
      ap_uint<SIMD * IN_BIT> data0;
      ap_uint<SIMD * IN_BIT> data1;
      ap_uint<IN_PE * IN_BIT * 2> buffer_data[SIMD / IN_PE];
#pragma HLS array_partition variable = buff_data complete
      ap_uint<2> rowBufferIdx = startRowBufferIdx + wr;
      for (unsigned i = 0; i < SIMD / IN_PE; i++) {
#pragma HLS unroll
        buffer_data[i] = row_buffer[i][rowBufferIdx][w * SIMDNUM + simdIdx];
      }

      if (outRowIdx - K / 2 + wr < 0 || outRowIdx - K / 2 + wr >= IN_H) {
        data0 = 0;
        data1 = 0;
      } else {
        for (unsigned i = 0; i < SIMD / IN_PE; i++) {
          data0((i + 1) * IN_PE_BIT - 1, i * IN_PE_BIT) =
              buffer_data[i](IN_PE_BIT - 1, 0);
          data1((i + 1) * IN_PE_BIT - 1, i * IN_PE_BIT) =
              buffer_data[i](IN_PE_BIT * 2 - 1, IN_PE_BIT);
        }
      }
      out.write((data1, data0));

      if (cycle == WLEN * K * SIMDNUM - 1) {
        w = 0;
      } else if (infoldIdx == K * SIMDNUM - 1) {
        w++;
      }

      if (infoldIdx == K * SIMDNUM - 1) {
        infoldIdx = 0;
      } else {
        infoldIdx++;
      }
    }
  }
}

template <unsigned K, unsigned IN_H, unsigned IN_W, unsigned IN_CH,
          unsigned IN_BIT, unsigned IN_PE, unsigned SIMD, unsigned OUTPENUM>
void conv3padding(stream<ap_uint<IN_PE * IN_BIT * 2>> &in,
                  stream<ap_uint<SIMD * IN_BIT * 2>> &out,
                  const unsigned reps = 1) {
  static_assert(SIMD % IN_PE == 0, "SIMD %IN_PE !=0");
  static_assert(K == 3, "K!=3");

  ap_uint<IN_PE * IN_BIT * 2> row_buffer[SIMD / IN_PE][4]
                                        [(IN_W / 2 + 1) * IN_CH / SIMD];
#pragma HLS ARRAY_PARTITION variable = row_buffer dim = 1 complete
#pragma HLS RESOURCE variable = row_buffer core = RAM_S2P_BRAM
  ap_uint<8> inh = 0;
  ap_uint<8> outh = 0;

  ap_uint<2> storeBufferIdx = 0;
  ap_uint<2> loadBufferIdx = 1;
  ap_int<10> rowIdx = -2;

  for (unsigned rep = 0; rep < reps * IN_H + 2; rep++) {
#pragma HLS dependence intra false variable = row_buffer
    stream_in_row<IN_W, IN_CH, IN_BIT, IN_PE, SIMD>(
        in, row_buffer, (rep >= reps * IN_H), storeBufferIdx);
    stream_out_data<K, IN_H, IN_W, IN_CH, IN_BIT, IN_PE, SIMD, OUTPENUM>(
        out, row_buffer, (rep < 2), rowIdx, loadBufferIdx);
    loadBufferIdx++;
    storeBufferIdx++;

    if (rowIdx == IN_H - 1) {
      rowIdx = 0;
    } else {
      rowIdx++;
    }
  }
}

template <unsigned OUT_ROW, unsigned OUT_COL, unsigned OUT_CH, unsigned M_BIT,
          unsigned OUT_BIT, unsigned INC_BIT, unsigned BIAS_BIT,
          unsigned IN_BIT, unsigned W_BIT, unsigned L_SHIFT, unsigned PE>
void streamBnRelu(stream<ap_uint<PE * M_BIT * 2>> &in,
                  const ap_int<INC_BIT> inc[PE][OUT_CH / PE],
                  const ap_int<BIAS_BIT> bias[PE][OUT_CH / PE],
                  stream<ap_uint<PE * OUT_BIT * 2>> &out,
                  const unsigned rep = 1) {
#pragma HLS ARRAY_PARTITION variable = inc complete dim = 1
#pragma HLS ARRAY_PARTITION variable = bias complete dim = 1
  for (int r = 0; r < OUT_ROW * rep; r++)
    for (int peIdx = 0; peIdx < OUT_CH / PE; peIdx++)
      for (int w = 0; w < OUT_COL; w += 2) {

#pragma HLS pipeline II = 2
        ap_uint<M_BIT * PE * 2> data;
        ap_uint<OUT_BIT * PE * 2> data0, data1;
        ap_int<M_BIT> invec[PE];
#pragma HLS array_partition variable = invec dim = 1 complete
        data = in.read();
        for (int i = 0; i < PE * 2; i++) {
          invec[i] = data((i + 1) * M_BIT - 1, i * M_BIT);
        }
        for (int i = 0; i < PE * 2; i++) {
          data0((i + 1) * OUT_BIT - 1, i * OUT_BIT) =
              bn_qurelu_fixed<M_BIT, OUT_BIT, INC_BIT, BIAS_BIT, IN_BIT, W_BIT,
                              L_SHIFT>(invec[i], inc[i % PE][peIdx],
                                       bias[i % PE][peIdx]);
        }
        out.write(data0);
      }
}

template <unsigned IN_BIT, unsigned SIMD, unsigned PROD_BIT>
void pack_input_data(ap_uint<IN_BIT * SIMD> A, ap_uint<IN_BIT * SIMD> B,
                     ap_uint<PROD_BIT + IN_BIT> ipack[SIMD]) {
#pragma HLS array_partition variable = ipack

  for (int i = 0; i < SIMD; i++) {
    ipack[i] =
        (A(i * IN_BIT + IN_BIT - 1, i * IN_BIT), (ap_uint<PROD_BIT - IN_BIT>)0,
         B(i * IN_BIT + IN_BIT - 1, i * IN_BIT));
  }
}

template <unsigned W_BIT, unsigned SIMD, unsigned PROD_BIT>
void pack_weight_data(ap_uint<W_BIT * SIMD> w2, ap_uint<W_BIT * SIMD> w1,
                      ap_uint<W_BIT * SIMD> w0,
                      ap_int<PROD_BIT * 2 + W_BIT> wpack[SIMD]) {
#pragma HLS array_partition variable = wpack

  for (int i = 0; i < SIMD; i++) {
    ap_int<W_BIT> w2_seg = w2(i * W_BIT + W_BIT - 1, i * W_BIT);
    ap_int<W_BIT> w1_seg = w1(i * W_BIT + W_BIT - 1, i * W_BIT);
    ap_int<W_BIT> w0_seg = w0(i * W_BIT + W_BIT - 1, i * W_BIT);
    wpack[i] =
        (w0_seg * (1 << (PROD_BIT * 2))) + (w1_seg * (1 << PROD_BIT)) + w2_seg;
  }
}

template <unsigned W_BIT, unsigned IN_BIT, unsigned SIMD, unsigned PROD_BIT>
void simd_MAC_normal(ap_int<W_BIT * SIMD> w0, ap_int<W_BIT * SIMD> w1,
                     ap_int<W_BIT * SIMD> w2, ap_uint<IN_BIT * SIMD> i0,
                     ap_uint<IN_BIT * SIMD> i1, ap_int<PROD_BIT + 5> &partial0,
                     ap_int<PROD_BIT + 5> &partial1,
                     ap_int<PROD_BIT + 5> &partial2,
                     ap_int<PROD_BIT + 5> &partial3) {
  ap_int<PROD_BIT + 5> r0, r1, r2, r3;
  r0 = 0;
  r1 = 0;
  r2 = 0;
  r3 = 0;
  for (int i = 0; i < SIMD; i++) {
    ap_int<W_BIT> w0_seg = w0((i + 1) * W_BIT - 1, i * W_BIT);
    ap_int<W_BIT> w1_seg = w1((i + 1) * W_BIT - 1, i * W_BIT);
    ap_int<W_BIT> w2_seg = w2((i + 1) * W_BIT - 1, i * W_BIT);
    ap_uint<IN_BIT> x0_seg = i0((i + 1) * IN_BIT - 1, i * IN_BIT);
    ap_uint<IN_BIT> x1_seg = i1((i + 1) * IN_BIT - 1, i * IN_BIT);

    // cout << "Weight :" << w0_seg << "," << w1_seg << "," << w2_seg << ",";
    // cout << "Input :" << x0_seg << "," << x1_seg << endl;

    // cout << x1_seg * w0_seg << ",";
    // cout << x0_seg * w0_seg + x1_seg * w1_seg << ",";
    // cout << x0_seg * w1_seg + x1_seg * w2_seg << ",";
    // cout << x0_seg * w2_seg << endl;

    r0 += x0_seg * w2_seg;
    r1 += x0_seg * w1_seg + x1_seg * w2_seg;
    r2 += x0_seg * w0_seg + x1_seg * w1_seg;
    r3 += x1_seg * w0_seg;
  }
  partial0 = r0;
  partial1 = r1;
  partial2 = r2;
  partial3 = r3;
}

template <unsigned W_BIT, unsigned IN_BIT, unsigned PROD_BIT, unsigned SIMD,
          unsigned CASCADE>
void simd_MAC(ap_int<PROD_BIT * 2 + W_BIT> wpack[SIMD],
              ap_uint<PROD_BIT + IN_BIT> ipack[SIMD],
              ap_int<PROD_BIT + 5> &partial0, ap_int<PROD_BIT + 5> &partial1,
              ap_int<PROD_BIT + 5> &partial2, ap_int<PROD_BIT + 5> &partial3) {
#pragma HLS ARRAY_PARTITION variable = wpack complete
#pragma HLS ARRAY_PARTITION variable = ipack complete
  ap_int<PROD_BIT + 5> r0, r1, r2, r3;
  r0 = 0;
  r1 = 0;
  r2 = 0;
  r3 = 0;
  for (int i = 0; i < SIMD; i += CASCADE) {
#pragma HLS unroll
    ap_int<PROD_BIT * 4> m = 0;
    for (int cs = 0; cs < CASCADE; cs++) {
#pragma HLS unroll
      m += wpack[i + cs] * ipack[i + cs];
    }

    ap_int<PROD_BIT> p0 = m(PROD_BIT - 1, 0);
    ap_int<PROD_BIT> p1 = m(PROD_BIT * 2 - 1, PROD_BIT) + m[PROD_BIT - 1];
    ap_int<PROD_BIT> p2 =
        m(PROD_BIT * 3 - 1, PROD_BIT * 2) + m[PROD_BIT * 2 - 1];
    ap_int<PROD_BIT> p3 =
        m(PROD_BIT * 4 - 1, PROD_BIT * 3) + m[PROD_BIT * 3 - 1];

    r0 += p0;
    r1 += p1;
    r2 += p2;
    r3 += p3;
  }
  partial0 = r0;
  partial1 = r1;
  partial2 = r2;
  partial3 = r3;
}

template <unsigned W_BIT, unsigned IN_BIT, unsigned PROD_BIT, unsigned SIMD>
void simd_MAC_compare(ap_int<PROD_BIT * 2 + W_BIT> wpack[SIMD],
                      ap_uint<PROD_BIT + IN_BIT> ipack[SIMD],
                      ap_int<W_BIT * SIMD> w0, ap_int<W_BIT * SIMD> w1,
                      ap_int<W_BIT * SIMD> w2, ap_uint<IN_BIT * SIMD> i0,
                      ap_uint<IN_BIT * SIMD> i1, ap_int<PROD_BIT + 5> &partial0,
                      ap_int<PROD_BIT + 5> &partial1,
                      ap_int<PROD_BIT + 5> &partial2,
                      ap_int<PROD_BIT + 5> &partial3) {

  ap_int<PROD_BIT + 5> r0, r1, r2, r3;
  r0 = 0;
  r1 = 0;
  r2 = 0;
  r3 = 0;
  for (int i = 0; i < SIMD; i++) {

    ap_int<PROD_BIT * 4> m = wpack[i] * ipack[i];
    ap_int<PROD_BIT> p0 = m(PROD_BIT - 1, 0);
    ap_int<PROD_BIT> p1 = m(PROD_BIT * 2 - 1, PROD_BIT) + m[PROD_BIT - 1];
    ap_int<PROD_BIT> p2 =
        m(PROD_BIT * 3 - 1, PROD_BIT * 2) + m[PROD_BIT * 2 - 1];
    ap_int<PROD_BIT> p3 =
        m(PROD_BIT * 4 - 1, PROD_BIT * 3) + m[PROD_BIT * 3 - 1];

    ap_int<W_BIT> w0_seg = w0((i + 1) * W_BIT - 1, i * W_BIT);
    ap_int<W_BIT> w1_seg = w1((i + 1) * W_BIT - 1, i * W_BIT);
    ap_int<W_BIT> w2_seg = w2((i + 1) * W_BIT - 1, i * W_BIT);
    ap_uint<IN_BIT> x0_seg = i0((i + 1) * IN_BIT - 1, i * IN_BIT);
    ap_uint<IN_BIT> x1_seg = i1((i + 1) * IN_BIT - 1, i * IN_BIT);
    // cout << "Weight :" << w0_seg << "," << w1_seg << "," << w2_seg << endl;
    // cout << "Input :" << x0_seg << "," << x1_seg << endl;
    // cout << "Wpack :" << wpack[i] << endl;
    // cout << "Ipack :" << ipack[i] << endl;

    // cout << r0 << "," << p0 << "," << x0_seg * w2_seg << endl;
    // cout << r1 << "," << p1 << "," << x0_seg * w1_seg + x1_seg * w2_seg <<
    // endl; cout << r2 << "," << p2 << "," << x0_seg * w0_seg + x1_seg *
    // w1_seg
    // << endl; cout << r3 << "," << p3 << "," << x1_seg * w0_seg << endl;

    // assert(p0 == x0_seg * w2_seg);
    // assert(p1 == x0_seg * w1_seg + x1_seg * w2_seg);
    // assert(p2 == x0_seg * w0_seg + x1_seg * w1_seg);
    // assert(p3 == x1_seg * w0_seg);

    r0 += p0;
    r1 += p1;
    r2 += p2;
    r3 += p3;
  }
  partial0 = r0;
  partial1 = r1;
  partial2 = r2;
  partial3 = r3;
}

template <unsigned K, unsigned IN_BIT, unsigned IN_CH, unsigned OUT_BIT,
          unsigned OUT_W, unsigned OUT_H, unsigned OUT_CH, unsigned W_BIT,
          unsigned GUARD_BIT, unsigned M_BIT, unsigned INC_BIT,
          unsigned BIAS_BIT, unsigned SIMD, unsigned CASCADE, unsigned PE,
          unsigned L_SHIFT>
void convDSPOpt(
    stream<ap_uint<SIMD * IN_BIT * 2>> &vec,
    const ap_uint<SIMD * W_BIT> weights[PE][3][K * IN_CH / SIMD * OUT_CH / PE],
    const ap_int<INC_BIT> inc[PE][OUT_CH / PE],
    const ap_int<BIAS_BIT> bias[PE][OUT_CH / PE],
    stream<ap_uint<PE * OUT_BIT * 2>> &out,
    // stream<ap_uint<PE * M_BIT * 2>> &out,
    const unsigned reps = 1) {

  static_assert(IN_CH % SIMD == 0, "IN_CH % SIMD !=0");
  static_assert(SIMD % CASCADE == 0, "SIMD % CASCADE != 0");
  static_assert(CASCADE <= 4, "SIMD % CASCADE != 0");
  const unsigned PENUM = OUT_CH / PE;
  const unsigned SIMDNUM = IN_CH / SIMD;
  const unsigned PROD_BIT = W_BIT + IN_BIT + GUARD_BIT;
  const unsigned WPACK_BIT = W_BIT * 3 + IN_BIT * 2 + GUARD_BIT * 2;
  const unsigned IPACK_BIT = IN_BIT * 2 + W_BIT + GUARD_BIT * 1;
  const unsigned INFOLD = K * SIMDNUM;

#pragma HLS ARRAY_PARTITION variable = weights complete dim = 1
#pragma HLS ARRAY_PARTITION variable = weights complete dim = 2
#pragma HLS ARRAY_PARTITION variable = inc complete dim = 1
#pragma HLS ARRAY_PARTITION variable = bias complete dim = 1
  //   ap_int<PROD_BIT * 2> m_hi[INFOLD][PE][SIMD];
  // #pragma HLS ARRAY_PARTITION variable = m_hi complete dim = 2
  //   ap_int<PROD_BIT * 2> m_lo[PE][SIMD];
  // #pragma HLS ARRAY_PARTITION variable = m_lo complete dim = 1

  ap_int<WPACK_BIT> wpacks[PE][SIMD];
#pragma HLS ARRAY_PARTITION variable = wpacks complete dim = 1
#pragma HLS ARRAY_PARTITION variable = wpacks complete dim = 2

  ap_uint<IPACK_BIT> ipack[SIMD];
#pragma HLS ARRAY_PARTITION variable = ipack complete dim = 1

  // ap_uint<12> weightAddr = 0;
  ap_int<M_BIT> firPartialRes0[PE];
#pragma HLS ARRAY_PARTITION variable = firPartialRes0 complete dim = 1
  ap_int<M_BIT> firPartialRes1[PE];
#pragma HLS ARRAY_PARTITION variable = firPartialRes1 complete dim = 1

  ap_int<M_BIT> outPartialArr0[PE];
#pragma HLS ARRAY_PARTITION variable = outPartialArr0 complete dim = 1
  ap_int<M_BIT> outPartialArr1[PE];
#pragma HLS ARRAY_PARTITION variable = outPartialArr1 complete dim = 1

  for (unsigned int h = 0; h < OUT_H * reps; h++) {
    for (unsigned int peIdx = 0; peIdx < PENUM; peIdx++) {
      for (unsigned int w = 0; w < OUT_W + K - 1; w += 2) {
        for (unsigned int infoldIdx = 0; infoldIdx < INFOLD; infoldIdx++) {
#pragma HLS pipeline
          bool m_clear = (w == 0);
          bool o_clear = (infoldIdx == 0);
          bool o_out = (infoldIdx == INFOLD - 1 && w != 0);
          ap_uint<SIMD * IN_BIT> data1, data0;
          (data1, data0) = vec.read();
          pack_input_data<IN_BIT, SIMD, PROD_BIT>(data1, data0, ipack);
          for (unsigned p = 0; p < PE; p++) {
            pack_weight_data<W_BIT, SIMD, PROD_BIT>(
                weights[p][2][peIdx * INFOLD + infoldIdx],
                weights[p][1][peIdx * INFOLD + infoldIdx],
                weights[p][0][peIdx * INFOLD + infoldIdx], wpacks[p]);
          }

          for (int p = 0; p < PE; p++) {
            // cout << "FIR result compare " << endl;

            ap_int<PROD_BIT + 5> firPartial0;
            ap_int<PROD_BIT + 5> firPartial1;
            ap_int<PROD_BIT + 5> firPartial2;
            ap_int<PROD_BIT + 5> firPartial3;

            // simd_MAC_normal<W_BIT, IN_BIT, SIMD, PROD_BIT>(
            //     weights[p][0][peIdx * INFOLD + infoldIdx],
            //     weights[p][1][peIdx * INFOLD + infoldIdx],
            //     weights[p][2][peIdx * INFOLD + infoldIdx], data0, data1,
            //     firPartial0, firPartial1, firPartial2, firPartial3);
            simd_MAC<W_BIT, IN_BIT, PROD_BIT, SIMD, CASCADE>(
                wpacks[p], ipack, firPartial0, firPartial1, firPartial2,
                firPartial3);
            // getchar();

            if (o_clear) {
              outPartialArr0[p] = firPartial0 + firPartialRes0[p];
              outPartialArr1[p] = firPartial1 + firPartialRes1[p];
              firPartialRes0[p] = firPartial2;
              firPartialRes1[p] = firPartial3;
            } else {
              outPartialArr0[p] += firPartial0;
              outPartialArr1[p] += firPartial1;
              firPartialRes0[p] += firPartial2;
              firPartialRes1[p] += firPartial3;
            }

            // if (p == 0) {
            //   cout << firPartial0 << "," << firPartial1 << "," <<
            //   firPartial2
            //        << "," << firPartial3 << endl;
            //   cout << outPartialArr0[p] << "," << outPartialArr1[p] <<
            //   endl; getchar();
            // }
          }
          ap_int<OUT_BIT * PE> oData0;
          ap_int<OUT_BIT * PE> oData1;

          if (o_out) {
            // ap_uint<PE * M_BIT> out_buf0;
            // ap_uint<PE * M_BIT> out_buf1;
            for (int p = 0; p < PE; p++) {
              // out_buf0(p * M_BIT + M_BIT - 1, p * M_BIT) = outPartialArr0[p];
              // out_buf1(p * M_BIT + M_BIT - 1, p * M_BIT) = outPartialArr1[p];
              oData0((p + 1) * OUT_BIT - 1, p * OUT_BIT) =
                  bn_qurelu_fixed<M_BIT, OUT_BIT, INC_BIT, BIAS_BIT, IN_BIT,
                                  W_BIT, L_SHIFT>(
                      outPartialArr0[p], inc[p][peIdx], bias[p][peIdx]);
              oData1((p + 1) * OUT_BIT - 1, p * OUT_BIT) =
                  bn_qurelu_fixed<M_BIT, OUT_BIT, INC_BIT, BIAS_BIT, IN_BIT,
                                  W_BIT, L_SHIFT>(
                      outPartialArr1[p], inc[p][peIdx], bias[p][peIdx]);
            }
            out.write((oData1, oData0));
            // out.write((out_buf1, out_buf0));
          }
        }
      }
    }
  }
}
/**
 * 矩阵向量计算单元
 * 同时进行量化激活处理
 */

// weights-> PE->SIMD->wr->col

/**
 * 卷积计算单元 同时计算bn_层与激活层
 * 在矩阵向量计算后立即计算得到激活输出值
 * 只计算 3x3 的卷积 K = 3, P = 1 S = 1
 * 输入数据宽度 为 IN_STREAM_BIT
 * 输出数据宽度为 PE * OUT_BIT
 */
template <unsigned IN_ROW, unsigned IN_COL, unsigned IN_CH, unsigned IN_BIT,

          unsigned OUT_CH,
          unsigned OUT_BIT, // 量化激活后的位宽

          unsigned W_BIT, unsigned M_BIT, unsigned INC_BIT, unsigned BIAS_BIT,

          unsigned SIMD, unsigned CASCADE, unsigned IN_PE, unsigned PE,
          unsigned L_SHIFT>
void conv3x3_bn_act_DSPopt(
    stream<ap_uint<IN_BIT * IN_PE * 2>> &in,
    const ap_uint<SIMD * W_BIT> weights[PE][3]
                                       [((IN_CH * 3) / SIMD) * (OUT_CH / PE)],
    const ap_int<INC_BIT> inc[PE][OUT_CH / PE],
    const ap_int<BIAS_BIT> bias[PE][OUT_CH / PE],
    stream<ap_uint<OUT_BIT * PE * 2>> &out, const unsigned reps = 1) {
#pragma HLS DATAFLOW

  const unsigned INTER_ROW = IN_ROW + 2;
  const unsigned INTER_COL = IN_COL + 2;
  // 暂时认为输入 输出维度不变
  const unsigned OUT_ROW = IN_ROW;
  const unsigned OUT_COL = IN_COL;

  stream<ap_uint<SIMD * IN_BIT * 2>> padding_out("padding_out");
  conv3padding<3, IN_ROW, IN_COL, IN_CH, IN_BIT, IN_PE, SIMD, OUT_CH / PE>(
      in, padding_out, reps);

  stream<ap_uint<PE * OUT_BIT * 2>> mvau_out("mvau_out");
  convDSPOpt<3, IN_BIT, IN_CH, OUT_BIT, OUT_COL, OUT_ROW, OUT_CH, W_BIT, 3,
             M_BIT, INC_BIT, BIAS_BIT, SIMD, CASCADE, PE, L_SHIFT>(
      padding_out, weights, inc, bias, out, reps);
}
#endif
