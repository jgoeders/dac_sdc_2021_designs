#include "config.h"
#include "conv1x1DSP2.hpp"
#include "conv2d.h"
#include "conv2d_DSPopt.hpp"
#include "conv2d_l0.hpp"
#include "debug.hpp"
#include "param.h"
#include "weight3.hpp"
#include "weights.hpp"
#include <ap_int.h>
#include <hls_stream.h>
#include <hls_video.h>
#include <stdint.h>
#define IN_IMAGE_WIDTH 640
#define IN_IMAGE_HEIGHT 360

#define RESIZE_IMAGE_WIDTH 320
#define RESIZE_IMAGE_HEIGHT 160

void conv3x3_bn_act_DSPopt_hls_wrapper(
    stream<ap_uint<CONV_0_IN_BIT * CONV_0_INPE>> &in,
    stream<ap_uint<CONV_0_OUT_BIT * CONV_0_PE_DSP6 * 2>> &out,
    unsigned int reps) {

#pragma HLS array_partition variable = conv_0_w_dspopt dim = 1 complete
#pragma HLS array_partition variable = conv_0_w_dspopt dim = 2 complete
#pragma HLS array_partition variable = conv_0_inc dim = 1 complete
#pragma HLS array_partition variable = conv_0_bias dim = 1 complete
  conv3x3_l0_bn_act_DSPopt<CONV_0_IFM_ROW, CONV_0_IFM_COL, CONV_0_IFM_CH,
                           CONV_0_IN_BIT, CONV_0_OFM_CH, CONV_0_OUT_BIT,
                           CONV_0_W_BIT, 26, CONV_0_INC_BIT_NEW,
                           CONV_0_BIAS_BIT_NEW, CONV_0_SIMD_DSP6, 3,
                           CONV_0_INPE, CONV_0_PE_DSP6, CONV_0_L_SHIFT>(
      in, conv_0_w_new, conv_0_inc_new, conv_0_bias_new, out, reps);
}
// void conv3x3_bn_act_hls_wrapper(
//     stream<ap_uint<CONV_0_IN_BIT * CONV_0_IFM_CH>> &in,
//     const ap_uint<CONV_0_SIMD_DSP6 * CONV_0_W_BIT>
//         weights[CONV_0_PE][((CONV_0_IFM_CH * 3 * 3) / CONV_0_SIMD_DSP6) *
//                            (CONV_0_OFM_CH / CONV_0_PE)],
//     const ap_int<CONV_0_INC_BIT> inc[CONV_0_PE][CONV_0_OFM_CH / CONV_0_PE],
//     const ap_int<CONV_0_BIAS_BIT> bias[CONV_0_PE][CONV_0_OFM_CH / CONV_0_PE],
//     stream<ap_uint<CONV_0_OUT_BIT * CONV_0_OFM_CH>> &out, unsigned int reps)
//     {

// #pragma HLS array_partition variable = conv_0_w dim = 1 complete
// #pragma HLS array_partition variable = conv_0_inc dim = 1 complete
// #pragma HLS array_partition variable = conv_0_bias dim = 1 complete

//   conv3x3_bn_act<CONV_0_IFM_ROW, CONV_0_IFM_COL, CONV_0_IFM_CH,
//   CONV_0_IN_BIT,

//                  CONV_0_OFM_CH, CONV_0_OUT_BIT,

//                  CONV_0_W_BIT, 32, CONV_0_INC_BIT, CONV_0_BIAS_BIT,

//                  CONV_0_SIMD, CONV_0_PE, CONV_0_L_SHIFT>(
//       in, conv_0_w, conv_0_inc, conv_0_bias, out, reps);
// }

template <unsigned K, unsigned IN_CH, unsigned OUT_CH, unsigned PE,
          unsigned SIMD, unsigned W_BIT>
void initialziation(
    ap_uint<SIMD * W_BIT> weights[PE][K][K * OUT_CH / PE * IN_CH / SIMD],
    string method) {

  for (int kr = 0; kr < K; kr++)
    for (int i = 0; i < IN_CH; i += SIMD)
      for (int o = 0; o < OUT_CH; o += PE)
        for (int kc = 0; kc < K; kc++)
          for (int p = 0; p < PE; p++) {
            ap_uint<SIMD * W_BIT> data;
            for (int s = 0; s < SIMD; s++) {
              if (method == "odepth") {
                data((s + 1) * W_BIT - 1, s * W_BIT) = p;
              } else if (method == "kernel") {
                data((s + 1) * W_BIT - 1, s * W_BIT) = kr * K + kc;
              } else {
                if (kr == K / 2 && kc == K / 2)
                  data((s + 1) * W_BIT - 1, s * W_BIT) = 1;
                else
                  data((s + 1) * W_BIT - 1, s * W_BIT) = 0;
              }
            }
            weights[p][kc][o / PE * K * IN_CH / SIMD + kr * IN_CH / SIMD +
                           i / SIMD] = data;
          }
}

template <unsigned IN_CH, unsigned OUT_CH, unsigned PE, unsigned SIMD,
          unsigned W_BIT>
void initialziation1x1(
    ap_uint<SIMD * W_BIT> weights[PE][OUT_CH / PE * IN_CH / SIMD],
    string method) {

  for (int i = 0; i < IN_CH; i += SIMD)
    for (int o = 0; o < OUT_CH; o += PE)
      for (int p = 0; p < PE; p++) {
        ap_uint<SIMD * W_BIT> data;
        for (int s = 0; s < SIMD; s++) {
          if (method == "odepth") {
            data((s + 1) * W_BIT - 1, s * W_BIT) = (ap_int<W_BIT>)(o + p);
          } else if (method == "kernel") {
            data((s + 1) * W_BIT - 1, s * W_BIT) = 1;
          }
        }
        weights[p][o / PE * IN_CH / SIMD + i / SIMD] = data;
      }
}

int main(int argc, char **argv) {

  hls::stream<ap_uint<CONV_0_IN_BIT * CONV_0_SIMD>> golden_in("golden_in");
  hls::stream<ap_uint<CONV_0_IN_BIT * CONV_0_SIMD>> test_in("test_in");

  ap_uint<CONV_0_IN_BIT> IFM[CONV_0_IFM_CH][CONV_0_IFM_ROW][CONV_0_IFM_COL];
  // for (int i = 0; i < CONV_0_IFM_CH; i++) {
  int initial = 1;

  load_featuremap<CONV_0_IFM_ROW, CONV_0_IFM_COL, CONV_0_IFM_CH, CONV_0_IN_BIT>(
      "data/featuremap/conv0_in.bin", IFM, 255);

  // for (int r = 0; r < CONV_0_IFM_ROW; r++)
  //   for (int c = 0; c < CONV_0_IFM_COL; c++) {
  //     IFM[0][r][c] = 461 * initial * 1 % 3613;
  //     IFM[1][r][c] = 461 * initial * 2 % 3613;
  //     IFM[2][r][c] = 461 * initial * 3 % 3613;
  //     initial++;
  //   }
  // // }

  // for (int r = 0; r < CONV_0_IFM_ROW; r++) {
  //   for (int c = 0; c < CONV_0_IFM_COL; c++) {

  //     for (int i = 0; i < CONV_0_IFM_CH; i += CONV_0_SIMD) {
  //       ap_uint<CONV_0_IN_BIT * CONV_0_SIMD> data;
  //       for (int s = 0; s < CONV_0_SIMD; s++) {
  //         data((s + 1) * CONV_0_IN_BIT - 1, s * CONV_0_IN_BIT) =
  //             IFM[i + s][r][c];
  //       }
  //       // golden_in << data;
  //       test_in << data;
  //     }
  //   }
  // }
  for (int r = 0; r < CONV_0_IFM_ROW; r++) {
    for (int c = 0; c < CONV_0_IFM_COL; c++) {

      for (int i = 0; i < CONV_0_IFM_CH; i += CONV_0_SIMD) {
        ap_uint<CONV_0_IN_BIT * CONV_0_SIMD> data;
        for (int s = 0; s < CONV_0_SIMD; s++) {
          data((s + 1) * CONV_0_IN_BIT - 1, s * CONV_0_IN_BIT) =
              IFM[i + s][r][c];
        }
        golden_in << data;
        test_in << data;
      }
    }
  }
  // for (int r = 0; r < CONV_0_IFM_ROW; r++) {
  //   for (int c = 0; c < CONV_0_IFM_COL; c++) {

  //     for (int i = 0; i < CONV_0_IFM_CH; i += CONV_0_SIMD) {
  //       ap_uint<CONV_0_IN_BIT * CONV_0_SIMD> data;
  //       for (int s = 0; s < CONV_0_SIMD; s++) {
  //         data((s + 1) * CONV_0_IN_BIT - 1, s * CONV_0_IN_BIT) =
  //             IFM[i + s][r][c];
  //       }
  //       golden_in << data;
  //       test_in << data;
  //     }
  //   }
  // }
  // test_in << data;
  // initialziation1x1<CONV_0_IFM_CH, CONV_0_OFM_CH, CONV_0_PE_DSP2,
  //                   CONV_0_SIMD_DSP2, CONV_0_W_BIT>(conv_0_w_dspopt,
  //                   "odepth");

  // hls::stream<ap_uint<CONV_0_OUT_BIT * CONV_0_OFM_CH>>
  // golden_out("golden_out");

  // conv3x3_bn_act_hls_wrapper(golden_in, conv_0_w, conv_0_inc, conv_0_bias,
  //                            golden_out, 1);

  // print_mavu_stream_through<CONV_0_OFM_ROW, CONV_0_OFM_COL, CONV_0_OFM_CH,
  //                           CONV_0_PE, CONV_0_OUT_BIT>(
  //     golden_out, "conv_ultranet_out.txt", 1);

  hls::stream<ap_uint<CONV_0_OUT_BIT * CONV_0_PE_DSP6 * 2>> test_out(
      "test_out");

  conv3x3_bn_act_DSPopt_hls_wrapper(test_in, test_out, 1);

  print_mavu_DSPopt_stream_through<CONV_0_OFM_ROW, CONV_0_OFM_COL,
                                   CONV_0_OFM_CH, CONV_0_PE_DSP6,
                                   CONV_0_OUT_BIT>(test_out,
                                                   "conv_DSP2_out.txt", 1);

  ap_uint<CONV_0_OUT_BIT> OFM[CONV_0_OFM_CH][CONV_0_OFM_ROW][CONV_0_OFM_COL];

  load_featuremap<CONV_0_OFM_ROW, CONV_0_OFM_COL, CONV_0_OFM_CH,
                  CONV_0_OUT_BIT>("data/featuremap/conv0_out.bin", OFM, 15);
  print_output_featuremap<CONV_0_OFM_ROW, CONV_0_OFM_COL, CONV_0_OFM_CH,
                          CONV_0_PE_DSP6, CONV_0_OUT_BIT>(
      OFM, "conv_gold_out.txt", 1);
}
