#include <numeric> // std::iota
#include <iostream>
#include <vector>
#include "Eigen/Dense"
#include "ctrConstants.h"
#include "segmenting.h"

using namespace Eigen;

namespace CtrLib
{

  void sort_indexes(
      const Eigen::Matrix<double, 2 * NB_TUBES + 1, 1> &v,
      Eigen::Matrix<double, 2 * NB_TUBES + 1, 1> &v_sorted_out,
      Eigen::Matrix<int, 2 * NB_TUBES + 1, 1> &index_out) {

    constexpr int NB_2p1 = 2 * NB_TUBES + 1;
    std::array<int, NB_2p1> idx;
    for (int i = 0; i < NB_2p1; ++i)
      idx[i] = i;

    // Insertion sort more efficient for N < 20–50 (very fast, cache-friendly)
    // Even though std::stable_sort may already switch internally to insertion
    // sort for small fixed-size Eigen vectors (say N ≤ 10–20), But doing it m
    // anually gives full control, avoids dynamic memory allocation (for std
    // ::array), and ensures consistent performance.
    // We may later switch the sorting algorithm depending on N
    /*
    ** std::stable_sort(idx.begin(), idx.end(),
    ** [&v](int i1, int i2){ return v(i1) < v(i2); });
    */
    for (int i = 1; i < NB_2p1; ++i) {
      int key_idx = idx[i];
      double key_val = v(key_idx);
      int j = i - 1;
      while (j >= 0 && v(idx[j]) > key_val) {
        idx[j + 1] = idx[j];
        --j;
      }
      idx[j + 1] = key_idx;
    }

    // Fill sorted values
    for (int i = 0; i < NB_2p1; ++i) {
      index_out(i) = idx[i];
      v_sorted_out(i) = v(idx[i]);
    }
  }

  void diff(
      const Vector<double, 2 * NB_TUBES + 1> &v,
      Vector<double, 2 * NB_TUBES> &diff_v_out) {
    int n = v.size();
    for (int i = 0; i < n - 1; i++) {
      diff_v_out(i) = v(i + 1) - v(i);
    }
  }

  int segmenting(
      const Vector_q &q,
      const tubeParameters &tubes,

      segmentedData &segmented_out) {

    // out : [S,iEnd,EE,II,GG,JJ,UUx]
    Vector<double, NB_TUBES> beta = q(Eigen::seqN(0, NB_TUBES));                       // arc length of tube base (translation actuation)
    Vector<double, NB_TUBES> tubeTipArcLength = tubes.l + beta;                        // arc lengths of the tip of the tubes
    Vector<double, NB_TUBES> tubePrecurvatureArcLength = tubeTipArcLength - tubes.l_k; // arc-length where the tubes precurvature starts
    Vector<double, 2 * NB_TUBES + 1> points;                                           // number of segment delimitations = 1 + tubePrecurvatureArcLength.size() + tubeTipArcLength.size()
    points << 0, tubePrecurvatureArcLength, tubeTipArcLength;

    Vector<double, 2 * NB_TUBES + 1> points_srt;
    Vector<int, 2 * NB_TUBES + 1> index;

    sort_indexes(points, points_srt, index);

    Vector<double, 2 * NB_TUBES> L; // Segment length
    diff(points_srt, L);
    // # Why?????
    for (int i = 0; i < 2 * NB_TUBES; i++) {
      L(i) = 1e-5 * floor(L(i) * 1e5); // remove "empty segment" (shorter than 10 µm)
    }

    // Check that the telescopic configuration of the tubes is correct
    // i.e. beta_1 < beta_2 < beta_3 < 0 < tip_3 < tip_2 < tip_1 (for a 3-tube CTR)
    for (int i = 0; i < NB_TUBES - 1; i++) {
      if (beta(i) > beta(i + 1)) {
        PRINT_DEBUG_MSG("proximal end of tube " << i + 1 << " is clashing into tube " << i + 2);
        PRINT_DEBUG_MSG("beta = " << std::endl
                                  << beta);
        PRINT_DEBUG_MSG("q = " << std::endl
                               << q);
        return -1;
      }
      if (tubeTipArcLength(i) < tubeTipArcLength(i + 1)) {
        PRINT_DEBUG_MSG("distal end of tube " << i + 1 << " is clashing into tube " << i + 2);
        PRINT_DEBUG_MSG("tubeTipArcLength = " << std::endl
                                              << tubeTipArcLength);
        PRINT_DEBUG_MSG("q = " << std::endl
                               << q);
        return -1;
      }
    }
    if (beta(NB_TUBES - 1) > 0) {
      PRINT_DEBUG_MSG("proximal end of tube " << NB_TUBES - 1 << " is outside the actuation unit");
      PRINT_DEBUG_MSG("beta = " << std::endl
                                << beta);
      PRINT_DEBUG_MSG("q = " << std::endl
                             << q);
      return -1;
    }
    if (tubeTipArcLength(NB_TUBES - 1) < 0) {
      PRINT_DEBUG_MSG("distal end of tube " << NB_TUBES - 1 << " is inside the actuation unit");
      PRINT_DEBUG_MSG("tubeTipArcLength = " << std::endl
                                            << tubeTipArcLength);
      PRINT_DEBUG_MSG("q = " << std::endl
                             << q);
      return -1;
    }

    int i0;
    Vector<int, NB_TUBES> iCurved;
    segmented_out.iEnd = Vector<int, NB_TUBES>::Zero();
    // for (int i = 0; i < NB_TUBES; i++) {
    //   for (int j = 0; j < 2 * NB_TUBES + 1; j++) {
    //     if (index(j) == 0) {
    //       i0 = j;
    //     }
    //     else if (index(j) == i + 1) {
    //       iCurved(i) = j;
    //     }
    //     else if (index(j) == NB_TUBES + i + 1) {
    //       segmented_out.iEnd(i) = j;
    //     }
    //   }
    // }

    for (int j = 0; j < 2 * NB_TUBES + 1; j++) {
      if (index(j) == 0) {
        i0 = j;
      }
      else if (0 <index(j) && index(j) <= NB_TUBES) {
        // # TODO... For correct implementation (in wich segments are '0' indexed) this should be
        // iCurved(index(j)-1) = j-1;
        iCurved(index(j)-1) = j;
      }
      else if (NB_TUBES < index(j)) {
        // # TODO... For correct implementation (in wich segments are '0' indexed) this should be
        // segmented_out.iEnd(index(j) - 1 - NB_TUBES) = j-1;        
        segmented_out.iEnd(index(j) - 1 - NB_TUBES) = j;
      }
    }
    /*Optionnal zero-padding : just for easier debugging*/
    segmented_out.S = Vector_S::Zero();
    segmented_out.KKxy = Matrix_segmented::Zero();
    segmented_out.KKz = Matrix_segmented::Zero();
    segmented_out.UUx = Matrix_segmented::Zero();
    double sumL = 0;
    int iSegment = 0;
    for (int j = i0; j < 2 * NB_TUBES; j++) {
      sumL += L(j);
      segmented_out.S(iSegment) = sumL;
      for (int i = 0; i < NB_TUBES; i++) {
        // # iEnd is 'vector of the segment index where each tube ends' for the '1' indexed 
        // segments (where the end of a tube is the end of the '1' indexed segements)
        // # The point is, up unitl now, we have segment with index 6 and no segment with '0' 
        // index which means that the segments is '1' indexed
        // # TODO... For correct implementation (in wich segments are '0' indexed) this should be
        // if (j <= segmented_out.iEnd(i))
        if (j < segmented_out.iEnd(i)) { // if tube exists
          // # Somehow This must be wrong as IEnd holds the '1' indexed segment index 
          // and both KKxy and KKz hold '0' indexed segment index
          segmented_out.KKxy(i, iSegment) = tubes.Kxy(i);
          segmented_out.KKz(i, iSegment) = tubes.Kz(i);
        }
        // # TODO... For correct implementation (in wich segments are '0' indexed) this should be
        // if (j >= iCurved(i) && j <= segmented_out.iEnd(i))
        if (j >= iCurved(i) && j < segmented_out.iEnd(i)) { // if tube is curved
          segmented_out.UUx(i, iSegment) = tubes.Ux(i);
        }
      }
      iSegment++;
    }
    // # Now IEnd is '0' Indexed but iCurved is not and
    // this is not a probelm as it is a local variable
    segmented_out.iEnd.array() -= (i0 + 1); // remove segments before 0 so iEnd correspond to the index of the last segment where ith tube exists
    return 0;
  }
}