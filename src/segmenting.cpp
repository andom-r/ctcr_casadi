#include <numeric>      // std::iota
#include <iostream>
#include <vector>
#include "Eigen/Dense"
#include "ctrConstants.h"
#include "segmenting.h"

using namespace Eigen;

namespace CtrLib{
  void sort_indexes(
    const Vector<double, 2 * NB_TUBES + 1>  &v,
    Vector<double,2 * NB_TUBES + 1>  &v_sorted_out,
    Vector<int,2 * NB_TUBES + 1>  &index_out){

    int n = v.size();
    std::vector<int> idx(n);
    iota(idx.begin(), idx.end(), 0);

    std::stable_sort(idx.begin(), idx.end(),
      [&v](size_t i1, size_t i2) {return v[i1] < v[i2];});
    for(int i = 0; i < n; i++){
      index_out(i) = idx[i];
      v_sorted_out(i) = v(idx[i]);
    }
  }
  void diff(
    const Vector<double,2 * NB_TUBES + 1>  &v,
    Vector<double,2 * NB_TUBES>  &diff_v_out){

    int n = v.size();
    for(int i = 0; i < n - 1; i++){
      diff_v_out(i) = v(i + 1) - v(i);
    }
  }
  int segmenting(
    const Vector_q &q,
    const tubeParameters &tubes,

    segmentedData &segmented_out){

    // out : [S,iEnd,EE,II,GG,JJ,UUx]
    Vector<double, NB_TUBES> beta = q(Eigen::seqN(0, NB_TUBES)); // arc length of tube base (translation actuation)
    Vector<double, NB_TUBES> tubeTipArcLength = tubes.l + beta;      // arc lengths of the tip of the tubes
    Vector<double, NB_TUBES> tubePrecurvatureArcLength = tubeTipArcLength - tubes.l_k;   // arc-length where the tubes precurvature starts
    Vector<double, 2 * NB_TUBES + 1> points; // number of segment delimitations = 1 + tubePrecurvatureArcLength.size() + tubeTipArcLength.size()
    points << 0, tubePrecurvatureArcLength, tubeTipArcLength;

    Vector<double, 2 * NB_TUBES + 1> points_srt;
    Vector<int, 2 * NB_TUBES + 1> index;

    sort_indexes(points, points_srt, index);

    Vector<double,2 * NB_TUBES> L; // Segment length
    diff(points_srt,L);
    for(int i = 0; i < 2 * NB_TUBES; i++){
      L(i) = 1e-5 * floor(L(i) * 1e5); // remove "empty segment" (shorter than 10 µm) 
    }
    
    // Check that the telescopic configuration of the tubes is correct
    // i.e. beta_1 < beta_2 < beta_3 < 0 < tip_3 < tip_2 < tip_1 (for a 3-tube CTR)
    for (int i = 0; i < NB_TUBES - 1; i++){
      if (beta(i) > beta(i + 1)){
        PRINT_DEBUG_MSG("proximal end of tube " << i + 1 << " is clashing into tube " << i + 2);
        PRINT_DEBUG_MSG("beta = " << std::endl << beta);
        PRINT_DEBUG_MSG("q = " << std::endl << q);
        return -1;
      }
      if(tubeTipArcLength(i) < tubeTipArcLength(i + 1)){
        PRINT_DEBUG_MSG("distal end of tube " << i + 1 << " is clashing into tube " << i + 2);
        PRINT_DEBUG_MSG("tubeTipArcLength = " << std::endl << tubeTipArcLength);
        PRINT_DEBUG_MSG("q = " << std::endl << q);
        return -1;
      }
    }
    if(beta(NB_TUBES - 1) > 0){
        PRINT_DEBUG_MSG("proximal end of tube " << NB_TUBES - 1 << " is outside the actuation unit");
        PRINT_DEBUG_MSG("beta = " << std::endl << beta);
        PRINT_DEBUG_MSG("q = " << std::endl << q);
        return -1;
    }
    if(tubeTipArcLength(NB_TUBES - 1) < 0){
        PRINT_DEBUG_MSG("distal end of tube " << NB_TUBES - 1 << " is inside the actuation unit");
        PRINT_DEBUG_MSG("tubeTipArcLength = " << std::endl << tubeTipArcLength);
        PRINT_DEBUG_MSG("q = " << std::endl << q);
        return -1;
    }

    int i0;
    Vector<int, NB_TUBES> iCurved;
    segmented_out.iEnd = Vector<int, NB_TUBES>::Zero();
    for (int i = 0; i < NB_TUBES; i++){
      for (int j = 0; j < 2 * NB_TUBES + 1; j++){
        if(index(j) == 0){
          i0 = j;
        }
        else if(index(j) == i + 1){
          iCurved(i) = j;
        }
        else if(index(j) == NB_TUBES + i + 1){
          segmented_out.iEnd(i) = j;
        }
      }

    }
    /*Optionnal zero-padding : just for easier debugging*/
    segmented_out.S    = Vector_S::Zero();
    segmented_out.KKxy = Matrix_segmented::Zero();
    segmented_out.KKz  = Matrix_segmented::Zero();
    segmented_out.UUx  = Matrix_segmented::Zero();
    double sumL = 0;
    int iSegment = 0;
    for (int j = i0; j < 2 * NB_TUBES; j++){
        sumL += L(j);
        segmented_out.S(iSegment) = sumL;
        for (int i = 0; i < NB_TUBES; i++){
          if(j < segmented_out.iEnd(i)){ // if tube exists
            segmented_out.KKxy(i,iSegment) = tubes.Kxy(i);
            segmented_out.KKz(i,iSegment) = tubes.Kz(i);
          }
          if(j >= iCurved(i) && j < segmented_out.iEnd(i)){ // if tube is curved
            segmented_out.UUx(i,iSegment) = tubes.Ux(i);
          }
        }
        iSegment++;
    }
    segmented_out.iEnd.array() -= (i0 + 1); // remove segments before 0 so iEnd correspond to the index of the last segment where ith tube exists
    return 0;
  }
}