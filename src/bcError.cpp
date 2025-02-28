#include <iostream>
#include "Eigen/Dense"
#include "ctrConstants.h"

using namespace Eigen;

namespace CtrLib{
Vector_bc bcError( 
  const Matrix_yTot &y,
  const tubeParameters &tubes,
  const Eigen::Vector<int, NB_TUBES> &iEnd,
  const Vector_w &w)
{

  // err(0) : sum of internal moment along x axis should be equal to external moment at the end of the CTR
  // err(1) : sum of internal moment along y axis should be equal to external moment at the end of the CTR
  // err(2) : sum of internal moment along z axis should be equal to external moment at the end of the CTR
  // err(3) : u2z should be zero at the end of second tube (assuming no external moment at the end of the second tube)
  // err(4) : u3z should be zero at the end of third tube (assuming no external moment at the end of the third tube)

  Vector_bc err;

  // Convert index of segment to column index in y matrix
  int yTot_iEnd0 = getYtotIndexFromIend(iEnd(0));
  int yTot_iEnd1 = getYtotIndexFromIend(iEnd(1));
  int yTot_iEnd2 = getYtotIndexFromIend(iEnd(2));

  Vector<double, NB_STATE_VAR> y_tip = y(all, yTot_iEnd0); // state variables at the tip of the CTR

  // Compute internal moment at the end of the CTR (assuming that only the first tube is present at the end of the CTR)
  //Matrix3d R1 = getRFromYtot(y,segmented);
  Matrix3d R1_tip = y_tip(seqN(3,9)).reshaped<RowMajor>(3,3);
  Vector3d u1_tip = y_tip(seqN(12,3));
  Matrix3d K1 = Matrix3d(Vector3d(tubes.Kxy(0), tubes.Kxy(0), tubes.Kz(0)).asDiagonal()); 
  Vector3d U1(tubes.Ux(0), 0, 0);
  Vector3d deltaU1_tip = u1_tip - U1;
  Vector3d m1_tip = R1_tip * K1 * deltaU1_tip;

  err(seqN(0,3)) = m1_tip - w(seqN(3,3)); // residual for internal moment at the tip of the CTR
  err(3) = y(15,yTot_iEnd1); // residual for u2z at the end of the second tube
  err(4) = y(16,yTot_iEnd2); // residual for u2z at the end of the second tube

  return err;
}
}
