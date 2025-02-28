#include <iostream>
#include "Eigen/Dense"
#include "ctrConstants.h"

using namespace Eigen;
using namespace CTR_CONST;

Vector<double, NB_BC> bcError( 
  const Matrix<double,nStateVar,nSegMax*nIntPoints> &y,
  const Vector<int,3> &iEnd,
  double kxy1,
  double kz1,
  double Ux1,
  const Vector_w &w){

  // err(0) : sum of internal moment along x axis should be equal to external moment at the end of the CTR
  // err(1) : sum of internal moment along y axis should be equal to external moment at the end of the CTR
  // err(2) : sum of internal moment along z axis should be equal to external moment at the end of the CTR
  // err(3) : u2z should be zero at the end of second tube (assuming no external moment at the end of the second tube)
  // err(4) : u3z should be zero at the end of third tube (assuming no external moment at the end of the third tube)

  Vector<double, NB_BC> err;

  // Convert index of segment to column index in y matrix
  int i_end0 = getYtotIndexFromIend(iEnd(0));
  int i_end1 = getYtotIndexFromIend(iEnd(1));
  int i_end2 = getYtotIndexFromIend(iEnd(2));
  
  // Compute internal moment at the end of the CTR (assuming that only the first tube is present at the end of the CTR)
  Matrix3d R1 = y(seqN(3,9), i_end0).reshaped<RowMajor>(3,3);
  Matrix3d K1 = Matrix3d(Vector3d(kxy1, kxy1, kz1).asDiagonal()); 
  Vector3d u1 = y(seqN(12,3),i_end0);
  Vector3d U1(Ux1, 0, 0);
  Vector3d deltaU1 = u1 - U1;
  Vector3d m1 = R1 * K1 * deltaU1;

  err(seqN(0,3)) = m1 - w(seqN(3,3)); // residual for internal moment at the end of the CTR
  err(3) = y(15,i_end1); // residual for u2z at the end of the second tube
  err(4) = y(16,i_end2); // residual for u2z at the end of the second tube

  return err;
}
