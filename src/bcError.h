#ifndef BC_ERROR_H
#define BC_ERROR_H

#include "Eigen/Dense"
#include "ctrConstants.h"

namespace CtrLib{
/**
 * @brief Error function computing the residuals of the boundary conditions
 * 
 * @param y     Matrix containing the state variable at each integration node
 * @param iEnd  Vector containing the index of the segment where each tube ends
 * @param kxy1  Bending stiffness of inner tube
 * @param Ux1   Precurvature of inner tube
 * @param w     External wrench applied on the end-effector
 * @return Vector containing the residuals of the boundary conditions [u1z, u2z, u3z, mx, my].
 */

Vector_bc bcError(
  const Matrix_yTot &y,
  const Eigen::Vector<int, NB_TUBES> &iEnd,
  double kxy1,
  double kz1,
  double Ux1,
  const Vector_w &w);
}
#endif //BC_ERROR_H
