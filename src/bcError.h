#ifndef BC_ERROR_H
#define BC_ERROR_H

#include "Eigen/Dense"
#include "ctrConstants.h"

namespace CtrLib{
/**
 * @brief Error function computing the residuals of the boundary conditions
 * 
 * @param y     Matrix containing the state variable at each integration node
 * @param tubes Struct containing tube parameters
 * @param iEnd  Vector of the segment index where each tube ends
 * @param w     External wrench applied on the end-effector
 * @return Vector containing the residuals of the boundary conditions [mx, my, mz, u2z, u3z].
 */

Vector_bc bcError(
  const Matrix_yTot &y,
  const tubeParameters &tubes,
  const Eigen::Vector<int, NB_TUBES> &iEnd,
  const Vector_w &w);
}
#endif //BC_ERROR_H
