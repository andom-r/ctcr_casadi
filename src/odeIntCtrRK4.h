#ifndef ODE_INT_CTR_RK4_H
#define ODE_INT_CTR_RK4_H

#include "Eigen/Dense"
#include "ctrConstants.h"

namespace CtrLib{
  /**
  * @brief Forward integration of ODEs using Runge-Kutta 4 scheme
  * 
  * @param[in] y0 value of state variables at the beginning of the segment
  * @param[in] s0 arc-length at the beginning of the segment
  * @param[in] sf arc-length at the end of the segment
  * @param[in] Kxy Vector of bending stiffness for each tube on the considered segment
  * @param[in] Kz Vector of torsional stiffness for each tube on the considered segment
  * @param[in] Ux Vector of precurvature for each tube on the considered segment
  * @param[in] w external wrench applied at the end-effector
  * 
  * @return Matrix containing the values of the state variables for each integration node on the considered segment 
  */

  Eigen::Matrix<double, NB_STATE_VAR, NB_INTEGRATION_NODES> odeIntCtrRK4(
    Eigen::Vector<double, NB_STATE_VAR> y0,
    double s0,
    double sf,
    const Eigen::Vector<double, NB_TUBES>& Kxy,
    const Eigen::Vector<double, NB_TUBES>& Kz,
    const Eigen::Vector<double, NB_TUBES>& Ux,
    const Vector_w& w);
}
#endif //ODE_INT_CTR_RK4_H
