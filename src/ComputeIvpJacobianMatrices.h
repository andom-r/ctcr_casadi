#ifndef IVP_FD_H
#define IVP_FD_H

#include "Eigen/Dense"
#include "ctrConstants.h"

namespace CtrLib{
  // templated computation options
  // NO_LOAD : Compute only the model, unloaded
  // NO_LOAD_J : Compute the model along with the robot Jacobian matrix J, unloaded
  // LOAD : Compute only the model, with external loads
  // LOAD_J : Compute the model along with the robot Jacobian matrix J, with external loads
  // LOAD_J_C : Compute the model along with the robot Jacobian matrix J and compliance matrix C, with external loads
  //enum COMPUTATION_OPTION {NO_LOAD, NO_LOAD_J, LOAD, LOAD_J, LOAD_J_C};
  enum COMPUTATION_OPTION {LOAD, LOAD_J, LOAD_J_C};
  // Warning ! This is a temporary version of the code. Computation options NO_LOAD, NO_LOAD_J, and LOAD_J_C will be implemented shortly.
  enum COMPUTATION_OPTION;

  /**
  * @brief Compute  jacobian matrices using the IVP finite differences method
  * 
  * @tparam opt Option for chosing which elements to compute
  * @param[in] q Actuation variables [beta_i, alpha_i] with beta the translations [m] and alpha the rotations [rad] 
  * @param[in] yu0 Guess of unknown initial conditions : [mx0, my0, u1z0, u2z0, u3z0] (sum of bending moments along x and y axis, and torsion in each tube, at arc length s=0)
  * @param[in] Kxy Vector of bending stiffness for each tube
  * @param[in] Kz Vector of torsional stiffness for each tube
  * @param[in] Ux Vector of precurvature for each tube
  * @param[in] l Vector of effective length for each tube
  * @param[in] l_k Vector of precurved length for each tube (we assume a tube is composed of a straight part and a precurved part with constant precurvature)
  * @param[in] w External wrench applied at the end-effector (6d-vector expressed in the robot base frame)
  * 
  * @param[out] yTot_out Output state variables at each integration node associated with the "nominal" computation solveIVP(yu0,q,...)
  * @param[out] b_out Output residuals of boundary conditions associated with the "nominal" computation solveIVP(yu0,q,...)
  * @param[out] Eq_out Output jacobian matrix Eq (delta_X/delta_q)
  * @param[out] Eu_out Output jacobian matrix Eu (delta_X/delta_yu0)
  * @param[out] Ew_out Output jacobian matrix Ew (delta_X/delta_w)
  * @param[out] Bq_out Output jacobian matrix Bq (delta_b/delta_q)
  * @param[out] Bu_out Output jacobian matrix Bu (delta_b/delta_yu0)
  * @param[out] Bw_out Output jacobian matrix Bw (delta_b/delta_w)
  * @return int (0 indicates success and < 0 indicates failure )
  */
  int ComputeIvpJacobianMatrices( 
    const Vector_q &q,
    const Vector_yu0 &yu0,
    const Eigen::Vector<double, NB_TUBES> &Kxy,
    const Eigen::Vector<double, NB_TUBES> &Kz,
    const Eigen::Vector<double, NB_TUBES> &Ux,
    const Eigen::Vector<double, NB_TUBES> &l,
    const Eigen::Vector<double, NB_TUBES> &l_k,
    const Vector_w &w,

    Matrix_yTot &yTot_out,
    Vector_bc &b_out,
    Matrix_Eq &Eq_out,
    Matrix_Eu &Eu_out,
    Matrix_Ew &Ew_out,
    Matrix_Bq &Bq_out,
    Matrix_Bu &Bu_out,
    Matrix_Bw &Bw_out,
    const computationOptions &opt);

  template <COMPUTATION_OPTION opt> 
  int ComputeIvpJacobianMatrices( 
    const Vector_q &q,
    const Vector_yu0 &yu0,
    const Eigen::Vector<double, NB_TUBES> &Kxy,
    const Eigen::Vector<double, NB_TUBES> &Kz,
    const Eigen::Vector<double, NB_TUBES> &Ux,
    const Eigen::Vector<double, NB_TUBES> &l,
    const Eigen::Vector<double, NB_TUBES> &l_k,
    const Vector_w &w,

    Matrix_yTot &yTot_out,
    Vector_bc &b_out,
    Matrix_Eq &Eq_out,
    Matrix_Eu &Eu_out,
    Matrix_Ew &Ew_out,
    Matrix_Bq &Bq_out,
    Matrix_Bu &Bu_out,
    Matrix_Bw &Bw_out,
    uint nThread);
}
#endif //IVP_FD_H
