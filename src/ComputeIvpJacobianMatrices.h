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

  struct ComputeIvpJacMatOut{
    Matrix_yTot yTot;    // Output state variables at each integration node associated with the "nominal" computation solveIVP(yu0,q,...)
    Vector_bc   b;       // Output residuals of boundary conditions associated with the "nominal" computation solveIVP(yu0,q,...)
    Matrix_Eq   Eq;      // Output jacobian matrix Eq (delta_X/delta_q)
    Matrix_Eu   Eu;      // Output jacobian matrix Eu (delta_X/delta_yu0)
    Matrix_Ew   Ew;      // Output jacobian matrix Ew (delta_X/delta_w)
    Matrix_Bq   Bq;      // Output jacobian matrix Bq (delta_b/delta_q)
    Matrix_Bu   Bu;      // Output jacobian matrix Bu (delta_b/delta_yu0)
    Matrix_Bw   Bw;      // Output jacobian matrix Bw (delta_b/delta_w)
  };

  struct SingleIVPOut{
        Matrix_yTot yTot;    // Output state variables at each integration node associated with the "nominal" computation solveIVP(yu0,q,...)
        Vector_bc   b;       // Output residuals of boundary conditions associated with the "nominal" computation solveIVP(yu0,q,...)
        segmentedData segmentation;
  };

  /**
  * @brief Compute  jacobian matrices using the IVP finite differences method
  * 
  * @tparam opt Option for chosing which elements to compute
  * @param[in] q Actuation variables [beta_i, alpha_i] with beta the translations [m] and alpha the rotations [rad] 
  * @param[in] yu0 Guess of unknown initial conditions : [mx0, my0, u1z0, u2z0, u3z0] (sum of bending moments along x and y axis, and torsion in each tube, at arc length s=0)
  * @param[in] tubes Struct containing tube parameters
  * 
  * @param[out] out Struct for function output
  * @return int (0 indicates success and < 0 indicates failure )
  */
  int ComputeIvpJacobianMatrices( 
    const Vector_q &q,
    const Vector_yu0 &yu0,
    const tubeParameters &tubes,
    const Vector_w &w,
    const computationOptions &opt,

    ComputeIvpJacMatOut &out);

  template <COMPUTATION_OPTION opt> 
  int ComputeIvpJacobianMatrices( 
    const Vector_q &q,
    const Vector_yu0 &yu0,
    const tubeParameters &tubes,
    const Vector_w &w,
    uint nThread,

    ComputeIvpJacMatOut &out);

  int ComputeSingleIVP( 
    const Vector_q &q,
    const Vector_yu0 &yu0,
    const tubeParameters &tubes,
    const Vector_w &w,
    const computationOptions &opt,

    SingleIVPOut &out);
}
#endif //IVP_FD_H
