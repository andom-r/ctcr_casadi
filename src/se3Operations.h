#pragma once
#include "Eigen/Dense"

namespace CtrLib
{
/**
 * @brief Apply the vee operator to an se(3) matrix.
 * 
 * Converts a 4x4 matrix from the Lie algebra se(3)
 * into a 6x1 vector in R^6:
 * [v; ω] where:
 *  - v is the translation part (top-right 3x1 column)
 *  - ω is the rotation part (extracted from the skew-symmetric top-left 3x3)
 * 
 * Approximating using (R-Rt)v = 2w since R = exp(w) => R ~ I + w^ ...
 * 
 * @param se3mat 4x4 matrix in se(3)
 * @return Eigen::Matrix<double, 6, 1> 6D vector [v; ω]
 */
Eigen::Matrix<double, 6, 1> se3Vee(const Eigen::Matrix4d& se3mat);

} // namespace CtrLib
