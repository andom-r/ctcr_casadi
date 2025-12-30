#pragma once
#include "Eigen/Dense"
/**
 * @file MathOperations.hpp
 * 
 * Provides utility functions for basic SE(3) math operations.
 */

namespace CtrLib
{

/**
 * @brief Compute the finite difference of homogeneous transformations.
 * 
 * Computes:
 *      finiteDifference = (g_inv * (g(x+dx) - g(x))) / dx
 * 
 * @param g0 Initial 4x4 homogeneous transformation (SE(3))
 * @param gi Target 4x4 homogeneous transformation (SE(3))
 * @param dx Small scalar for finite difference
 * @return Eigen::Matrix4d Resulting 4x4 matrix
 */
Eigen::Matrix4d finiteDifference(
    const Eigen::Matrix4d& gInit,
    const Eigen::Matrix4d& gDx,
    double dx);

} // namespace CtrLib
