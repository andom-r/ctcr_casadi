#include "mathOperations.h"
#include "Eigen/Dense"

namespace CtrLib
{

Eigen::Matrix4d finiteDifference(
    const Eigen::Matrix4d& gInit,
    const Eigen::Matrix4d& gDx,
    double dx)
{
    Eigen::Matrix4d gInv = gInit.inverse();

    // Compute finite difference
    Eigen::Matrix4d temp = (gInv * (gDx - gInit)) / dx;

    return temp;
}

} // namespace CtrLib
