#include "se3Operations.h"
#include "Eigen/Dense"

namespace CtrLib
{

Eigen::Matrix<double, 6, 1> se3Vee(const Eigen::Matrix4d& se3mat)
{
    Eigen::Matrix<double, 6, 1> xi;

    // Translation part
    xi.segment<3>(0) = se3mat.block<3,1>(0,3);
    
    // Rotation part (from skew-symmetric 3x3)
    // Approximating using (R-Rt) = 2w^
    xi(3) = 0.5 * (se3mat(2,1) - se3mat(1,2));
    xi(4) = 0.5 * (se3mat(0,2) - se3mat(2,0));
    xi(5) = 0.5 * (se3mat(1,0) - se3mat(0,1));

    return xi;
}

} // namespace CtrLib
