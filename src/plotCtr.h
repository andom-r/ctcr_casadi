#ifndef  PLOT_CTR_H
#define  PLOT_CTR_H

#include "Eigen/Dense"
#include "ctrConstants.h"

namespace CtrLib{
  void plotCtr(
    const Matrix_yTot& y,
    const Eigen::Vector<int, NB_TUBES>& iEnd,
    const Eigen::MatrixXd& xScatter);
}
#endif // PLOT_CTR_H
