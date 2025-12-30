#ifndef  PLOT_CTR_H
#define  PLOT_CTR_H
#include "Eigen/Dense"
#include <gnuplot-iostream.h>
#include "ctrConstants.h"

namespace CtrLib{
  extern Gnuplot multiPlot; // declare public
  void plotCtr(
    const Matrix_yTot& y,
    const Eigen::Vector<int, NB_TUBES>& iEnd,
    const Eigen::MatrixXd& xScatter);

    void plotCtrWithTargetTrajectory(
    const Matrix_yTot& y,
    const Eigen::Vector<int, NB_TUBES>& iEnd,
    const Eigen::MatrixXd& xScatter,
    const Eigen::MatrixXd& xTrajScatter,
    const Eigen::MatrixXd& prediction = Eigen::MatrixXd());

    gnuplotio::PlotGroup subPlotCtr(
    const Matrix_yTot& y,
    const Eigen::Vector<int, NB_TUBES>& iEnd,
    const Eigen::MatrixXd& xScatter,
    int armIndex,
  gnuplotio::PlotGroup* plotsPtr = nullptr );// optional pointer


    // class MultiArmPlot
    // {
    //   public:
    //       MultiArmPlot(const int numberOfArms);
    //       virtual ~MultiArmPlot();

    //       void plot( const Matrix_yTot& y,
    //         const Eigen::Vector<int, NB_TUBES>& iEnd,
    //         int armIndex
    //       ); 

    //   protected:
    //       Eigen::MatrixXd plotPoints; 
    //   private:
    // };
}
#endif // PLOT_CTR_H
