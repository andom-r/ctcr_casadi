#include <iostream>
#include "Eigen/Dense"
#define EIGEN_CORE_H
#include <gnuplot-iostream.h>
#include "ctrConstants.h"

using namespace Eigen;

namespace CtrLib{
Gnuplot multiPlot;  // file-scoped static
void plotCtr(const Matrix_yTot& y, const Vector<int,3>& iEnd, const MatrixXd& xScatter){
  // static Gnuplot multiPlot;
  static bool init = false;
  if(!init){
    init = true;
    multiPlot << "set terminal qt" << std::endl;
    multiPlot << "set xrange [-0.1:0.1]" << std::endl;
    multiPlot << "set yrange [-0.1:0.1]" << std::endl;
    multiPlot << "set zrange [0:0.25]" << std::endl;
    multiPlot << "set xlabel 'x (m)'" << std::endl;
    multiPlot << "set ylabel 'y (m)'" << std::endl;
    multiPlot << "set zlabel 'z (m)'" << std::endl;
    multiPlot << "set view equal xyz" << std::endl;
    multiPlot << "set grid xtics ytics ztics" << std::endl;
    multiPlot << "set ticslevel 0" << std::endl;
  }
  int sz_end0 = getYtotIndexFromIend(iEnd(0)) + 1;
  int sz_end1 = getYtotIndexFromIend(iEnd(1)) + 1;
  int sz_end2 = getYtotIndexFromIend(iEnd(2)) + 1;
  // std::cout << "init: " << sz_end0 << std::endl;

  MatrixXd r1bis = y.topLeftCorner(NB_TUBES, sz_end0);
  MatrixXd r2bis = y.topLeftCorner(NB_TUBES, sz_end1);
  MatrixXd r3bis = y.topLeftCorner(NB_TUBES, sz_end2);

  MatrixXd r1T = r1bis.transpose();
  MatrixXd r2T = r2bis.transpose();
  MatrixXd r3T = r3bis.transpose();

  auto plots = multiPlot.splotGroup();
  plots.add_plot1d(r1T, "with lines lw 3 linecolor rgb 'red' title 'tube 1'");
  plots.add_plot1d(r2T, "with lines lw 4 linecolor rgb 'blue' title 'tube 2'");
  plots.add_plot1d(r3T, "with lines lw 5 linecolor rgb 'green' title 'tube 3'");
  plots.add_plot1d(xScatter, "with lines lw 1 linecolor rgb 'black' title 'trajectory'");
  multiPlot << plots;
}

void plotCtrWithTargetTrajectory(
  const Matrix_yTot& y, 
  const Vector<int,3>& iEnd, 
  const MatrixXd& xScatter, 
  const MatrixXd& xTrajScatter,
  const MatrixXd& prediction
){
  // static Gnuplot multiPlot;
  static bool init = false;
  if(!init){
    init = true;
    multiPlot << "set terminal qt" << std::endl;
    multiPlot << "set xrange [-0.1:0.1]" << std::endl;
    multiPlot << "set yrange [-0.1:0.1]" << std::endl;
    multiPlot << "set zrange [0:0.25]" << std::endl;
    multiPlot << "set xlabel 'x (m)'" << std::endl;
    multiPlot << "set ylabel 'y (m)'" << std::endl;
    multiPlot << "set zlabel 'z (m)'" << std::endl;
    multiPlot << "set view equal xyz" << std::endl;
    multiPlot << "set grid xtics ytics ztics" << std::endl;
    multiPlot << "set ticslevel 0" << std::endl;
  }
  int sz_end0 = getYtotIndexFromIend(iEnd(0)) + 1;
  int sz_end1 = getYtotIndexFromIend(iEnd(1)) + 1;
  int sz_end2 = getYtotIndexFromIend(iEnd(2)) + 1;
  // std::cout << "init: " << sz_end0 << std::endl;

  MatrixXd r1bis = y.topLeftCorner(NB_TUBES, sz_end0);
  MatrixXd r2bis = y.topLeftCorner(NB_TUBES, sz_end1);
  MatrixXd r3bis = y.topLeftCorner(NB_TUBES, sz_end2);

  MatrixXd r1T = r1bis.transpose();
  MatrixXd r2T = r2bis.transpose();
  MatrixXd r3T = r3bis.transpose();

  auto plots = multiPlot.splotGroup();
  plots.add_plot1d(r1T, "with lines lw 3 linecolor rgb 'red' title 'tube 1'");
  plots.add_plot1d(r2T, "with lines lw 4 linecolor rgb 'blue' title 'tube 2'");
  plots.add_plot1d(r3T, "with lines lw 5 linecolor rgb 'green' title 'tube 3'");
  plots.add_plot1d(xScatter, "with lines lw 1 linecolor rgb 'black' title 'trajectory'");
  // plots.add_plot1d(xTrajScatter, "with lines lw 1 linecolor rgb 'yellow' title 'target trajectory'");

  // Make xTrajScatter dashed and vary colors
  // plots.add_plot1d(xTrajScatter, "with lines lw 2 dashtype 2 linecolor rgb 'orange' title 'target trajectory (orange)'");
  // plots.add_plot1d(xTrajScatter, "with lines lw 2 dashtype 3 linecolor rgb 'purple' title 'target trajectory (purple)'");
  plots.add_plot1d(xTrajScatter, "with lines lw 2 dashtype 4 linecolor rgb 'cyan' title 'target trajectory (cyan)'");
  if (prediction.size() != 0)
    plots.add_plot1d(prediction, "with lines lw 2 dashtype 4 linecolor rgb 'purple' title 'Next N Predction (purple)'");

  multiPlot << plots;
}

gnuplotio::PlotGroup subPlotCtr(
  const Matrix_yTot& y, 
  const Vector<int,3>& iEnd, 
  const MatrixXd& xScatter,
  int armIndex,
  gnuplotio::PlotGroup* plotsPtr = nullptr // optional pointer
){
  // static Gnuplot gp;
  static bool init = false;
  if(!init){
    init = true;
    multiPlot << "set terminal qt" << std::endl;
    multiPlot << "set xrange [-0.1:0.1]" << std::endl;
    multiPlot << "set yrange [-0.1:0.1]" << std::endl;
    multiPlot << "set zrange [0:0.25]" << std::endl;
    multiPlot << "set xlabel 'x (m)'" << std::endl;
    multiPlot << "set ylabel 'y (m)'" << std::endl;
    multiPlot << "set zlabel 'z (m)'" << std::endl;
    multiPlot << "set view equal xyz" << std::endl;
    multiPlot << "set grid xtics ytics ztics" << std::endl;
    multiPlot << "set ticslevel 0" << std::endl;
  }
  int sz_end0 = getYtotIndexFromIend(iEnd(0)) + 1;
  int sz_end1 = getYtotIndexFromIend(iEnd(1)) + 1;
  int sz_end2 = getYtotIndexFromIend(iEnd(2)) + 1;
  // std::cout << "init: " << sz_end0 << std::endl;

  MatrixXd r1bis = y.topLeftCorner(NB_TUBES, sz_end0);
  MatrixXd r2bis = y.topLeftCorner(NB_TUBES, sz_end1);
  MatrixXd r3bis = y.topLeftCorner(NB_TUBES, sz_end2);

  MatrixXd r1T = r1bis.transpose();
  MatrixXd r2T = r2bis.transpose();
  MatrixXd r3T = r3bis.transpose();

  gnuplotio::PlotGroup plots = plotsPtr ? *plotsPtr : multiPlot.splotGroup();
  plots.add_plot1d(r1T, "with lines lw 3 linecolor rgb 'red' title 'arm " + std::to_string(armIndex) + " - tube 1'");
  plots.add_plot1d(r2T, "with lines lw 4 linecolor rgb 'blue' title 'arm " + std::to_string(armIndex) + " - tube 2'");
  plots.add_plot1d(r3T, "with lines lw 5 linecolor rgb 'green' title 'arm " + std::to_string(armIndex) + " - tube 3'");
  plots.add_plot1d(xScatter, "with lines lw 1 linecolor rgb 'black' title 'trajectory'");
  return plots;
}

//   MultiArmPlot::MultiArmPlot(const int numberOfArms){
//       Eigen::MatrixXd<double, numberOfArms * 3, Eigen::Dynamic> v;
//       plotPoints = v;
//   }

//   MultiArmPlot::~MultiArmPlot(){
//     //dtor
//   }
  
//   void MultiArmPlot::plot(const Matrix_yTot& y,
//     const Eigen::Vector<int, NB_TUBES>& iEnd,
//     int armIndex){
//   static Gnuplot gp;
//   static bool init = false;
//   if(!init){
//     init = true;
//     gp << "set terminal qt" << std::endl;
//     gp << "set xrange [-0.1:0.1]" << std::endl;
//     gp << "set yrange [-0.1:0.1]" << std::endl;
//     gp << "set zrange [0:0.25]" << std::endl;
//     gp << "set xlabel 'x (m)'" << std::endl;
//     gp << "set ylabel 'y (m)'" << std::endl;
//     gp << "set zlabel 'z (m)'" << std::endl;
//     gp << "set view equal xyz" << std::endl;
//     gp << "set grid xtics ytics ztics" << std::endl;
//     gp << "set ticslevel 0" << std::endl;
//   }
//   int sz_end0 = getYtotIndexFromIend(iEnd(0)) + 1;
//   int sz_end1 = getYtotIndexFromIend(iEnd(1)) + 1;
//   int sz_end2 = getYtotIndexFromIend(iEnd(2)) + 1;
//   // std::cout << "init: " << sz_end0 << std::endl;

//   MatrixXd r1bis = y.topLeftCorner(NB_TUBES, sz_end0);
//   MatrixXd r2bis = y.topLeftCorner(NB_TUBES, sz_end1);
//   MatrixXd r3bis = y.topLeftCorner(NB_TUBES, sz_end2);

//   MatrixXd r1T = r1bis.transpose();
//   MatrixXd r2T = r2bis.transpose();
//   MatrixXd r3T = r3bis.transpose();

//   plotPoints(seqN(armIndex,3)) = r1bis;
//   auto plots = gp.splotGroup();
//   plots.add_plot1d(r1T, "with lines lw 3 linecolor rgb 'red' title 'tube 1'");
//   plots.add_plot1d(r2T, "with lines lw 4 linecolor rgb 'blue' title 'tube 2'");
//   plots.add_plot1d(r3T, "with lines lw 5 linecolor rgb 'green' title 'tube 3'");
//   plots.add_plot1d(xScatter, "with lines lw 1 linecolor rgb 'black' title 'trajectory'");
//   gp << plots;
// }


}