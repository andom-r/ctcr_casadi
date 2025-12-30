#include <iostream>
#include "Eigen/Dense"
#include "odeCtr.h"
#include "ctrConstants.h"

using namespace Eigen;

namespace CtrLib{
  Matrix<double, NB_STATE_VAR , NB_INTEGRATION_NODES> odeIntCtrRK4(
    Vector<double, NB_STATE_VAR> y0,
    double s0,
    double sf,
    const Vector<double, NB_TUBES>& Kxy,
    const Vector<double, NB_TUBES>& Kz,
    const Vector<double, NB_TUBES>& Ux,
    const Vector_w& w){
      
    Matrix<double,NB_STATE_VAR, NB_INTEGRATION_NODES> y;
    y.col(0) = y0;

    double ds = (sf - s0) / (NB_INTEGRATION_NODES-1);
    double half_ds =  ds / 2;
    double sixth_ds = ds / 6;
    double L = sf - s0;
    constexpr int Nm1 = NB_INTEGRATION_NODES - 1;

    //Classic 4th-order Runge-Kutta method 
    Vector<double, NB_STATE_VAR> k0, k1, k2, k3;
    double s = s0;
    
    for(int i = 0; i < Nm1; i++){
      odeCtr(s, y0, Kxy, Kz, Ux, w, k0);
      y0 += k0 * half_ds;
      s += half_ds;
      odeCtr(s, y0, Kxy, Kz, Ux, w, k1);
      y0 = y.col(i) + k1 * half_ds;
      odeCtr(s, y0, Kxy, Kz, Ux, w, k2);
      s = s0 + (L * (i + 1)) / Nm1;
      y0 = y.col(i) + k2 * ds;
      odeCtr(s, y0, Kxy, Kz, Ux, w, k3);

      y0 = y.col(i) + (k0 + 2 * (k1 + k2) + k3) * sixth_ds;
      y.col(i + 1) = y0;
    }
    return y;
  }
}