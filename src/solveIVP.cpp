#include <iostream>
#include "Eigen/Dense"
#include "odeIntCtrRK4.h"
#include "ctrConstants.h"

using namespace Eigen;

namespace CtrLib{
  void solveIVP(  
    const Vector_yu0    &yu0,
    const Vector_q      &q,
    const segmentedData &segmented,
    const Vector_w      &w,

    Matrix_yTot         &yTot_out)
  {

    // Unpack segmentedData struct for readability

    const Vector_S         &S    =  segmented.S;
    const Matrix_segmented &KKxy =  segmented.KKxy;
    const Matrix_segmented &KKz  =  segmented.KKz;
    const Matrix_segmented &UUx  =  segmented.UUx;
    const int               nSeg =  segmented.iEnd(0) + 1;

    
    // For better convergence, the two first elements of the initial guess (yu0(0) and yu0(1)) are the sum of bending moments along x and y axis (at s=0)
    // But the ODEs are implemented using curvature along x and y axis as state variables
    double mx_0 = yu0(0);
    double my_0 = yu0(1);
      
    double u1z_0 = yu0(2);
    double u2z_0 = yu0(3);
    double u3z_0 = yu0(4);

    Vector<double, NB_TUBES> alpha,beta;
    beta  = q(Eigen::seqN(0, NB_TUBES));
    alpha = q(Eigen::seqN(NB_TUBES, NB_TUBES));

    Vector<double, NB_SEGMENT_MAX + 1> span; span << 0,S;
    Vector3d r0 = Vector3d::Zero();

    double alpha1_0 = alpha(0) - beta(0) * u1z_0;
    double alpha2_0 = alpha(1) - beta(1) * u2z_0;
    double alpha3_0 = alpha(2) - beta(2) * u3z_0;

    double ca0 = cos(alpha1_0);
    double sa0 = sin(alpha1_0);

    Vector<double,9> R0;
    R0 << ca0, -sa0, 0, sa0, ca0, 0, 0, 0, 1;

    double t2_0 = alpha2_0 - alpha1_0;
    double t3_0 = alpha3_0 - alpha1_0;

    // Recompute curvature along x and y axis (at s=0) from the sum of bending moments along x and y axis (at s=0)
    double K = KKxy(all,0).sum();
    double sum_x = KKxy(0,0)*UUx(0,0) + cos(t2_0)*KKxy(1,0)*UUx(1,0) + cos(t3_0)*KKxy(2,0)*UUx(2,0);
    double sum_y =                      sin(t2_0)*KKxy(1,0)*UUx(1,0) + sin(t3_0)*KKxy(2,0)*UUx(2,0); 
    double u1x_0 = (mx_0 + sum_x) / K;
    double u1y_0 = (my_0 + sum_y) / K;
      
    Vector3d u1_0;
    u1_0 << u1x_0, u1y_0, u1z_0;

    for(int seg = 0; seg < nSeg; seg++){  // For each segment
      // Construct vector of initial state variables
      Eigen::Vector<double, NB_STATE_VAR> y0;
      y0 << r0 , R0 , u1_0 , u2z_0 , u3z_0, t2_0, t3_0;
      // Get start and end arc-length
      double s0 = span(seg);
      double sL = span(seg+1);

      // Get stiffness and precurvature for the current segment
      Vector<double, NB_TUBES> Kxys = KKxy(all,seg);
      Vector<double, NB_TUBES> Kzs = KKz(all,seg);
      Vector<double, NB_TUBES> Uxs = UUx(all,seg);
      // Forward integration for the current segment
      Matrix<double, NB_STATE_VAR, NB_INTEGRATION_NODES> ySeg = odeIntCtrRK4(y0, s0, sL, Kxys,Kzs,Uxs,w);
      // Append the result of integration to the matrix containing the result of the whole CTR
      yTot_out(all,Eigen::seqN(seg * NB_INTEGRATION_NODES,NB_INTEGRATION_NODES)) = ySeg;
      // If there is a next segment, compute next initial conditions regarding
      // continuity equations
      if (seg < nSeg-1){
        r0 << ySeg({0,1,2}, NB_INTEGRATION_NODES - 1); // r- = r+
        R0 << ySeg(Eigen::seqN(3,9), NB_INTEGRATION_NODES - 1); // R- = R+
        // Potential discontinuity for u1x & u1y : enforce that the sum of internal bending moment before the discontinuity equals the sum of internal bending moments after the discontinuity.
        double t2 = ySeg(17, NB_INTEGRATION_NODES - 1); 
        double t3 = ySeg(18, NB_INTEGRATION_NODES - 1);
        double s2 = sin(t2);
        double c2 = cos(t2);
        double s3 = sin(t3);
        double c3 = cos(t3);
        Matrix2d R2; R2 << c2,-s2,s2,c2;
        Matrix2d R3; R3 << c3,-s3,s3,c3;

        double u1x = ySeg(12, NB_INTEGRATION_NODES - 1);
        double u1y = ySeg(13, NB_INTEGRATION_NODES - 1);
        Vector2d u1; u1 << u1x,u1y;
        Vector2d u2 = R2.transpose() * u1;
        Vector2d u3 = R3.transpose() * u1;

        Vector2d U1; U1 << Uxs(0),0;
        Vector2d U2; U2 << Uxs(1),0;
        Vector2d U3; U3 << Uxs(2),0;

        Vector2d delta_u1 = u1 - U1;
        Vector2d delta_u2 = u2 - U2;
        Vector2d delta_u3 = u3 - U3;

        double k1xy = Kxys(0);
        double k2xy = Kxys(1);
        double k3xy = Kxys(2);

        Matrix2d k1; k1 << k1xy, 0, 0, k1xy;
        Matrix2d k2; k2 << k2xy, 0, 0, k2xy;
        Matrix2d k3; k3 << k3xy, 0, 0, k3xy;

        Vector2d sumMomentsBefore = k1 * delta_u1 + R2*k2*delta_u2 + R3*k3*delta_u3;

        // Bending stiffness after the discontinuity

        k1xy = KKxy(0,seg+1);
        k2xy = KKxy(1,seg+1);
        k3xy = KKxy(2,seg+1);

        k1 << k1xy, 0, 0, k1xy;
        k2 << k2xy, 0, 0, k2xy;
        k3 << k3xy, 0, 0, k3xy;

        Matrix2d K = k1 + k2 + k3;

        U1 << UUx(0,seg+1), 0;
        U2 << UUx(1,seg+1), 0;
        U3 << UUx(2,seg+1), 0;

        // Sum of R_i * k_i * U_i after discontinuity
        Vector2d KU = k1 * U1 + R2*k2*U2 + R3*k3*U3;

        Matrix2d invK; invK << 1/K(0,0), 0, 0, 1/K(1,1);
        u1 = invK * (sumMomentsBefore + KU);

        u1_0(0) = u1(0);
        u1_0(1) = u1(1);
        // u_i,z- = u_i,z+
        u1_0(2) = ySeg(14, NB_INTEGRATION_NODES - 1);
        u2z_0   = ySeg(15, NB_INTEGRATION_NODES - 1);
        u3z_0   = ySeg(16, NB_INTEGRATION_NODES - 1);
        // theta_i- = theta_i+
        t2_0    = ySeg(17, NB_INTEGRATION_NODES - 1);
        t3_0    = ySeg(18, NB_INTEGRATION_NODES - 1);
      }
    }
  }
}