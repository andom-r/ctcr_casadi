#include <iostream>
#include "Eigen/Dense"
#include "CtrModel.h"
#include "ComputeIvpJacobianMatrices.h"
#include "pinv.h"

#include "segmenting.h"
#include "solveIVP.h"
#include "loadParameters.h"

#include <chrono>
#include <omp.h>

#define rt_printf(x) printf(x)

using namespace Eigen;

namespace CtrLib{
  constexpr int nIterationsMax = 1000;
  constexpr double maxNormB = 1e-10;

  CtrModel::CtrModel(const parameters &p){

      const Vector<double, NB_TUBES> G = p.E.cwiseQuotient(2 * (p.mu + Vector<double,3>::Ones())); // shear modulus
      const Vector<double, NB_TUBES> I = (pi/4) * (p.rOut.array().pow(4) - p.rIn.array().pow(4)); // 2nd moment of inertia
      const Vector<double, NB_TUBES> J = 2.0 * I;   // polar moment of inertia

      Kxy = p.E.cwiseProduct(I); // bending stiffness
      Kz  =   G.cwiseProduct(J); // torsionnal stiffness

      Ux  = p.Ux; // precurvature along x-axis
      l   = p.l; // tube length
      l_k = p.l_k; // precurved length

      offset = p.offset;
      offset(seqN(NB_TUBES, NB_TUBES)) *= 180.0 * pi; // convert rotational offset from degree to radian

      // Set arbitrary initial configuration close to the "fully deployed" configuration with a small margin between each tube base
      constexpr double margin = 5e-3; 
      q = offset + Vector_q(-3.0 * margin, -2.0 * margin, -margin, 0.0, 0.0, 0.0);

      w << p.force , p.moment;  // external wrench applied at the end-effector
      yu0 = Vector<double, NB_YU0>::Zero();

      Compute(q, opt_LOAD_J_C); // Compute the model for the initial configuration to initialize yu0, X, J and C.
  }

  CtrModel::~CtrModel(){
    //dtor
  }
  
  int CtrModel::Compute(const Vector_q &argQ, const computationOptions &opt){

    if(!opt.isExternalLoads){
      std::cout << "CtrModel::Compute() >> Error ! Unloaded model is not implemented yet" << std::endl;
      abort();
    }
    if(opt.isComputeCompliance && !opt.isExternalLoads){
      std::cout << "CtrModel::Compute() >> Error ! Computation of the compliance matrix requires to use the loaded model !" << std::endl;
      abort();
    }

      int nbIteration = 0;
      q = argQ;
      Vector_yu0 yu0_tilde = yu0;

      Matrix_yTot yTot_out;
      Vector_bc b_out;
      Matrix_Eq Eq_out;
      Matrix_Eu Eu_out;
      Matrix_Ew Ew_out;
      Matrix_Bq Bq_out;
      Matrix_Bu Bu_out;
      Matrix_Bw Bw_out;

      int nIter = 0;
      
      // first iteration without J and/or C, just to compute the BC residuals and the Bu matrix
      if(ComputeIvpJacobianMatrices(q,yu0_tilde,Kxy,Kz,Ux,l,l_k,w,yTot_out,b_out,Eq_out,Eu_out,Ew_out,Bq_out,Bu_out,Bw_out, opt) != 0){
        std::cout << "CtrModel::Compute()>> ComputeIvpJacobianMatrices() returned non-zero !" << std::endl;
        return -1;
      }
      yu0_tilde -= pinv(Bu_out) * b_out; // update guess using Gauss-Newton
      nIter++;
      nbIteration++;
      do{ // first pass with yu0 from previous computation
        if(ComputeIvpJacobianMatrices(q,yu0_tilde,Kxy,Kz,Ux,l,l_k,w,yTot_out,b_out,Eq_out,Eu_out,Ew_out,Bq_out,Bu_out,Bw_out, opt) != 0){
          std::cout << "CtrModel::Compute()>> ComputeIvpJacobianMatrices() returned non-zero !" << std::endl;
          return -1;
        }
        yu0_tilde -= pinv(Bu_out) * b_out; // update guess using Gauss-Newton
        nIter++;
        nbIteration++;
      }while(b_out.norm() > maxNormB && nIter < nIterationsMax);
      if(b_out.hasNaN() || b_out.norm() > maxNormB){ // if the first pass fails
        std::cout << "CtrModel::Compute()>> Model failed to converge after " << nbIteration << " iterations."
          "Potential bifurcation detected, switching sign of initial guess for torsions and trying again." << std::endl;
        //switch sign of torsion but keep x,y curvature
        yu0_tilde(0) = yu0(0);
        yu0_tilde(1) = yu0(1);
        yu0_tilde(2) = -yu0(2);
        yu0_tilde(3) = -yu0(3);
        yu0_tilde(4) = -yu0(4);
        nIter = 0;
        do{ //second pass, with opposite torsion
          if(ComputeIvpJacobianMatrices(q,yu0_tilde,Kxy,Kz,Ux,l,l_k,w,yTot_out,b_out,Eq_out,Eu_out,Ew_out,Bq_out,Bu_out,Bw_out, opt) != 0){
            std::cout << "CtrModel::Compute()>> ComputeIvpJacobianMatrices() returned non-zero !" << std::endl;
            return -1;
          }
          yu0_tilde -= pinv(Bu_out) * b_out; // update guess using Gauss-Newton
          nIter++;
          nbIteration++;
        }while(b_out.norm() > maxNormB && nIter < nIterationsMax);
      }
      if(b_out.hasNaN() || b_out.norm() > maxNormB){ // if the second pass fail
        std::cout << "CtrModel::Compute()>> Model failed to converge even when switching sign of torsions !" << std::endl;
        return -1;
      }
      
      // The model converged sucessfully : Update member variables 

      yu0 = yu0_tilde;

      segmentedData segmented_out;
      if(segmenting(q,Kxy,Kz,Ux,l,l_k,segmented_out)!=0){
        std::cout << "CtrModel::Compute()>> segmenting returned non-zero !" << std::endl;
        return -1;
      }
      segmented = segmented_out;

      yTot = yTot_out;
      P = getPFromYtot(yTot_out,segmented_out);
      R = getRFromYtot(yTot_out,segmented_out);

      Matrix<double,6,6> RR = Matrix<double,6,6>::Zero();
      RR(seq(0,2),seq(0,2)) = R;
      RR(seq(3,5),seq(3,5)) = R;
      if(opt.isComputeJacobian){
        J = RR * (Eq_out - Eu_out * pinv(Bu_out) * Bq_out); 
      }
      if(opt.isComputeCompliance){
        C = RR * (Ew_out - Eu_out * pinv(Bu_out) * Bw_out);
      }

    return nbIteration;
  }
}