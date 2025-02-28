#include <iostream>
#include <stdexcept>

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

      tubes = { .Ux  = p.Ux, // precurvature along x-axis
                .l   = p.l, // tube length
                .l_k = p.l_k, // precurved length
                .Kxy = p.E.cwiseProduct(I), // bending stiffness
                .Kz  =   G.cwiseProduct(J)}; // torsionnal stiffness
                
      offset = p.offset;
      offset(seqN(NB_TUBES, NB_TUBES)) *= pi / 180.0; // convert rotational offset from degree to radian

      // Set arbitrary initial configuration close to the "fully deployed" configuration with a small margin between each tube base
      constexpr double margin = 5e-3; 
      q = offset + Vector_q(-3.0 * margin, -2.0 * margin, -margin, 0.0, 0.0, 0.0);

      w << p.force , p.moment;  // external wrench applied at the end-effector
      yu0 = Vector<double, NB_YU0>::Zero();

      // Compute the model for the initial configuration to initialize yu0, X, J and C.
      if(Compute(q, opt_LOAD_J_C) < 0){
        throw std::runtime_error("CtrModel constructor failed ! Compute() returned non-zero !");
      }
  }

  CtrModel::~CtrModel(){
    //dtor
  }
  
  int CtrModel::Compute(const Vector_q &argQ, const computationOptions &opt){

    if(!opt.isExternalLoads){
      throw std::invalid_argument("CtrModel::Compute() >> Error ! Unloaded model is not implemented yet");
    }
    if(opt.isComputeCompliance && !opt.isExternalLoads){
      throw std::invalid_argument("CtrModel::Compute() >> Error ! Computation of the compliance matrix requires to use the loaded model !");
    }

      int nbIteration = 0;
      q = argQ;
      Vector_yu0 yu0_tilde = yu0;

      ComputeIvpJacMatOut out;

      int nIter = 0;
      
      // first iteration without J and/or C, just to compute the BC residuals and the Bu matrix
      if(ComputeIvpJacobianMatrices(q,yu0_tilde,tubes,w, opt, out) != 0){
        std::cout << "CtrModel::Compute()>> ComputeIvpJacobianMatrices() returned non-zero !" << std::endl;
        return -1;
      }
      yu0_tilde -= pinv(out.Bu) * out.b; // update guess using Gauss-Newton
      nIter++;
      nbIteration++;
      do{ // first pass with yu0 from previous computation
        if(ComputeIvpJacobianMatrices(q, yu0_tilde, tubes, w, opt, out) != 0){
          std::cout << "CtrModel::Compute()>> ComputeIvpJacobianMatrices() returned non-zero !" << std::endl;
          return -1;
        }
        yu0_tilde -= pinv(out.Bu) * out.b; // update guess using Gauss-Newton
        nIter++;
        nbIteration++;
      }while(out.b.norm() > maxNormB && nIter < nIterationsMax);
      if(out.b.hasNaN() || out.b.norm() > maxNormB){ // if the first pass fails
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
          if(ComputeIvpJacobianMatrices(q, yu0_tilde, tubes, w, opt, out) != 0){
            std::cout << "CtrModel::Compute()>> ComputeIvpJacobianMatrices() returned non-zero !" << std::endl;
            return -1;
          }
          yu0_tilde -= pinv(out.Bu) * out.b; // update guess using Gauss-Newton
          nIter++;
          nbIteration++;
        }while(out.b.norm() > maxNormB && nIter < nIterationsMax);
      }
      if(out.b.hasNaN() || out.b.norm() > maxNormB){ // if the second pass fail
        std::cout << "CtrModel::Compute()>> Model failed to converge even when switching sign of torsions !" << std::endl;
        return -1;
      }
      
      // The model converged sucessfully : Update member variables 

      yu0 = yu0_tilde;

      segmentedData segmented_out;
      if(segmenting(q,tubes,segmented_out)!=0){
        std::cout << "CtrModel::Compute()>> segmenting returned non-zero !" << std::endl;
        return -1;
      }
      segmented = segmented_out;

      yTot = out.yTot;
      P = getPFromYtot(out.yTot,segmented);
      R = getRFromYtot(out.yTot,segmented);

      Matrix<double,6,6> RR = Matrix<double,6,6>::Zero();
      RR(seq(0,2),seq(0,2)) = R;
      RR(seq(3,5),seq(3,5)) = R;
      if(opt.isComputeJacobian){
        J = RR * (out.Eq - out.Eu * pinv(out.Bu) * out.Bq); 
      }
      if(opt.isComputeCompliance){
        C = RR * (out.Ew - out.Eu * pinv(out.Bu) * out.Bw);
      }

    return nbIteration;
  }
}