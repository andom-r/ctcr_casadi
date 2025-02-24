#include <iostream>
#include <chrono>
#include <cmath>
#include "ctrConstants.h"
#include "CtrModel.h"
#include "ComputeIvpJacobianMatrices.h"

using namespace std::chrono;
using namespace Eigen;
using namespace CTR_CONST;

template<COMPUTATION_OPTION opt> int timingIVP(int nSteps, int nMaxThread);

int main(int, char**){
  // Example 3 : Computation rate for the IVP 

  int nMaxThread = omp_get_max_threads();
  std::cout << "This computer has " << nMaxThread << " threads. Computing IVP frequency using 1 to " << nMaxThread << " threads." << std::endl;

  constexpr int nSteps = 100000; // Number of IVP computations

  int nMaxThread_LOAD     = 6;
  int nMaxThread_LOAD_J   = 12;
  int nMaxThread_LOAD_J_C = 18;

  timingIVP<LOAD>(nSteps, std::min(nMaxThread,nMaxThread_LOAD));
  timingIVP<LOAD_J>(nSteps, std::min(nMaxThread,nMaxThread_LOAD_J));
  timingIVP<LOAD_J_C>(nSteps, std::min(nMaxThread,nMaxThread_LOAD_J_C));

  return 0;
}
template<COMPUTATION_OPTION opt>
int timingIVP(int nSteps, int nMaxThread){

  // Load parameters corresponding to CTR
  std::vector<parameters> vParameters;
	if(loadParameters("../parameters/parameters.csv",vParameters) != 0){
    return -1;
  }
  parameters &pNominal =  vParameters[0];

  CtrModel ctr(pNominal);
  // Declare actuation variables q
  Vector<double,NB_Q> q;
  q << -0.3, -0.2, -0.1, 0, 0, 0; // arbitrary initial configuration

  VectorXd a1_v(nSteps);
  for (int i = 0; i < nSteps; i++){
    a1_v(i) = i * (2*pi) / (nSteps-1);
  }

  Vector<double,NB_YU0> yu0 = ctr.GetYu0();
  Vector<double,n> Kxy      = ctr.GetKxy();
  Vector<double,n> Kz       = ctr.GetKz();
  Vector<double,n> Ux       = ctr.GetUx();
  Vector<double,n> l        = ctr.GetL();
  Vector<double,n> l_k      = ctr.GetL_k();
  Vector_w w = Vector_w::Zero();

  Matrix<double,nStateVar,nSegMax*nIntPoints> yTot_out;
  Vector<double,NB_BC> b_out;
  Matrix<double,6,CTR_CONST::NB_Q> Eq_out;
  Matrix<double,6,CTR_CONST::NB_YU0> Eu_out;
  Matrix<double,6,CTR_CONST::NB_W> Ew_out;
  Matrix<double,CTR_CONST::NB_BC,CTR_CONST::NB_Q> Bq_out;
  Matrix<double,CTR_CONST::NB_BC,CTR_CONST::NB_YU0> Bu_out;
  Matrix<double,CTR_CONST::NB_BC,CTR_CONST::NB_W> Bw_out;

  for(int j = 0; j < nMaxThread; j++){
    int nThread = j + 1;
    auto start = high_resolution_clock::now();
    for (int i = 0; i < nSteps; i++){
      q(3) = a1_v(i); // rotate inner tube (tube 1)
      if(ComputeIvpJacobianMatrices<opt>(q,yu0,Kxy,Kz,Ux,l,l_k,w,yTot_out,b_out,Eq_out,Eu_out,Ew_out,Bq_out,Bu_out,Bw_out, nThread) != 0){
        std::cout << "CtrModel::Compute()>> ComputeIvpJacobianMatrices() returned non-zero !" << std::endl;
        return -1;
      }
    }
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    std::string strOpt;
    if(opt == LOAD){
      strOpt = "LOAD      ";
    }
    else if(opt == LOAD_J){
      strOpt = "LOAD_J    ";
    }
    else if(opt == LOAD_J_C){
      strOpt = "LOAD_J_C  ";
    }
    else{
      strOpt = "unknown option";
    }
    std::cout << nSteps << " computations of IVP (" << nThread << " threads, "<< strOpt << ") took : " << 1e-6 * duration.count() << " seconds (" << 1e3*(double)nSteps / (double)duration.count() << " kHz)." << std::endl;
  }
  return 0;
}
template int timingIVP<LOAD>(int nSteps, int nMaxThread);
template int timingIVP<LOAD_J>(int nSteps, int nMaxThread);
template int timingIVP<LOAD_J_C>(int nSteps, int nMaxThread);