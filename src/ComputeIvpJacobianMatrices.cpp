#include <iostream>
#include "Eigen/Dense"
#include "ComputeIvpJacobianMatrices.h"
#include "segmenting.h"
#include "solveIVP.h"
#include "bcError.h"
#include "pinv.h"
#include "ctrConstants.h"

#include <omp.h>

using namespace Eigen;
using namespace CTR_CONST;


int ComputeIvpJacobianMatrices( 
  const Vector<double,NB_Q> &q,
  const Vector<double,NB_YU0> &yu0,
  const Vector<double,n> &Kxy,
  const Vector<double,n> &Kz,
  const Vector<double,n> &Ux,
  const Vector<double,n> &l,
  const Vector<double,n> &l_k,
  const Vector_w &w,

  Matrix<double,nStateVar,nSegMax*nIntPoints> &yTot_out,
  Vector<double,NB_BC> &b_out,
  Matrix<double,6,NB_Q> &Eq_out,
  Matrix<double,6,NB_YU0> &Eu_out,
  Matrix<double,6,NB_W> &Ew_out,
  Matrix<double,NB_BC,NB_Q> &Bq_out,
  Matrix<double,NB_BC,NB_YU0> &Bu_out,
  Matrix<double,NB_BC,NB_W> &Bw_out,
  computationOptions opt){

  if(opt.isExternalLoads && !opt.isComputeJacobian && !opt.isComputeCompliance){
    return ComputeIvpJacobianMatrices<LOAD>(q, yu0,Kxy,Kz,Ux, l, l_k,w, yTot_out, b_out, Eq_out, Eu_out, Ew_out, Bq_out, Bu_out, Bw_out, opt.nbThreads);
  }
  else if(opt.isExternalLoads && opt.isComputeJacobian && !opt.isComputeCompliance){
    return ComputeIvpJacobianMatrices<LOAD_J>(q, yu0,Kxy,Kz,Ux, l, l_k,w, yTot_out, b_out, Eq_out, Eu_out, Ew_out, Bq_out, Bu_out, Bw_out, opt.nbThreads);
  }
  else if(opt.isExternalLoads && opt.isComputeJacobian && opt.isComputeCompliance){
    return ComputeIvpJacobianMatrices<LOAD_J_C>(q, yu0,Kxy,Kz,Ux, l, l_k,w, yTot_out, b_out, Eq_out, Eu_out, Ew_out, Bq_out, Bu_out, Bw_out, opt.nbThreads);
  }
  else{
    std::cout << "ComputeIvpJacobianMatrices() >> Error ! Unsupported combination of computation options !" << std::endl;
    abort();
  }

  }

template <COMPUTATION_OPTION opt> 
int ComputeIvpJacobianMatrices( 
  const Vector<double,NB_Q> &q,
  const Vector<double,NB_YU0> &yu0,
  const Vector<double,n> &Kxy,
  const Vector<double,n> &Kz,
  const Vector<double,n> &Ux,
  const Vector<double,n> &l,
  const Vector<double,n> &l_k,
  const Vector_w &w,

  Matrix<double,nStateVar,nSegMax*nIntPoints> &yTot_out,
  Vector<double,NB_BC> &b_out,
  Matrix<double,6,NB_Q> &Eq_out,
  Matrix<double,6,NB_YU0> &Eu_out,
  Matrix<double,6,NB_W> &Ew_out,
  Matrix<double,NB_BC,NB_Q> &Bq_out,
  Matrix<double,NB_BC,NB_YU0> &Bu_out,
  Matrix<double,NB_BC,NB_W> &Bw_out,
  uint nThread)
{
  Matrix<double,18,NB_Q> arrQ;
  Matrix<double,18,NB_YU0> arrYu0;
  Matrix<double,18,NB_W> arrW;
  Vector<Matrix<double,4,4>,18> arrG;
  Matrix<double,18,NB_BC> arrB;

  // i = 0 : nominal configuration
  arrQ(0,all) = q;
  arrYu0(0,all) = yu0;
  arrW(0,all) = w;
  
  // 1 <= i <= 5 : yu0 finite diff
  constexpr double epsilon_yu0 = 1e-3;
  for(int i = 0; i < NB_YU0; i++){
    arrQ(1 + i,all) = q;  // nominal value for q
    arrW(1 + i,all) = w;  // nominal value for w
    Vector<double,NB_YU0> yu0i = yu0;
    yu0i(i) = yu0(i) + epsilon_yu0;
    arrYu0(1 + i,all) = yu0i;
  } 
  
  constexpr double epsilon_q = 1e-3;
  if(opt == LOAD_J || opt == LOAD_J_C){
    // 6 <= i <= 11 : q finite diff
    for(int i = 0; i < NB_Q; i++){
      arrYu0(6 + i,all) = yu0; // nominal value for yu0
      arrW(6 + i,all) = w;  // nominal value for w
      Vector<double,NB_Q> qi = q;
      qi(i) = q(i) + epsilon_q;
      arrQ(6 + i,all) = qi;
    }
  }
  
  constexpr double epsilon_f = 1e-3;
  constexpr double epsilon_l = 1e-4;
  if(opt == LOAD_J_C){
    // 12 <= i <= 14 : f finite diff
    for(int i = 0; i < 3; i++){
      arrYu0(12 + i,all) = yu0; // nominal value for yu0
      arrQ(12 + i,all) = q;    // nominal value for q
      Vector<double,NB_W> wi = w;
      wi(i) = w(i) + epsilon_f;
      arrW(12 + i,all) = wi;
      //std::cout << "i = " << i << " -> arrW[" << 12+i << "] = " << 
    }
    for(int i = 3; i < 6; i++){
      arrYu0(12 + i,all) = yu0; // nominal value for yu0
      arrQ(12 + i,all) = q;    // nominal value for q
      Vector<double,NB_W> wi = w;
      wi(i) = w(i) + epsilon_l;
      arrW(12 + i,all) = wi;
    }
  }

  int iMax; // Required number of IVP computations
  switch(opt){
    case LOAD:
      iMax = 6;
      break;
    case LOAD_J:
      iMax = 12;
      break;
    case LOAD_J_C:
      iMax = 18;
      break;
    default:
      abort();
      break;
  }
  #pragma omp parallel for num_threads(nThread)
  for(int i = 0; i < iMax; i++){
      Vector<double,NB_Q> q_i   = arrQ(i,all);
      Vector<double,NB_YU0> yu0_i = arrYu0(i,all);
      Vector<double,NB_W> w_i = arrW(i,all);

      segmentedData segmented_out;
      if(segmenting(q_i,Kxy,Kz,Ux,l,l_k,segmented_out)!=0){
        std::cout << "ComputeIvpJacobianMatrices()>> segmenting returned non-zero !" << std::endl;
        //return -1;
      }
      Matrix<double,nStateVar,nSegMax*nIntPoints> yTot;
      solveIVP(yu0_i, q_i, segmented_out, w_i, yTot);
      if(i == 0){
        yTot_out = yTot;
      }
      Vector<double,NB_BC> b = bcError(yTot, segmented_out.iEnd, Kxy(0), Kz(0), Ux(0), w_i);

      Vector3d X = getXFromYtot(yTot,segmented_out);
      Vector<double,9> Rv = getRvFromYtot(yTot,segmented_out);
      arrG(i)(seq(0,2),seq(0,2)) = Rv.reshaped(3,3);
      arrG(i)(seq(0,2),3) = X;
      arrG(i)(3,seq(0,2)) = Vector3d::Zero();
      arrG(i)(3,3) = 1.0;
      arrB(i,all) = b;
  }

  b_out = arrB(0,all);

  Matrix<double,4,4> g0 = arrG(0);
  Vector3d X0 = g0(seq(0,2),3);
  Matrix3d R0 = g0(seq(0,2),seq(0,2));

  Matrix<double,4,4> g0_inv;
  g0_inv(seq(0,2),seq(0,2)) = R0.transpose();
  g0_inv(seq(0,2),3) = -R0.transpose() * X0;
  g0_inv(3,seq(0,2)) = Vector3d::Zero();
  g0_inv(3,3) = 1.0;

  // Eu,Bu
  for (int i = 0; i < NB_YU0; i++){

    Matrix<double,4,4> gi = arrG(1 +i);
    Matrix4d temp = (g0_inv * (gi - g0)) / epsilon_yu0;
    // Convert from se(3) to R^6 (apply vee operator)
    Eu_out(seq(0,2),i) = temp(seq(0,2),3);                                   
    Eu_out(3,i) = 0.5 * (temp(2,1) - temp(1,2));
    Eu_out(4,i) = 0.5 * (temp(0,2) - temp(2,0));
    Eu_out(5,i) = 0.5 * (temp(1,0) - temp(0,1));

    Vector<double,NB_BC> bi = arrB(1 + i,all);

    Bu_out(all,i) = (bi-b_out) / epsilon_yu0;
  }
  if(opt == LOAD_J || opt == LOAD_J_C){
    for (int i = 0; i < NB_Q; i++){
      Matrix<double,4,4> gi = arrG(6 + i);
      Matrix4d temp = (g0_inv * (gi - g0)) / epsilon_q;
      // Convert from se(3) to R^6 (apply vee operator)
      Eq_out(seq(0,2),i) = temp(seq(0,2),3);
      Eq_out(3,i) = 0.5 * (temp(2,1) - temp(1,2));
      Eq_out(4,i) = 0.5 * (temp(0,2) - temp(2,0));
      Eq_out(5,i) = 0.5 * (temp(1,0) - temp(0,1));

      Vector<double,NB_BC> bi = arrB(6 + i,all);
      Bq_out(all,i) = (bi-b_out) / epsilon_q;
    }
  }
  if(opt == LOAD_J_C){
    for (int i = 0; i < 3; i++){
      Matrix<double,4,4> gi = arrG(12 + i);
      Matrix4d temp = (g0_inv * (gi - g0)) / epsilon_f;
      // Convert from se(3) to R^6 (apply vee operator)
      Ew_out(seq(0,2),i) = temp(seq(0,2),3);
      Ew_out(3,i) = 0.5 * (temp(2,1) - temp(1,2));
      Ew_out(4,i) = 0.5 * (temp(0,2) - temp(2,0));
      Ew_out(5,i) = 0.5 * (temp(1,0) - temp(0,1));

      Vector<double,NB_BC> bi = arrB(12 + i,all);
      Bw_out(all,i) = (bi-b_out) / epsilon_f;
    }
    for (int i = 3; i < 6; i++){
      Matrix<double,4,4> gi = arrG(12 + i);
      Matrix4d temp = (g0_inv * (gi - g0)) / epsilon_l;
      // Convert from se(3) to R^6 (apply vee operator)
      Ew_out(seq(0,2),i) = temp(seq(0,2),3);
      Ew_out(3,i) = 0.5 * (temp(2,1) - temp(1,2));
      Ew_out(4,i) = 0.5 * (temp(0,2) - temp(2,0));
      Ew_out(5,i) = 0.5 * (temp(1,0) - temp(0,1));

      Vector<double,NB_BC> bi = arrB(12 + i,all);
      Bw_out(all,i) = (bi-b_out) / epsilon_l;
    }
  }

  return 0;
}

template int ComputeIvpJacobianMatrices<LOAD>( 
  const Vector<double,NB_Q> &q,
  const Vector<double,NB_YU0> &yu0,
  const Vector<double,n> &Kxy,
  const Vector<double,n> &Kz,
  const Vector<double,n> &Ux,
  const Vector<double,n> &l,
  const Vector<double,n> &l_k,
  const Vector_w &w,

  Matrix<double,nStateVar,nSegMax*nIntPoints> &yTot_out,
  Vector<double,NB_BC> &b_out,
  Matrix<double,6,NB_Q> &Eq_out,
  Matrix<double,6,NB_YU0> &Eu_out,
  Matrix<double,6,NB_W> &Ew_out,
  Matrix<double,NB_BC,NB_Q> &Bq_out,
  Matrix<double,NB_BC,NB_YU0> &Bu_out,
  Matrix<double,NB_BC,NB_W> &Bw_out,
  uint nThread);

template int ComputeIvpJacobianMatrices<LOAD_J>( 
  const Vector<double,NB_Q> &q,
  const Vector<double,NB_YU0> &yu0,
  const Vector<double,n> &Kxy,
  const Vector<double,n> &Kz,
  const Vector<double,n> &Ux,
  const Vector<double,n> &l,
  const Vector<double,n> &l_k,
  const Vector_w &w,

  Matrix<double,nStateVar,nSegMax*nIntPoints> &yTot_out,
  Vector<double,NB_BC> &b_out,
  Matrix<double,6,NB_Q> &Eq_out,
  Matrix<double,6,NB_YU0> &Eu_out,
  Matrix<double,6,NB_W> &Ew_out,
  Matrix<double,NB_BC,NB_Q> &Bq_out,
  Matrix<double,NB_BC,NB_YU0> &Bu_out,
  Matrix<double,NB_BC,NB_W> &Bw_out,
  uint nThread);

template int ComputeIvpJacobianMatrices<LOAD_J_C>( 
  const Vector<double,NB_Q> &q,
  const Vector<double,NB_YU0> &yu0,
  const Vector<double,n> &Kxy,
  const Vector<double,n> &Kz,
  const Vector<double,n> &Ux,
  const Vector<double,n> &l,
  const Vector<double,n> &l_k,
  const Vector_w &w,

  Matrix<double,nStateVar,nSegMax*nIntPoints> &yTot_out,
  Vector<double,NB_BC> &b_out,
  Matrix<double,6,NB_Q> &Eq_out,
  Matrix<double,6,NB_YU0> &Eu_out,
  Matrix<double,6,NB_W> &Ew_out,
  Matrix<double,NB_BC,NB_Q> &Bq_out,
  Matrix<double,NB_BC,NB_YU0> &Bu_out,
  Matrix<double,NB_BC,NB_W> &Bw_out,
  uint nThread);