#include <iostream>
#include <thread>
#include <chrono>

#include "loadParameters.h"
#include "CtrModel.h"
#include "pinv.h"
#include "plotCtr.h"

using namespace std::chrono;
using namespace Eigen;
using namespace CtrLib;

int main(int, char**){
  // Example 2 : Simple control scenario for a straight line
  //  (here the same CtrModel is used for both the simulation of the robot and the model running in the controller)
  
  ///// Part I : Initialize CTR model
  // load parameters corresponding to CTR
  std::vector<parameters> vParameters;
	if(loadParameters("../parameters/parameters.csv",vParameters) != 0){
    return -1;
  }
  parameters &pNominal =  vParameters[0];

  CtrModel ctr(pNominal);
  // declare actuation variables q
  Vector_q q(-0.3, -0.2, -0.1, 0.0, 0.0, 0.0); // arbitrary initial configuration
  // Compute model and robot Jacobian matrix
  if(ctr.Compute(q, opt_LOAD_J) < 0){
      std::cout << "main()>> Error ! ctr.Compute() returned non-zero " << std::endl;
      return -1;
    }
  

  ///// Part II : Initialize control simulation
  // Get end-effector position
  Vector3d P = ctr.GetP();
  // Set desired position as the current position +20 mm towards the +x direction and +20 mm towards the +y direction
  Vector3d Pdes = P + Vector3d(20e-3, 20e-3, 0.0); 

  constexpr double dt = 0.025; // Time sampling for the simulation : 40 Hz / 25 ms
  // Control parameters from the parameter file
  double lambda = pNominal.lambda;
  double w0_t = pNominal.w(0);
  double w0_r = pNominal.w(1);
  double w1_t = pNominal.w(2);
  double w1_r = pNominal.w(3);

  Matrix<double,6,6> W0 = Matrix<double,6,6>(Vector<double,6>(w0_t, w0_t, w0_t, w0_r, w0_r, w0_r).asDiagonal());
  Matrix<double,6,6> W1 = Matrix<double,6,6>(Vector<double,6>(w1_t, w1_t, w1_t, w1_r, w1_r, w1_r).asDiagonal());

  constexpr double epsilon = 0.1e-3; // Run the control simulation until the robot tip position is at most 0.1 mm away from the desired position
    
  // Initialize plot
  plotCtr(ctr.GetYTot(), ctr.segmented.iEnd, P.transpose());
  std::this_thread::sleep_for(1s); // wait for the plot to initialize

  constexpr int logSize = 100; //should be enough
  // trajectory log to display the simulated trajectory
  Matrix<double,logSize,3> logTraj = Matrix<double,logSize,3>::Zero();
  int i = 0;

  ///// Part III : Simulation loop
  while((Pdes - P).norm() > epsilon){
    // Simple example without actuation constraints & obstacle avoidance (just v0,v1 & not v2,v3)

    // Determine velocity for each task
    Vector_X X_prime_des; // Desired end-effector velocity
    X_prime_des(seqN(0,3)) = lambda * (Pdes - P) / dt;    // Tracking in translation
    X_prime_des(seqN(3,3)) = Vector<double,3>::Zero();  // No tracking in rotation here
    Vector_X v0 = X_prime_des;
    //Vector_q v1 = Vector<double,6>::Zero(); // v1 = 0 for damping (implicit)

    // Compute actuation speed qp_d using Generalized-Damped-Least-Squares method
    Matrix_J J = ctr.GetJ();
    Matrix<double,6,6> temp = J.transpose() * W0 * J + W1; // pinv(expression) without temp variable won't compile
    Vector_q qp_d = pinv(temp) * (J.transpose() * W0 * v0);

    // Simulate CTR
    q += qp_d * dt;
    if(ctr.Compute(q, opt_LOAD_J) < 0){
      std::cout << "main()>> Error ! ctr.Compute() returned non-zero " << std::endl;
      return -1;
    }
    P = ctr.GetP();
    
    // Log and plot
    logTraj(i,all) = P;
    plotCtr(ctr.GetYTot(), ctr.segmented.iEnd, logTraj(seq(0,i),all));
    i++;

    std::this_thread::sleep_for(50ms); // tempo for the animation
  }
  std::cout << "Press <ENTER> to exit. (This is to keep the plot interactive.)" << std::endl;
  std::cin.get();
  return 0;
}