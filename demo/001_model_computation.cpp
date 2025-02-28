#include <iostream>

#include "loadParameters.h"
#include "CtrModel.h"

using namespace Eigen;
using namespace CtrLib;

int main(int, char**){
  // Example 1 : simply compute the model

  // Load parameters corresponding to CTR
  std::vector<parameters> vParameters;
	if(loadParameters("../parameters/parameters.csv",vParameters) != 0){
    return -1;
  }
  parameters &pNominal =  vParameters[0];

  CtrModel ctr(pNominal);
  // Declare actuation variables q
  Vector_q q(-0.3, -0.2, -0.1, 0, 0, 0); // arbitrary initial configuration

  // Compute model
  // Use predefined options (defined in "ctrConstants.h") :
  //    - opt_LOAD      to compute the loaded model
  //    - opt_LOAD_J    to compute the loaded model and the robot Jacobian matrix
  //    - opt_LOAD_J_C  to compute the loaded model and the robot Jacobian and compliance matrices
  ctr.Compute(q, opt_LOAD);
  
  // Or create user-defined option (e.g. to adjust the number of threads for parallel computing)
  computationOptions opt = {.isExternalLoads = true, 
                            .isComputeJacobian = false, 
                            .isComputeCompliance = false, 
                            .nbThreads = 4};
  ctr.Compute(q, opt);

  // Get end-effector position
  Vector3d P = ctr.GetP();
  std::cout << "P = [" << P.transpose() << "]" << std::endl;
  return 0;
}
