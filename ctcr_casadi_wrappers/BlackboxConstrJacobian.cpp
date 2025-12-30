#include "BlackboxConstrJacobian.hpp"

// using Jacobian for model compute
std::vector<casadi::DM> BlackboxConstrJacobian::eval(const std::vector<casadi::DM>& in) const {
    if(numberOfModelComputes % 100000 == 0){
        std::cout << "Call for Eval *-*-*-*-*-*-*-*-*-* " << numberOfModelComputes << std::endl;
    }
    numberOfModelComputes = numberOfModelComputes + 1 ;
    // Convert CasADi DM to std::vector<double> to Eigen
    std::vector<double> u_std = in[0].nonzeros();
    Eigen::VectorXd u = Eigen::Map<const Eigen::VectorXd>(u_std.data(), static_cast<Eigen::Index>(u_std.size()));

    // Linearization point
    Eigen::VectorXd u0 = ctrModel.GetQ();   // previous input
    Eigen::VectorXd p0 = ctrModel.GetP();     // position at u0
    Eigen::MatrixXd J  = ctrModel.GetJ().topRows(3);     // Jacobian at u0

    // Linearized forward model:
    // p(u) = p0 + J * (u - u0)
    Eigen::VectorXd p_lin = p0 + J * (u - u0);
    // if(numberOfModelComputes % 100000 == 0){
    //     std::cout << "Call for Eval *-*-*-*-*-*-* command  " << u_std << " u0: " << u0.transpose()<< std::endl;
    //     std::cout << "Call for Eval *-*-*-*-*-*-* position  " << p_lin.transpose() <<" p0: " << p0.transpose()<< std::endl;
    // }

    std::vector<double> predictedV(p_lin.data(), p_lin.data() + p_lin.size());
    // Convert back to DM for CasADi
    std::vector<casadi::DM> out(1);
    out[0] = casadi::DM(predictedV);
    return out;
}