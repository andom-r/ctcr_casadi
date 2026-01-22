#include <iostream>
#include <thread>
#include <chrono>

#include "loadParameters.h"
#include "CtrModel.h"
#include "pinv.h"
#include "plotCtr.h"
#include <casadi/casadi.hpp>
#include "BlackboxConstrYu0.hpp"


using namespace std::chrono;
using namespace Eigen;
using namespace CtrLib;

int test_casadi_mpc();
// generate sequence of points between two points
std::vector<Eigen::Vector3d> generatePoints( const Eigen::Vector3d& P, const Eigen::Vector3d& Pdes, double dl);
// Eigen vector of 3d points ----> std::vector of 3d points
std::vector<std::vector<double>> eigenToStdNestedVector(const Eigen::Vector3d& matPoints);
casadi::DM buildXdHorizon(const std::vector<Eigen::Vector3d>& points, int start_idx, int N, int nx);
Eigen::Vector3d safePoint(const std::vector<Eigen::Vector3d>& points, int idx);

int nx = NB_X - 3;
int bcNu = NB_BC;
int nu = NB_Q;
int NPred  = 2;
int unkNu = NB_YU0;

int main(int, char **) {

  // Example: Control scenario for a straight line
  // (here the same CtrModel is used for both the simulation of the robot and the model running in the controller)
  auto t_start = std::chrono::high_resolution_clock::now();
  
  test_casadi_mpc();

  // ---- Print total runtime ----
  auto t_end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> total_time = t_end - t_start;

  std::cout << "\n=============================\n";
  std::cout << "Total execution time: " << total_time.count() << " seconds\n";
  std::cout << "=============================\n\n";

  std::cout << "Press <ENTER> to exit. (This is to keep the plot interactive.)" << std::endl;
  std::cin.get();

  return 0;
}

int test_casadi_mpc() {
    // ---------------- Part I : Initialize CTR model ----------------
    std::vector<parameters> vParameters;
    if (loadParameters("../parameters/parameters.csv", vParameters) != 0 || vParameters.empty()) {
        return -1;
    }
    
    parameters& pNominal = vParameters[0];
    CtrLib::CtrModel ctr(pNominal);
    // return 0;
    std::cout << "initial unkown" << ctr.GetYu0()<< std::endl;
    
    Vector_q q(-0.1, -0.1, -0.1, 0, 0, 0);
    Vector_q _q(-0.3, -0.2, -0.1, 0, 0, 0);
    // imp imp todo... set mode applied wrench to zero...
    if (ctr.Compute(_q, opt_LOAD) < 0) {
        std::cout << "Error: ctr.Compute() failed\n";
        return -1;
    }

    Vector_yu0 initialYu0 = ctr.GetYu0(); 

    // ---------------- Part II : Initialize control simulation ----------------
    Eigen::Vector3d P = ctr.GetP();
    Eigen::Vector3d Pdes = P + Eigen::Vector3d(20e-3,0 , 0);

    constexpr double dl = 5e-4;
    std::vector<Eigen::Vector3d> points = generatePoints(P, Pdes, dl);
    if (points.size() < 2) {
        points = {P, Pdes};
    }

    // Prediction horizen
    const int N = NPred;

    // Constraint settings
    const double betweenTubeBaseMargin = 1e-5;
    const double inputTranslationVariationLimit = 1e-3;
    const double inputRotationVariationLimit    = 1e-2;
    const double dxLimit    = dl*30.3;
    const double xErrorLimit    = 1e-1;
    const double boundaryConditionsMaxNorm = 1e-8;
    const double initUnknowVarBoxDim = 1e-2;

    // gu layout per prediction step
    // //   0: -u0 + u1
    // //   1: -u1 + u2
    //   2..(2+nu-1): delta_u(j)
    const int nbTranUConst = nu ;

    // ---------------- Plot init ----------------
    plotCtrWithTargetTrajectory(ctr.GetYTot(), ctr.segmented.iEnd, P.transpose(), P.transpose());
    std::this_thread::sleep_for(std::chrono::seconds(1));

    constexpr int logSize = 6000;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> logTraj(logSize, 3);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> targetTraj(logSize, 3);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> pointSolveTime(logSize, 2);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> inputsLog(logSize, nu);
    using Eigen::all;
    using Eigen::seq;

    logTraj(0, all) = P;
    targetTraj(0, all) = P;
    // ---------------- CasADi symbols ----------------
    casadi::MX X  = casadi::MX::sym("X", nx, N + 1);
    casadi::MX U  = casadi::MX::sym("U", nu, N);
    casadi::MX initUnknownVar  = casadi::MX::sym("unk", unkNu, N); // initial unknown variables u1x, u1y, u1z, u2z, u3z

    casadi::MX x0 = casadi::MX::sym("x0", nx);
    casadi::MX Xd = casadi::MX::sym("Xd", nx, N);   // reference horizon
    casadi::MX uPrev = casadi::MX::sym("uPrev", nu);

    // Cost matrices
    casadi::MX Q = 1e2 * casadi::MX::eye(nx);

    const double multi = 5e-1;
    casadi::MX R1 = multi * casadi::MX::eye(3);
    R1(0,0) = 0.02*R1(0,0);
    R1(1,1) = 0.009*R1(1,1);
    R1(2,2) = 0.005*R1(2,2);
    casadi::MX R2 = 0.8 * multi * casadi::MX::eye(3);
    casadi::MX R  = casadi::MX::diagcat({R1, R2});

    casadi::MX obj = 0;
    std::vector<casadi::MX> g;   // dynamics + initial condition
    std::vector<casadi::MX> gu;  // scalar inequality constraints per step for inputs
    std::vector<casadi::MX> gBcIneq;  // boundray conditions inequality

    // Initial condition constraint
    g.push_back(X(casadi::Slice(), 0) - x0);

    // Stage costs + gu constraints
    for (int k = 0; k < N; ++k) {
        casadi::MX xk  = X(casadi::Slice(), k + 1);
        casadi::MX xdk = Xd(casadi::Slice(), k);
        casadi::MX uk  = U(casadi::Slice(), k);

        casadi::MX xDiff = (xk - xdk);

        casadi::MX dx= (xk - X(casadi::Slice(), k)); // position rate constraint

        casadi::MX du;
        if (k == 0) du = (uk - uPrev);
        else        du = (uk - U(casadi::Slice(), k - 1));

        casadi::MX xQx = casadi::MX::mtimes(xDiff.T(), casadi::MX::mtimes(Q, xDiff));
        obj += xQx;
        // obj += casadi::MX::exp((xQx / 1e3 + 20)); 
        // obj += casadi::MX::exp(0.001*(xQx + 1000)); 
        obj += casadi::MX::mtimes(uk.T(), casadi::MX::mtimes( R, uk));

        // tube ordering constraints (translation)
        // gu.push_back(-uk(0) + uk(1));
        // gu.push_back(-uk(1) + uk(2));

        // input variation constraints (translation + rotation)
        // for (int j = 0; j < nu; ++j) {
        //     gu.push_back(du(j));
        // }
        gu.push_back(du(0) + du(1) + du(2));
        gu.push_back(du(1) + du(2));
        gu.push_back(du(2));
        for (int j = 3; j < nu; ++j) gu.push_back(du(j));

        // for (int j = 0; j < nx; ++j) {
        //     gu.push_back(dx(j));
        // }

        gu.push_back(casadi::MX::norm_2(dx));
        gu.push_back(casadi::MX::norm_2(xDiff));
        
    }

    // Blackbox for dynamics and boundry conditions constraints 
    BlackboxConstrYu0 f_blackbox("bb_constr", &ctr, nx, nu, bcNu, unkNu, opt_LOAD);
    // BlackboxConstr f_blackbox("bb_constr", &ctr);
    casadi::Function bb_fun = f_blackbox.factory(
        "bb_fun",
        {"u","unk"},
        {"g"},
        casadi::Callback::AuxOut(),
        casadi::Dict{
            {"expand", false},   // keep as call node
            {"enable_fd", true}  // FD derivatives
        }
    );

    for (int k = 0; k < N; ++k) {
        casadi::MX u = U(casadi::Slice(), k);
        casadi::MX _u = casadi::MX::vertcat({
            u(0) + u(1) + u(2),
            u(1) + u(2),
            u(2),
            u(3),// - u(4),
            u(4),// - u(5),
            u(5)
        }); // shape (6,1)
        std::vector<casadi::MX> args = { _u, initUnknownVar(casadi::Slice(), k) };
        casadi::MX funValue = bb_fun(args)[0]; // (nx+bcNu)x1

        // Split: x_next and bc residuals
        casadi::MX x_next = funValue(casadi::Slice(0, nx));                 // nx x 1
        casadi::MX bc     = funValue(casadi::Slice(nx+0, nx + bcNu));         // bcNu x 1

        // Dynamics equality: X_{k+1} - x_next = 0
        g.push_back(X(casadi::Slice(), k + 1) - x_next);

        // Boundary-condition inequality: ||bc||_2 - max <= 0
        gBcIneq.push_back(casadi::MX::norm_2(bc) - boundaryConditionsMaxNorm);
        // gBcIneq.push_back(bc);
    }

    // Decision vector: [vec(X); vec(U)]
    casadi::MX nlp_x = casadi::MX::vertcat({
        casadi::MX::reshape(X, nx * (N + 1), 1),
        casadi::MX::reshape(U, nu * N, 1),
        casadi::MX::reshape(initUnknownVar, unkNu * N, 1)
    });

    // Constraints vector: [gu; g; gBcIneq]
    casadi::MX nlp_g = casadi::MX::vertcat({
        casadi::MX::vertcat(gu),
        casadi::MX::vertcat(g),
        casadi::MX::vertcat(gBcIneq)
    });

    // Parameters vector: p = [x0; vec(Xd); uPrev]
    casadi::MX xdFlat = casadi::MX::reshape(Xd, nx * N, 1);
    casadi::MX nlp_p  = casadi::MX::vertcat({x0, xdFlat, uPrev});

    casadi::MXDict nlp = {{"x", nlp_x}, {"p", nlp_p}, {"f", obj}, {"g", nlp_g}};

    // NLP solver
    casadi::Dict opts;
    opts["expand"] = false;
    opts["ipopt.hessian_approximation"] = "limited-memory";
    opts["ipopt.derivative_test"] = "none";
    opts["ipopt.print_level"] = 0;
    opts["print_time"] = false;
    opts["ipopt.jacobian_approximation"] = "finite-difference-values";
    // opts["ipopt.acceptable_iter"] = 0;
    // opts["ipopt.constr_viol_tol"] = 1e-8;
    // opts["ipopt.acceptable_constr_viol_tol"] = 1e-8;
    // opts["ipopt.bound_relax_factor"] = 0;
    // opts["ipopt.honor_original_bounds"] = "yes";

    casadi::Function solver = casadi::nlpsol("solver", "ipopt", nlp, opts);

    // ---------------- Variable bounds x, u, unk ----------------
    const int n_dec = nx * (N + 1) + nu * N + unkNu * N;
    const int uOffset = nx * (N + 1);
    const int unkVarOffset = nx * (N + 1)+ nu * N;
    casadi::DM lbx = -casadi::DM::inf(n_dec);
    casadi::DM ubx =  casadi::DM::inf(n_dec);

    // Bounds on decision Variables - U(k)
    for (int k = 0; k < N; ++k) {
        const int idx = nx * (N + 1) + k * nu;
        lbx(idx + 0) = -ctr.GetTubeParameters().l(0) + ctr.GetTubeParameters().l(1) + 1 * betweenTubeBaseMargin;
        lbx(idx + 1) = -ctr.GetTubeParameters().l(1) + ctr.GetTubeParameters().l(2) + 1 * betweenTubeBaseMargin;
        lbx(idx + 2) = -ctr.GetTubeParameters().l(2) + 1 * betweenTubeBaseMargin;

        ubx(idx + 0) = -1 * betweenTubeBaseMargin;
        ubx(idx + 1) = -1 * betweenTubeBaseMargin;
        ubx(idx + 2) = -1 * betweenTubeBaseMargin;
    }

    // Bounds on decision Variables - initial unknown variables
    for (int k = unkVarOffset; k < n_dec; ++k) {
        lbx(k) = -initUnknowVarBoxDim;
        ubx(k) = initUnknowVarBoxDim;
    }

    // ---------------- Constraint bounds gu, g, gBcIneq ----------------
    const int ng = gu.size() + g.size() * nx + gBcIneq.size() * 1;
    const int bcGIndex = gu.size() + g.size() * nx ;
    casadi::DM lbg = casadi::DM::zeros(ng);
    casadi::DM ubg = casadi::DM::zeros(ng);

    // gu bounds (first gu block is length N * nbTranUConst)
    for (int i = 0; i < N; ++i) {
        // tube ordering
        // ubg(i * nbTranUConst + 0) = ctr.GetTubeParameters().l(0) - ctr.GetTubeParameters().l(1) - 3 * betweenTubeBaseMargin;
        // lbg(i * nbTranUConst + 0) = 1 * betweenTubeBaseMargin;

        // ubg(i * nbTranUConst + 1) = ctr.GetTubeParameters().l(1) - ctr.GetTubeParameters().l(2) - 3 * betweenTubeBaseMargin;
        // lbg(i * nbTranUConst + 1) = 1 * betweenTubeBaseMargin;

        // delta_u bounds
        const int base = i * nbTranUConst;
        // first 3 - translations
        for (int j = 0; j < 3 ; ++j) {
            ubg(base + j) =  inputTranslationVariationLimit;
            lbg(base + j) = -inputTranslationVariationLimit;
        }
        // remaining - rotations
        for (int j = 3; j < nu; ++j) {
            ubg(base + j) =  inputRotationVariationLimit;
            lbg(base + j) = -inputRotationVariationLimit;
        }

        // for (int j = nu; j < nu+nx; ++j) {
        //     ubg(base + j) =  dxLimit;
        //     lbg(base + j) = dxLimit/5;
        // }

        ubg(base + nu + 0) = dxLimit;
        lbg(base + nu + 0) = 0;             // dxLimit/5 - 1.5;
        ubg(base + nu + 1) = xErrorLimit;
        lbg(base + nu + 1) = 0;
    }

    // bounds on boundray conditions error
    for (int i = bcGIndex; i < ng; ++i) {
        // lbg(i) = -casadi::DM::inf();
        // ubg(i) = 0;
        lbg(i) = -boundaryConditionsMaxNorm;
        ubg(i) = boundaryConditionsMaxNorm;
    }

    std::cout<< "lbg: " << std::endl<< lbg << std::endl;
    std::cout<< "ubg: " << std::endl<< ubg << std::endl;

    // ---------------- Initial guess ----------------
    casadi::DM x_init = casadi::DM::zeros(n_dec);

    // initialize X guess from point path -repeat last point if needed
    for (int i = 0; i < N + 1; ++i) {
        Eigen::Vector3d pi = safePoint(points, i);
        x_init(i * nx + 0) = pi(0);
        x_init(i * nx + 1) = pi(1);
        x_init(i * nx + 2) = pi(2);
    }

    // initialize U guess from current configuration q
    for (int i = 0; i < nu * N; ++i) {
        x_init(uOffset + i) = q(i % nu);
    }

    // initialize unkVar guess after model cumputes
    for (int i = 0; i < unkNu * N; ++i) {
        x_init(unkVarOffset + i) = initialYu0(i % unkNu);
    }

    // ---------------- Solve loop ----------------
    casadi::DM Pc; // current predicted point used as new x0
    casadi::DM Uc; // current applied control used as new uPrev
    casadi::DM UnkC; // current applied unknown variables Yu0

    long plotIteration = 0;
    constexpr double epsilon = 1e-5;
    constexpr int maxAttemptsPerPoint = 1; //

    // initial p for solver paramters
    std::vector<double> p0_std(P.data(), P.data() + P.size());
    std::vector<double> uPrev_std(q.data(), q.data() + q.size());

    int jStart = 1;
    casadi::DM xdH = buildXdHorizon(points, jStart, N, nx);

    std::map<std::string, casadi::DM> arg;
    arg["x0"]  = x_init;
    arg["lbx"] = lbx;
    arg["ubx"] = ubx;
    arg["lbg"] = lbg;
    arg["ubg"] = ubg;
    arg["p"]   = casadi::DM::vertcat({casadi::DM(p0_std), xdH, casadi::DM(uPrev_std)});

    double sequareErrorSumPos = 0.0;   // sum of squared errors (||P - Pdes||^2)
    long   sampleCount  = 0;            // number of samples accumulated
    
    std::ofstream csv("traj_log_yu0.csv");              // or std::ios::app to append
    csv << "iter;Px;Py;Pz;Target x;Target y;Target z; Start solving time ms; Duration ms; d0; d1; d3; phi0; phi1; phi2\n";
    csv << std::setprecision(16);                   // keep numeric precision

    auto t0Start = std::chrono::high_resolution_clock::now();
    auto tStart = std::chrono::high_resolution_clock::now();
    auto tEnd = std::chrono::high_resolution_clock::now();
    pointSolveTime(0,0) = 0;
    pointSolveTime(0,1) = 0;
    inputsLog(0,all) = q;

    // Iterate over reference points
    for (int j = 1; j < points.size() - 1; ++j) {
        Eigen::Vector3d pDesEigen = safePoint(points, j + 1);

        for (int attempt = 0; attempt < maxAttemptsPerPoint; ++attempt) {
            if ((pDesEigen - P).norm() <= epsilon) break;
            tStart = std::chrono::high_resolution_clock::now();

            auto res = solver(arg);
            
            tEnd = std::chrono::high_resolution_clock::now();
            double tStartMs = std::chrono::duration<double, std::milli>( tStart - t0Start ).count();
            auto stepSolveTimeMs = std::chrono::duration<double, std::milli>(tEnd - tStart).count();
            pointSolveTime(plotIteration + 1,0) = tStartMs;
            pointSolveTime(plotIteration + 1,1) = stepSolveTimeMs;

            casadi::DM sol = res.at("x");

            const int jok = 1; // <= N
            casadi::DM PSolAll = sol(casadi::Slice(0, (N + 1) * nx));
            casadi::DM USolAll = sol(casadi::Slice(uOffset, uOffset + N * nu));
            casadi::DM UnkSolAll = sol(casadi::Slice(unkVarOffset, unkVarOffset + N * unkNu));

            Pc = sol(casadi::Slice(nx * jok, (jok + 1) * nx));
            Uc = sol(casadi::Slice(uOffset + (jok - 1) * nu, uOffset + jok * nu));
            UnkC = sol(casadi::Slice(unkVarOffset + (jok - 1) * unkNu, unkVarOffset + jok * unkNu));

            // shift X, U, Unk
            for (int i = 0; i < nx * N; ++i) {
                x_init(i) = PSolAll(i + nx); // shift by one
                if (i >= nx * (N - 1)) {
                    x_init(i + nx) = PSolAll(i + nx); // keep last
                }
            }
            for (int i = 0; i < nu * N; ++i) {
                x_init(uOffset + i) = USolAll(i);
            }
            for (int i = 0; i < unkNu * N; ++i) {
                x_init(unkVarOffset + i) = UnkSolAll(i);
            }

            // update robot state using applied Uc
            std::vector<double> UcV = Uc.nonzeros();
            std::vector<double> UnkV = UnkC.nonzeros();
            // Eigen::Map<Eigen::VectorXd> UcEigen(UcV.data(), UcV.size());
            // recover originl inputs
            Eigen::VectorXd UcEigen(nu);
            UcEigen << UcV[0] + UcV[1] + UcV[2],
                    UcV[1] + UcV[2],
                    UcV[2], 
                    UcV[3],// - UcV[4] ,
                    UcV[4],// - UcV[5],
                    UcV[5];
            Eigen::Map<Eigen::VectorXd> UnkEigen(UnkV.data(), UnkV.size());
            SingleIVPOut tmp;

            // ctr.Compute(UcEigen, opt_LOAD_J);
            ctr.ComputeIVP(UcEigen, UnkEigen, opt_LOAD, tmp);
            // ctr.Compute(UcEigen, opt_LOAD);
            P = ctr.GetP();

            // update logs/plots
            logTraj(plotIteration + 1, all) = P;
            targetTraj(plotIteration + 1, all) = pDesEigen;
            inputsLog(plotIteration + 1,all) = UcEigen;
            
            const Eigen::Vector3d e = (P - pDesEigen);
            sequareErrorSumPos += e.squaredNorm();
            ++sampleCount;

            // optional horizon display
            std::vector<double> PcAllVector = PSolAll.nonzeros();
            Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
                PcAllEigenMap(PcAllVector.data(), N + 1, nx);

            plotCtrWithTargetTrajectory(
                ctr.GetYTot(),
                ctr.segmented.iEnd,
                logTraj(seq(0, plotIteration + 1), all),
                targetTraj(seq(0, plotIteration + 1), all),
                PcAllEigenMap
            );

            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }

        ++plotIteration;

        // update horizon reference Xd for next step
        xdH = buildXdHorizon(points, j + 1, N, nx);

        // Update solver parameters
        arg["p"]  = casadi::DM::vertcat({Pc, xdH, Uc});
        arg["x0"] = x_init;
    }

    if (sampleCount > 0) {
        const double rmse = std::sqrt(sequareErrorSumPos / sampleCount);
        std::cout << "RMSE (position): " << rmse *1000 << " mm" <<"\n";
        std::cout << "Sample count: " << sampleCount <<"\n";
    }

    for(int i = 0; i< plotIteration;i++){
        csv << (i) << ";"
        << logTraj(i,0) << ";" << logTraj(i,1) << ";" << logTraj(i,2) << ";"
        << targetTraj(i,0) << ";" << targetTraj(i,1) << ";" << targetTraj(i,2)  << ";"
        << pointSolveTime(i,0) <<";"<< pointSolveTime(i,1)  << ";"
        << inputsLog(i,0) << ";" << inputsLog(i,1) << ";" << inputsLog(i,2)  << ";"
        << inputsLog(i,3) << ";" << inputsLog(i,4) << ";" << inputsLog(i,5)  << ";"
        << "\n";
    }
    csv.close();

    std::cout << "Final target:\n" << Pdes.transpose() << "\n";
    return 0;
}


// generate a line segment between two points
std::vector<Eigen::Vector3d> generatePoints(const Eigen::Vector3d& P,
                                           const Eigen::Vector3d& Pdes,
                                           double dl) {
    std::vector<Eigen::Vector3d> points;
    Eigen::Vector3d dir = Pdes - P;
    const double totalDist = dir.norm();

    if (totalDist < 1e-12) {
        points.push_back(P);
        return points;
    }

    const Eigen::Vector3d unitDir = dir / totalDist;
    const int nSteps = static_cast<int>(std::floor(totalDist / dl));
    points.reserve(nSteps + 2);

    for (int i = 0; i <= nSteps; ++i) {
        points.push_back(P + i * dl * unitDir);
    }
    if ((points.back() - Pdes).norm() > 1e-12) {
        points.push_back(Pdes);
    }
    return points;
}


casadi::DM buildXdHorizon(const std::vector<Eigen::Vector3d>& points,
                          int start_idx, int N, int nx) {
    casadi::DM xd = casadi::DM::zeros(nx * N, 1);
    const int last = points.size() - 1;

    for (int k = 0; k < N; ++k) {
        int idx = start_idx + k;
        if (idx > last) idx = last;
        xd(nx*k + 0) = points[idx][0];
        xd(nx*k + 1) = points[idx][1];
        xd(nx*k + 2) = points[idx][2];
    }
    return xd;
}

Eigen::Vector3d safePoint(const std::vector<Eigen::Vector3d>& points, int idx) {
    if (points.empty()) return Eigen::Vector3d::Zero();
    if (idx < 0) idx = 0;
    if (idx >= static_cast<int>(points.size())) idx = static_cast<int>(points.size()) - 1;
    return points[idx];
}