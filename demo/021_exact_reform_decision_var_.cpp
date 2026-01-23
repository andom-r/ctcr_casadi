#include <iostream>
#include <thread>
#include <chrono>

#include "loadParameters.h"
#include "CtrModel.h"
#include "pinv.h"
#include "plotCtr.h"
#include <casadi/casadi.hpp>
#include <BlackboxConstr.hpp>

using namespace std::chrono;
using namespace Eigen;
using namespace CtrLib;

int test_casadi_mpc();
// generate sequence of points between two points
std::vector<Eigen::Vector3d> generatePoints( const Eigen::Vector3d& P, const Eigen::Vector3d& Pdes, double dl);
// Eigen vector of 3d points ----> std::vector of 3d points
std::vector<std::vector<double>> eigenToStdNestedVector(const Eigen::Vector3d& matPoints);

int nx = NB_X - 3;
int nu = NB_Q;
int NPred  = 2;

struct MpcConfig {
    int N = NPred;
    int nx = 3;
    int nu = 6;
    int nbStageUConstr = nu + 2;// + 2; // input rate limit, 1 pos rate limit, 1 pos error limit  
    
    double dl = 5e-4;
    // double dl = 1e-4;
    double betweenTubeBaseMargin = 1e-5;

    double inputTranslationVariationLimit =2e-3;
    double inputRotationVariationLimit    =1e-2;

    // double inputTranslationVariationLimit =2e-4;
    // double inputRotationVariationLimit    =4e-3;
    double dxLimit        = dl*2000.5;
    double xErrorLimit    = 10e+1;

    double Q_scale = 1e3;
    double R_scale = 0*5e-2;
};

struct MpcProblem {
    std::shared_ptr<BlackboxConstr> blackboxConstr;  // keep callback alive
    casadi::Function bb_fun;             
    casadi::Function solver;

    int n_dec = 0;
    int ng = 0;

    casadi::DM lbx, ubx;
    casadi::DM lbg, ubg;

    // Decision layout helpers
    int xBlockSize(int nx, int N) const { return nx * (N + 1); }
    int uOffset(int nx, int N) const { return nx * (N + 1); }
};

int main(int, char **) {

  // Example : Simple control scenario for a straight line using the exact model with MPC
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

static casadi::MX makeBlockDiagR(int nu, double scale) {
    // TODO... guard for later
    if (nu != 6) {
        return scale * casadi::MX::eye(nu);
    }
    casadi::MX R1 = scale * casadi::MX::eye(3);
    R1(0,0) = -0.00008*R1(0,0);
    R1(1,1) = -0.00005*R1(1,1);
    R1(2,2) = -0.00002*R1(2,2);
    casadi::MX R2 = 2.4 * scale * casadi::MX::eye(3);
    return casadi::MX::diagcat({R1, R2});
}

static casadi::DM buildXdHorizonDM(
    const std::vector<Eigen::Vector3d>& points,
    int startIndex, // horizon begins at points[startIndex]
    int N,
    int nx
) {
    casadi::DM xd = casadi::DM::zeros(nx * N);
    for (int k = 0; k < N; ++k) {
        int idx = std::min(startIndex + k, (int)points.size() - 1);
        xd(nx*k + 0) = points[idx][0];
        xd(nx*k + 1) = points[idx][1];
        xd(nx*k + 2) = points[idx][2];
    }
    return xd;
}

static void applyInputBounds(CtrModel& ctr, const MpcConfig& cfg, MpcProblem& pb) {
    const int nx = cfg.nx;
    const int nu = cfg.nu;
    const int N  = cfg.N;

    for (int k = 0; k < N; ++k) {
        const int idx = pb.uOffset(nx, N) + k * nu;

        // We only bounded translations (0..2)
        if (nu >= 3) {
            pb.lbx(idx + 0) = -ctr.GetTubeParameters().l(0) + ctr.GetTubeParameters().l(1) + 1 * cfg.betweenTubeBaseMargin;
            pb.lbx(idx + 1) = -ctr.GetTubeParameters().l(1) + ctr.GetTubeParameters().l(2) + 1 * cfg.betweenTubeBaseMargin;
            pb.lbx(idx + 2) = -ctr.GetTubeParameters().l(2) + 1 * cfg.betweenTubeBaseMargin;

            pb.ubx(idx + 0) = -cfg.betweenTubeBaseMargin;
            pb.ubx(idx + 1) = -cfg.betweenTubeBaseMargin;
            pb.ubx(idx + 2) = -1 * cfg.betweenTubeBaseMargin;
        }
    }

    std::cout << "lbx: " << pb.lbx << std::endl;
    std::cout << "ubx: " << pb.ubx << std::endl;
}

static void applyStageConstraintsBounds(CtrModel& ctr, const MpcConfig& cfg, MpcProblem& pb) {
    const int N  = cfg.N;
    const int nu = cfg.nu;

    // Per-stage: 2 tube-ordering constraints + nu uDiff constraints
    // pb.nbStageUConstr = 2 + nu;

    for (int i = 0; i < N; ++i) {
        const int stage = i * cfg.nbStageUConstr;

        // uDiff(0..nu-1) variation limits
        for (int j = 0; j < nu; ++j) {
            const double lim = (j < 3) ? cfg.inputTranslationVariationLimit : cfg.inputRotationVariationLimit;
            pb.ubg(stage + j) =  lim;
            pb.lbg(stage + j) = -lim;
        }

        int idx = stage + cfg.nu;
        pb.ubg(idx + 0) =  cfg.dxLimit;
        pb.lbg(idx + 0) = 0;        // dxLimit/5;
        pb.ubg(idx + 1) =  cfg.xErrorLimit;
        pb.lbg(idx + 1) =  0;
    }

    std::cout << "lbg: " << pb.lbg << std::endl;
    std::cout << "ubg: " << pb.ubg << std::endl;
}

static void warmStartFromSolution(
    const casadi::DM& sol,
    const MpcConfig& cfg,
    casadi::DM& x0_guess // size n_dec
) {
    const int nx = cfg.nx;
    const int nu = cfg.nu;
    const int N  = cfg.N;

    const int xSize = nx * (N + 1);
    const int uOff  = xSize;

    // Extract full U and X blocks
    casadi::DM Uall = sol(casadi::Slice(uOff, uOff + N * nu));
    casadi::DM Xall = sol(casadi::Slice(0, xSize));

    // Put U guess = Uall
    for (int i = 0; i < N * nu; ++i) {
        x0_guess(uOff + i) = Uall(i);
    }

    // Shift X guess forward by one step X_guess[k] = X_sol[k+1] for k=0..N-1 and repeat last
    // for (int k = 0; k < N; ++k) {
    //     for (int i = 0; i < nx; ++i) {
    //         x0_guess(k * nx + i) = Xall((k + 1) * nx + i);
    //     }
    // }
    // // Last state guess repeats last state
    // for (int i = 0; i < nx; ++i) {
    //     x0_guess(N * nx + i) = Xall(N * nx + i);
    // }
}

static MpcProblem buildMpcProblem(CtrModel& ctr, const MpcConfig& cfg) {
    const int nx = cfg.nx;
    const int nu = cfg.nu;
    const int N  = cfg.N;

    MpcProblem pb;

    // Variables
    casadi::MX X  = casadi::MX::sym("X", nx, N + 1);
    casadi::MX U  = casadi::MX::sym("U", nu, N);
    casadi::MX x0 = casadi::MX::sym("x0", nx);
    casadi::MX Xd = casadi::MX::sym("Xd", nx, N);
    casadi::MX u0 = casadi::MX::sym("u0", nu);

    // Costs
    casadi::MX Q = cfg.Q_scale * casadi::MX::eye(nx);
    casadi::MX R = makeBlockDiagR(nu, cfg.R_scale);

    casadi::MX obj = 0;
    std::vector<casadi::MX> g;  // g.reserve((N + 1) * cfg.nx);
    std::vector<casadi::MX> gu; // gu.reserve(N * (cfg.nbStageUConstr));

    // Initial condition constraint
    g.push_back(X(casadi::Slice(), 0) - x0);

    // Stage cost + stage constraints on input differences
    for (int k = 0; k < N; ++k) {
        casadi::MX xk  = X(casadi::Slice(), k + 1);
        casadi::MX xdk = Xd(casadi::Slice(), k);
        casadi::MX uk  = U(casadi::Slice(), k);

        casadi::MX dx= (xk - X(casadi::Slice(), k)); // for position rate constraint
        casadi::MX xDiff = (xk - xdk);

        casadi::MX uDiff;
        if (k == 0) uDiff = (uk - u0);
        else        uDiff = uk - (U(casadi::Slice(), k - 1));

        casadi::MX xQX = casadi::MX::mtimes(xDiff.T(), casadi::MX::mtimes(Q, xDiff));

        // objective
        obj += xQX;
        obj += casadi::MX::mtimes(uDiff.T(), casadi::MX::mtimes(R, uDiff));
        // obj += casadi::MX::exp((xQX / 1e3 + 10));
        // obj += -casadi::MX::mtimes(uDiff.T(), casadi::MX::mtimes(0.1 * R, uDiff))
        //      * (1 - casadi::MX::sin(xQX / 1e2) * casadi::MX::sin(xQX / 1e2));

        // Input variation constraints: push each element of uDiff
        // for (int j = 0; j < nu; ++j) gu.push_back(uDiff(j));
        gu.push_back(uDiff(0) + uDiff(1) + uDiff(2));
        gu.push_back(uDiff(1) + uDiff(2));
        gu.push_back(uDiff(2));
        for (int j = 3; j < nu; ++j) gu.push_back(uDiff(j));

        gu.push_back(casadi::MX::norm_2(dx));
        gu.push_back(casadi::MX::norm_2(xDiff));
    }

    // Blackbox dynamics constraints
    // BlackboxConstr f_blackbox("bb_constr", &ctr, nx, nu, opt_LOAD);
    pb.blackboxConstr = std::make_shared<BlackboxConstr>("bb_constr", &ctr, cfg.nx, cfg.nu, opt_LOAD);
    pb.bb_fun = pb.blackboxConstr->factory(
        "bb_fun",
        {"u"},
        {"g"},
        casadi::Callback::AuxOut(),
        casadi::Dict{
            {"expand", false},
            {"enable_fd", true}
        }
    );

    for (int k = 0; k < N; ++k) {
        // std::vector<casadi::MX> args = {U(casadi::Slice(), k)};
        // extract the k‑th column as an MX vector
        casadi::MX u = U(casadi::Slice(), k);

        // casadi::MX args = u;
        // build argument vector
        casadi::MX args = casadi::MX::vertcat({
            u(0) + u(1) + u(2),
            u(1) + u(2),
            u(2),
            u(3),// - u(4),
            u(4),// - u(5),
            u(5)
        }); // shape (6,1)

        g.push_back(X(casadi::Slice(), k + 1) - pb.bb_fun(args)[0]);
    }

    // Decision vector
    casadi::MX nlp_x = casadi::MX::vertcat({
        casadi::MX::reshape(X, nx * (N + 1), 1),
        casadi::MX::reshape(U, nu * N, 1)
    });

    casadi::MX nlp_g = casadi::MX::vertcat({
        casadi::MX::vertcat(gu),
        casadi::MX::vertcat(g)
    });

    // Parameter vector: [x0 ; vec(Xd) ; u0]
    casadi::MX xdFlat = casadi::MX::reshape(Xd, nx * N, 1);
    casadi::MXDict nlp = {
        {"x", nlp_x},
        {"p", casadi::MX::vertcat({x0, xdFlat, u0})},
        {"f", obj},
        {"g", nlp_g}
    };

    // Solver options
    casadi::Dict opts;
    opts["expand"] = false;
    opts["ipopt.hessian_approximation"]         = "limited-memory";
    opts["ipopt.derivative_test"]               = "none";
    opts["ipopt.print_level"]                   = 0;
    opts["print_time"]                          = false;
    opts["ipopt.jacobian_approximation"]        = "finite-difference-values";
    opts["ipopt.acceptable_iter"]               = 0;
    opts["ipopt.constr_viol_tol"]               = 1e-8;
    opts["ipopt.acceptable_constr_viol_tol"]    = 1e-8;
    opts["ipopt.bound_relax_factor"]            = 0;
    opts["ipopt.honor_original_bounds"]         = "yes";

    pb.solver = casadi::nlpsol("solver", "ipopt", nlp, opts);

    pb.n_dec = nx * (N + 1) + nu * N;

    // Bounds
    pb.lbx = -casadi::DM::inf(pb.n_dec);
    pb.ubx =  casadi::DM::inf(pb.n_dec);

    // ng = |gu| + |g|*nx (each g item is nx-dim)
    // pb.nbStageUConstr = 2 + nu;
    pb.ng = (int)gu.size() + (int)g.size() * nx;

    pb.lbg = casadi::DM::zeros(pb.ng);
    pb.ubg = casadi::DM::zeros(pb.ng);

    applyInputBounds(ctr, cfg, pb);
    applyStageConstraintsBounds(ctr, cfg, pb);

    // Dynamics + initial condition are equality constraints (already zeros in lbg/ubg),
    // and we intentionally leave them as 0 == g.

    return pb;
}

int test_casadi_mpc() {
    // ===================== Part I: Initialize CTR model =====================
    std::vector<parameters> vParameters;
    if (loadParameters("../parameters/parameters.csv", vParameters) != 0) return -1;
    parameters& pNominal = vParameters[0];

    CtrModel ctr(pNominal);

    Vector_q _q(-0.3, -0.2, -0.1, 0.0, 0.0, 0.0);
    // Transformed vars
    Vector_q q(-0.1, -0.1, -0.1, 0.0, 0.0, 0.0);
    if (ctr.Compute(_q, opt_LOAD) < 0) {
        std::cout << "Error: ctr.Compute() failed\n";
        return -1;
    }

    // ===================== Part II: Simulation setup =====================
    MpcConfig cfg;

    Eigen::Vector3d P = ctr.GetP();
    Eigen::Vector3d Pdes = P + Eigen::Vector3d(20e-3,0,0);
    // Eigen::Vector3d Pdes = P + Eigen::Vector3d(0,-20e-3,0);
    // Eigen::Vector3d Pdes = P + Eigen::Vector3d(0,-40e-3,0);

    Eigen::Matrix<double, 1, 3> Pr = P.transpose(); // 1x3
    CtrLib::plotCtrWithTargetTrajectory(ctr.GetYTot(), ctr.segmented.iEnd, Pr, Pr);
    std::this_thread::sleep_for(1s);

    constexpr int logSize = 6000;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> logTraj(logSize, 3);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> targetTraj(logSize, 3);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> pointSolveTime(logSize, 2);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> inputsLog(logSize, nu);
    logTraj(0, all)   = P;
    targetTraj(0, all)= P;

    auto points = generatePoints(P, Pdes, cfg.dl);

    // ===================== Build MPC =====================
    MpcProblem pb = buildMpcProblem(ctr, cfg);

    // Initial guess
    casadi::DM x0_guess = casadi::DM::zeros(pb.n_dec);
    const int uOff = pb.uOffset(cfg.nx, cfg.N);

    // U initial guess
    for (int i = 0; i < cfg.nu * cfg.N; ++i) {
        x0_guess(uOff + i) = q(i % cfg.nu);// + std::floor((double)i / cfg.nu) * 5e-2 * q(i % cfg.nu);
    }
    // X initial guess from the straight-line points
    for (int i = 0; i < cfg.N + 1; ++i) {
        x0_guess(i * cfg.nx + 0) = points[i][0];
        x0_guess(i * cfg.nx + 1) = points[i][1];
        x0_guess(i * cfg.nx + 2) = points[i][2];
    }

    // Parameters p = [x0; Xd_flat; u0]
    std::vector<double> x0_std(P.data(), P.data() + P.size());
    std::vector<double> u0_std(q.data(), q.data() + q.size());

    // Horizon desired points starts at points[1]
    casadi::DM xd_dm = buildXdHorizonDM(points, /*startIndex=*/1, cfg.N, cfg.nx);

    std::map<std::string, casadi::DM> arg;
    arg["x0"]  = x0_guess;
    arg["lbx"] = pb.lbx;
    arg["ubx"] = pb.ubx;
    arg["lbg"] = pb.lbg;
    arg["ubg"] = pb.ubg;
    arg["p"]   = casadi::DM::vertcat({
        casadi::DM(x0_std),  // current position (nx)
        xd_dm,               // desired horizon (nx*N)
        casadi::DM(u0_std)   // previous input (nu)
    });

    // logs
    double sequareErrorSumPos = 0.0;   // sum of squared errors (||P - Pdes||^2)
    long   sampleCount  = 0;            // number of samples accumulated

    std::ofstream csv("exact_log.csv");              // or std::ios::app to append
    csv << "iter;Px;Py;Pz;Target x;Target y;Target z; Start solving time ms; Duration ms; d0; d1; d3; phi0; phi1; phi2\n";
    csv << std::setprecision(16);                   // keep numeric precision

    auto t0Start = std::chrono::high_resolution_clock::now();
    auto tStart = std::chrono::high_resolution_clock::now();
    auto tEnd = std::chrono::high_resolution_clock::now();
    pointSolveTime(0,0) = 0;
    pointSolveTime(0,1) = 0;
    inputsLog(0,all) = q;

    // ===================== Control loop =====================
    long convergenceIndex = 0;
    constexpr double epsilon = 0.1e-4;

    int plotIteration = 0;
    Eigen::Vector3d pDesEigen = points[1];

    casadi::DM Pc; // current state (nx)
    casadi::DM Uc; // applied input (nu)

    for (int j = 1; j < (int)points.size() - 1; ++j) {
        // refresh horizon desired points starting at points[j+1]
        xd_dm = buildXdHorizonDM(points, /*startIndex=*/j + 1, cfg.N, cfg.nx);

        // while but capped by convergenceIndex<1
        while (((pDesEigen - P).norm() > epsilon) && convergenceIndex < 1) {
            tStart = std::chrono::high_resolution_clock::now();

            auto res = pb.solver(arg);

            tEnd = std::chrono::high_resolution_clock::now();
            double tStartMs = std::chrono::duration<double, std::milli>( tStart - t0Start ).count();
            auto stepSolveTimeMs = std::chrono::duration<double, std::milli>(tEnd - tStart).count();
            pointSolveTime(plotIteration + 1,0) = tStartMs;
            pointSolveTime(plotIteration + 1,1) = stepSolveTimeMs;

            casadi::DM sol = res.at("x");

            const int jok = 1; // <= N
            Pc = sol(casadi::Slice(cfg.nx * jok, (jok + 1) * cfg.nx));
            Uc = sol(casadi::Slice(cfg.nx * (cfg.N + 1) + (jok - 1) * cfg.nu,
                                  cfg.nx * (cfg.N + 1) + jok * cfg.nu));

            // Warm-start update (shift + reuse)
            warmStartFromSolution(sol, cfg, x0_guess);

            // Update parameters: x0 := Pc, Xd := xd_dm, u0 := Uc
            arg["p"]  = casadi::DM::vertcat({Pc, xd_dm, Uc});
            arg["x0"] = x0_guess;

            // Apply input to CTR to update P
            std::vector<double> UcV = Uc.nonzeros();
            // Eigen::Map<Eigen::VectorXd> UcEigen(UcV.data(), (int)UcV.size());
            // UcEigen.row(2) = UcEigen.row(2);
            // UcEigen.row(1) = UcEigen.row(1) + UcEigen.row(2);
            // UcEigen.row(0) = UcEigen.row(0) + UcEigen.row(1) + UcEigen.row(2);
            
            // recover originl inputs
            Eigen::VectorXd UcEigen(nu);
            UcEigen << UcV[0] + UcV[1] + UcV[2],
                    UcV[1] + UcV[2],
                    UcV[2], 
                    UcV[3],// - UcV[4] ,
                    UcV[4],// - UcV[5],
                    UcV[5];
                    
            ctr.Compute(UcEigen, opt_LOAD);
            P = ctr.GetP();

            // Logging + plotting
            logTraj(plotIteration + 1, all)    = P;
            targetTraj(plotIteration + 1, all) = pDesEigen;

            inputsLog(plotIteration + 1,all) = UcEigen;

            const Eigen::Vector3d e = (P - pDesEigen);
            sequareErrorSumPos += e.squaredNorm();
            ++sampleCount;

            // reconstruct predicted X for plotting
            casadi::DM Xall = sol(casadi::Slice(0, cfg.nx * (cfg.N + 1)));
            std::vector<double> XallVec = Xall.nonzeros();
            Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
                Xpred(XallVec.data(), cfg.N + 1, cfg.nx);

            Eigen::MatrixXd traj    = logTraj.topRows(plotIteration + 1);
            Eigen::MatrixXd target  = targetTraj.topRows(plotIteration + 1);
            Eigen::MatrixXd pred    = Eigen::MatrixXd(Xpred);   // materialize Map -> Matrix

            CtrLib::plotCtrWithTargetTrajectory(
                ctr.GetYTot(),
                ctr.segmented.iEnd,
                traj,
                target,
                pred
            );


            ++convergenceIndex;
            // std::this_thread::sleep_for(1ms);
        }

    ++plotIteration;
    convergenceIndex = 0;
    pDesEigen = points[j + 1]; // next target point
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
        << pointSolveTime(i,0) <<";"<< pointSolveTime(i,1)<< ";"
        << inputsLog(i,0) << ";" << inputsLog(i,1) << ";" << inputsLog(i,2)  << ";"
        << inputsLog(i,3) << ";" << inputsLog(i,4) << ";" << inputsLog(i,5)  << ";"
        << "\n";
    }
    csv.close();

    std::cout << "Final desired point:\n" << Pdes << "\n";
    return 0;
}

std::vector<Eigen::Vector3d> generatePoints( const Eigen::Vector3d& P, const Eigen::Vector3d& Pdes, double dl)  { 
    std::vector<Eigen::Vector3d> points;
    // Direction vector
    Eigen::Vector3d dir = Pdes - P;
    double totalDist = dir.norm();

    // Handle edge case: same point
    if (totalDist < 1e-12) {
        points.push_back(P);
        return points;
    }

    // Normalize direction
    Eigen::Vector3d unitDir = dir.normalized();

    // Number of steps
    int nSteps = static_cast<int>(std::floor(totalDist / dl));

    points.reserve(nSteps + 2);

    // Generate points
    for (int i = 0; i <= nSteps; ++i) {
        points.push_back(P + i * dl * unitDir);
    }

    // Ensure final destination is included
    if ((points.back() - Pdes).norm() > 1e-12) {
        points.push_back(Pdes);
    }

    return points;
}



// std::vector<std::vector<double>> eigenToStdNestedVector(const Eigen::Matrix<double, Eigen::Dynamic, 3>& matPoints) {
std::vector<std::vector<double>> eigenToStdNestedVector(const Eigen::Vector3d& matPoints) {
    std::vector<std::vector<double>> points;
    points.reserve(matPoints.rows());

    for (int i = 0; i < matPoints.rows(); ++i) {
        points.push_back({matPoints(i,0), matPoints(i,1), matPoints(i,2)});
    }

    return points;
}

