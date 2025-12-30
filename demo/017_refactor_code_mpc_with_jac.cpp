#include <iostream>
#include <thread>
#include <chrono>

#include "loadParameters.h"
#include "CtrModel.h"
#include "pinv.h"
#include "plotCtr.h"
#include <casadi/casadi.hpp>
#include <BlackboxConstrJacobian.hpp>

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
int nu = NB_Q;
int NPred  = 2;

int main(int, char **) {

  // Example : Simple control scenario for a straight line using updated jacobian with MPC
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
    CtrModel ctr(pNominal);

    Vector_q q(-0.3, -0.2, -0.1, 0.0, 0.0, 0.0);
    if (ctr.Compute(q, opt_LOAD_J) < 0) {
        std::cout << "Error: ctr.Compute() failed\n";
        return -1;
    }

    // ---------------- Part II : Initialize control simulation ----------------
    Eigen::Vector3d P = ctr.GetP();
    Eigen::Vector3d Pdes = P + Eigen::Vector3d(-40e-3, -40e-3, 0.0);

    constexpr double dl = 2e-4;
    std::vector<Eigen::Vector3d> points = generatePoints(P, Pdes, dl);
    if (points.size() < 2) {
        points = {P, Pdes};
    }

    // Prediction horizen
    const int N = NPred;

    // Constraint settings
    const double betweenTubeBaseMargin = 1e-5;
    const double inputTranslationVariationLimit = 1e-3;
    const double inputRotationVariationLimit     = 1e-3;

    // gu layout per prediction step
    //   0: -u0 + u1
    //   1: -u1 + u2
    //   2..(2+nu-1): delta_u(j)
    const int nbTranUConst = 2 + nu;

    // ---------------- Plot init ----------------
    plotCtrWithTargetTrajectory(ctr.GetYTot(), ctr.segmented.iEnd, P.transpose(), P.transpose());
    std::this_thread::sleep_for(std::chrono::seconds(1));

    constexpr int logSize = 6000;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> logTraj(logSize, 3);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> targetTraj(logSize, 3);
    using Eigen::all;
    using Eigen::seq;

    logTraj(0, all) = P;
    targetTraj(0, all) = P;

    // ---------------- CasADi symbols ----------------
    casadi::MX X  = casadi::MX::sym("X", nx, N + 1);
    casadi::MX U  = casadi::MX::sym("U", nu, N);

    casadi::MX x0 = casadi::MX::sym("x0", nx);
    casadi::MX Xd = casadi::MX::sym("Xd", nx, N);   // reference horizon
    casadi::MX uPrev = casadi::MX::sym("uPrev", nu);

    // Cost matrices
    casadi::MX Q = 1e7 * casadi::MX::eye(nx);

    const double multi = 1e3;
    casadi::MX R1 = multi * 0.1 * casadi::MX::eye(3);
    casadi::MX R2 = multi * 0.1 * casadi::MX::eye(3);
    casadi::MX R  = casadi::MX::diagcat({R1, R2});

    casadi::MX obj = 0;
    std::vector<casadi::MX> g;   // dynamics + initial condition
    std::vector<casadi::MX> gu;  // scalar inequality constraints per step for inputs

    // Initial condition constraint
    g.push_back(X(casadi::Slice(), 0) - x0);

    // Stage costs + gu constraints
    for (int k = 0; k < N; ++k) {
        casadi::MX xk  = X(casadi::Slice(), k + 1);
        casadi::MX xdk = Xd(casadi::Slice(), k);
        casadi::MX uk  = U(casadi::Slice(), k);

        casadi::MX xDiff = (xk - xdk);

        casadi::MX du;
        if (k == 0) du = (uk - uPrev);
        else        du = (uk - U(casadi::Slice(), k - 1));

        casadi::MX xQx = casadi::MX::mtimes(xDiff.T(), casadi::MX::mtimes(Q, xDiff));
        obj += xQx;
        // obj += casadi::MX::exp((xQx / 1e3 + 10)); 
        // obj += casadi::MX::mtimes(du.T(), casadi::MX::mtimes(10 * R, du));

        // tube ordering constraints (translation)
        gu.push_back(-uk(0) + uk(1));
        gu.push_back(-uk(1) + uk(2));

        // input variation constraints (translation + rotation)
        for (int j = 0; j < nu; ++j) {
            gu.push_back(du(j));
        }
    }

    // Blackbox dynamics constraints X(k+1) - f(U(k)) = 0
    BlackboxConstrJacobian f_blackbox("bb_constr", &ctr, nx, nu, opt_LOAD);
    casadi::Function bb_fun = f_blackbox.factory(
        "bb_fun",
        {"u"},
        {"g"},
        casadi::Callback::AuxOut(),
        casadi::Dict{
            {"expand", false},   // keep as call node
            {"enable_fd", true}  // FD derivatives
        }
    );

    for (int k = 0; k < N; ++k) {
        g.push_back(X(casadi::Slice(), k + 1) - bb_fun(std::vector<casadi::MX>{U(casadi::Slice(), k)})[0]);
    }

    // Decision vector: [vec(X); vec(U)] 
    casadi::MX nlp_x = casadi::MX::vertcat({
        casadi::MX::reshape(X, nx * (N + 1), 1),
        casadi::MX::reshape(U, nu * N, 1)
    });

    // Constraints vector: [gu; g]
    casadi::MX nlp_g = casadi::MX::vertcat({
        casadi::MX::vertcat(gu),
        casadi::MX::vertcat(g)
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

    casadi::Function solver = casadi::nlpsol("solver", "ipopt", nlp, opts);

    // ---------------- Variable bounds ----------------
    const int n_dec = nx * (N + 1) + nu * N;
    casadi::DM lbx = -casadi::DM::inf(n_dec);
    casadi::DM ubx =  casadi::DM::inf(n_dec);

    // Bounds on U(k)
    for (int k = 0; k < N; ++k) {
        const int idx = nx * (N + 1) + k * nu;
        lbx(idx + 0) = -ctr.GetTubeParameters().l(0);
        lbx(idx + 1) = -ctr.GetTubeParameters().l(1);
        lbx(idx + 2) = -ctr.GetTubeParameters().l(2) + 3 * betweenTubeBaseMargin;

        ubx(idx + 0) = -3 * betweenTubeBaseMargin;
        ubx(idx + 1) = -2 * betweenTubeBaseMargin;
        ubx(idx + 2) = -1 * betweenTubeBaseMargin;
    }

    // ---------------- Constraint bounds ----------------
    const int ng = static_cast<int>(gu.size()) + static_cast<int>(g.size()) * nx;
    casadi::DM lbg = casadi::DM::zeros(ng);
    casadi::DM ubg = casadi::DM::zeros(ng);

    // gu bounds (first gu block is length N * nbTranUConst)
    for (int i = 0; i < N; ++i) {
        // tube ordering
        ubg(i * nbTranUConst + 0) = ctr.GetTubeParameters().l(0) - ctr.GetTubeParameters().l(1) - 3 * betweenTubeBaseMargin;
        lbg(i * nbTranUConst + 0) = 1 * betweenTubeBaseMargin;

        ubg(i * nbTranUConst + 1) = ctr.GetTubeParameters().l(1) - ctr.GetTubeParameters().l(2) - 3 * betweenTubeBaseMargin;
        lbg(i * nbTranUConst + 1) = 1 * betweenTubeBaseMargin;

        // delta_u bounds
        const int base = i * nbTranUConst + 2;
        // first 3 - translations
        for (int j = 0; j < 3 && j < nu; ++j) {
            ubg(base + j) =  inputTranslationVariationLimit;
            lbg(base + j) = -inputTranslationVariationLimit;
        }
        // remaining - rotations
        for (int j = 3; j < nu; ++j) {
            ubg(base + j) =  inputRotationVariationLimit;
            lbg(base + j) = -inputRotationVariationLimit;
        }
    }

    // ---------------- Initial guess ----------------
    casadi::DM x_init = casadi::DM::zeros(n_dec);
    const int u_offset = nx * (N + 1);

    // initialize U guess from current configuration q
    for (int i = 0; i < nu * N; ++i) {
        x_init(u_offset + i) = q(i % nu);
    }

    // initialize X guess from point path -repeat last point if needed
    for (int i = 0; i < N + 1; ++i) {
        Eigen::Vector3d pi = safePoint(points, i);
        x_init(i * nx + 0) = pi(0);
        x_init(i * nx + 1) = pi(1);
        x_init(i * nx + 2) = pi(2);
    }

    // ---------------- Solve loop ----------------
    casadi::DM Pc; // current predicted point used as new x0
    casadi::DM Uc; // current applied control used as new uPrev

    long plotIteration = 0;
    constexpr double epsilon = 1e-5;
    constexpr int maxAttemptsPerPoint = 1; //

    // initial p
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

    // Iterate over reference points
    for (int j = 1; j < static_cast<int>(points.size()) - 1; ++j) {
        Eigen::Vector3d pDesEigen = safePoint(points, j + 1);

        for (int attempt = 0; attempt < maxAttemptsPerPoint; ++attempt) {
            if ((pDesEigen - P).norm() <= epsilon) break;

            auto res = solver(arg);
            casadi::DM sol = res.at("x");

            const int jok = 2; // <= N
            casadi::DM UcAll = sol(casadi::Slice(nx * (N + 1), nx * (N + 1) + N * nu));
            casadi::DM PcAll = sol(casadi::Slice(0, (N + 1) * nx));

            Pc = sol(casadi::Slice(nx * jok, (jok + 1) * nx));
            Uc = sol(casadi::Slice(nx * (N + 1) + (jok - 1) * nu, nx * (N + 1) + jok * nu));

            // shift U and X
            for (int i = 0; i < nu * N; ++i) {
                x_init(u_offset + i) = UcAll(i);
            }
            for (int i = 0; i < nx * N; ++i) {
                x_init(i) = PcAll(i + nx); // shift by one
                if (i >= nx * (N - 1)) {
                    x_init(i + nx) = PcAll(i + nx); // keep last
                }
            }

            // update robot state using applied Uc
            std::vector<double> UcV = Uc.nonzeros();
            Eigen::Map<Eigen::VectorXd> UcEigen(UcV.data(), static_cast<int>(UcV.size()));
            ctr.Compute(UcEigen, opt_LOAD_J);
            P = ctr.GetP();

            // update logs/plots
            logTraj(plotIteration + 1, all) = P;
            targetTraj(plotIteration + 1, all) = pDesEigen;

            // optional horizon display
            std::vector<double> PcAllVector = PcAll.nonzeros();
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