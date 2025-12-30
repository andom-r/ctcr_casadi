// BlackboxConstr.cpp
#include "BlackboxConstr.hpp"

BlackboxConstr::BlackboxConstr(
        const std::string& name, 
        CtrLib::CtrModel* model_ptr, 
        int nx_, 
        int nu_,
        const CtrLib::computationOptions& opt_LOAD_
    ) : casadi::Callback(), 
        ctrModel(*model_ptr), 
        numberOfModelComputes(0), 
        nx(nx_), 
        nu(nu_),
        opt_LOAD(opt_LOAD_) {
        construct(name, casadi::Dict());
    }

// Number of inputs (x and u)
casadi_int BlackboxConstr::get_n_in() {
    return 1;  // u
}

// Number of outputs (g)
casadi_int BlackboxConstr::get_n_out() {
    return 1;
}

std::string BlackboxConstr::get_name_in(casadi_int i) {
    if(i==0) return "u";
    return "unknown";
}

std::string BlackboxConstr::get_name_out(casadi_int /*i*/) {
    return "g";
}

// Sparsity of inputs
casadi::Sparsity BlackboxConstr::get_sparsity_in(casadi_int i) {
    if(i == 0) return casadi::Sparsity::dense(nu, 1);   // u
    throw std::runtime_error("Bad input index");
}

// Sparsity of outputs
casadi::Sparsity BlackboxConstr::get_sparsity_out(casadi_int i) {
    if(i == 0) return casadi::Sparsity::dense(nx, 1);  // g
    throw std::runtime_error("Bad output index");
}

// bool has_jacobian() const override { return false; }
// allow FD by advertising capability - DERIVATIVES IMPLEMENTED MANUALLY
bool BlackboxConstr::has_forward(casadi_int /*nfwd*/) const { return true; }
bool BlackboxConstr::has_reverse(casadi_int /*nadj*/) const { return true; }

// Provide Jacobian sparsity
bool BlackboxConstr::has_jac_sparsity(casadi_int /*i*/, casadi_int /*o*/) const { return true; }

// Ensure has_forward returns true
casadi::Function BlackboxConstr::get_forward(casadi_int nfwd,
                                            const std::string& name,
                                            const std::vector<std::string>& /*inames*/,
                                            const std::vector<std::string>& /*onames*/,
                                            const casadi::Dict& opts) const {
    // Local helper type
    struct ForwardCallback : public casadi::Callback {
        const BlackboxConstr& parent;
        int ng,nu, nfwd_;
        double eps;
        ForwardCallback(const BlackboxConstr& p, int ng_, int nu_,int nfwd__, double eps_ = 1e-6)
        : Callback(), parent(p), ng(ng_), nu(nu_),nfwd_(nfwd__), eps(eps_) {
            construct("forward_cb", casadi::Dict());
        }

        casadi_int get_n_in() override  { return 3; } // u,g,du
        casadi_int get_n_out() override { return 1; } // dg

        std::string get_name_in(casadi_int i) override {
            if (i==0) return "u";
            if (i==1) return "g";
            if (i==2) return "du";
            return "in";
        }
        std::string get_name_out(casadi_int) override { return "dg"; }

        casadi::Sparsity get_sparsity_in(casadi_int i) override {
            if (i==0) return casadi::Sparsity::dense(nu, 1);      // u
            if (i==1) return casadi::Sparsity::dense(ng, 1);      // g  
            if (i==2) return casadi::Sparsity::dense(nu, nfwd_);  // du
            throw std::runtime_error("Bad input index");
        }
        casadi::Sparsity get_sparsity_out(casadi_int) override {
            return casadi::Sparsity::dense(ng, nfwd_);            // dg
        }

        // implement eval(...) here (using parent.eval(...) and loop over nfwd_ columns)
        std::vector<casadi::DM> eval(const std::vector<casadi::DM>& in) const override {
            std::cout << "Call for get_forwared eval *-*-*-*-*-*-*-*-* " << std::endl;
            // same multi-column FD logic
            casadi::DM u0 = in[0]; // nu x 1
            casadi::DM g_in = in[1]; // ng x 1 or ng x nfwd
            casadi::DM du  = in[2]; // nu x nfwd

            // Determine nfwd from dx columns
            int nfwd = 1;
            try {
                nfwd = du.size2(); // number of columns
            } catch (...) {
                nfwd = 1;
            }

            // Prepare output dg (ng x nfwd)
            casadi::DM dg = casadi::DM::zeros(ng, nfwd);

            // If g_in has nfwd columns use them, otherwise broadcast g0
            bool g_has_cols = (g_in.size2() == nfwd);

            for (int k = 0; k < nfwd; ++k) {
                // baseline g column
                casadi::DM g0_col;
                if (g_has_cols) {
                    g0_col = g_in(casadi::Slice(), k);
                } else {
                    g0_col = g_in; // nx x 1
                }

                // quick zero check for du(:,k)
                bool du_zero = true;
                for (int j = 0; j < nu; ++j) if (double(du(j,k)) != 0.0) { du_zero = false; break; }
                if (du_zero) {
                    // leave dg(:,k) = 0
                    continue;
                }

                // Build perturbed inputs: u + eps*du(:,k)
                casadi::DM up = u0;
                for (int j = 0; j < nu; ++j) up(j) = up(j) + eps * du(j,k);

                casadi::DM gp = parent.eval({up})[0]; // ng x 1
                casadi::DM col = (gp - g0_col) / eps;     // ng x 1

                // set column k of dg
                for (int r = 0; r < ng; ++r) dg(r, k) = col(r);
            }

            return { dg };
        }
    };

    // instantiate and factory the forward callback
    ForwardCallback *fcb = new ForwardCallback(*this, nx, nu, nfwd, 1e-6);
    std::vector<std::string> f_inames = {"u","g","du"};
    std::vector<std::string> f_onames = {"dg"};
    casadi::Dict factory_opts = opts;
    if (factory_opts.find("expand") == factory_opts.end()) factory_opts["expand"] = false;
    casadi::Function fwd_fun = fcb->factory(name.empty() ? "forward_fun" : name,
                                           f_inames, f_onames,
                                           casadi::Callback::AuxOut(), factory_opts);
    return fwd_fun;
}

casadi::Sparsity BlackboxConstr::get_jac_sparsity(casadi_int i, casadi_int o) const {
    // o=0 -> output "g" of size nx x 1
    // i=0 -> input "x" of size nx x 1, Jacobian is (nx x nx)
    // i=1 -> input "u" of size nu x 1, Jacobian is (nx x nu)
    if (o != 0) throw std::runtime_error("Bad output index in get_jac_sparsity");
    if (i == 0) return casadi::Sparsity::dense(nx, nu);
    throw std::runtime_error("Bad input index in get_jac_sparsity");
}

casadi::Function BlackboxConstr::get_reverse(casadi_int nadj,
                                            const std::string& name,
                                            const std::vector<std::string>& /*inames*/,
                                            const std::vector<std::string>& /*onames*/,
                                            const casadi::Dict& opts) const {
    // Local small callback that computes adjoints by finite differences
    struct ReverseCallback : public casadi::Callback {
        const BlackboxConstr& parent;
        int nx;
        int nu;
        int nadj;
        double eps;

        ReverseCallback(const BlackboxConstr& p, int nx_, int nu_,int nadj_, double eps_ = 1e-6)
            : Callback(), parent(p), nx(nx_), nu(nu_), nadj(nadj_), eps(eps_)
        {
            construct("reverse_cb", casadi::Dict());
        }

        // Inputs: u, g, lambda 
        casadi_int get_n_in() override { return 2 + nadj; }  // u, g, lambda[0..nadj-1]
        casadi_int get_n_out() override { return nadj; }     // adj_u[0..nadj-1]

        std::string get_name_in(casadi_int i) override {
            if (i == 0) return "u";
            if (i == 1) return "g";
            if (i >= 2 && i < 2 + nadj)
                return "lambda_" + std::to_string(i - 2);
            return "in";
        }

        std::string get_name_out(casadi_int i) override {
            if (i >= 0 && i < nadj)
                return "adj_u_" + std::to_string(i);
            return "out";
        }

        casadi::Sparsity get_sparsity_in(casadi_int i) override {
            if (i == 0) return casadi::Sparsity::dense(nu, 1); // u
            if (i == 1) return casadi::Sparsity::dense(nx, 1); // g
            if (i >= 2 && i < 2 + nadj)
                return casadi::Sparsity::dense(nx, 1);         // lambda_k
            throw std::runtime_error("Bad input index");
        }

        casadi::Sparsity get_sparsity_out(casadi_int i) override {
            if (i >= 0 && i < nadj)
                return casadi::Sparsity::dense(nu, 1);         // adj_u_k
            throw std::runtime_error("Bad output index");
        }

        // We implement eval: inputs are DMs
        std::vector<casadi::DM> eval(const std::vector<casadi::DM>& in) const override {
            casadi::DM u0 = in[0];
            casadi::DM g0 = in[1];

            std::vector<casadi::DM> out(nadj, casadi::DM::zeros(nu, 1));

            // for each adjoint seed
            for (int k = 0; k < nadj; ++k) {
                casadi::DM lambda = in[2 + k];
                casadi::DM adj_u = casadi::DM::zeros(nu, 1);

                // FD Jacobian column by column
                for (int j = 0; j < nu; ++j) {
                    casadi::DM up = u0;
                    up(j) += eps;

                    casadi::DM gp = parent.eval({up})[0];
                    casadi::DM diff = (gp - g0) / eps;

                    double s = 0.0;
                    for (int r = 0; r < nx; ++r)
                        s += double(diff(r)) * double(lambda(r));

                    adj_u(j) = s;
                }

                out[k] = adj_u;
            }

            return out;
        }
    };

    // nadj is the number of adjoints requested
    double eps = 1e-6;
    ReverseCallback* rcb = new ReverseCallback(*this, nx, nu, nadj, eps);

    // build dynamic input names
    std::vector<std::string> rev_inames;
    rev_inames.push_back("u");
    rev_inames.push_back("g");
    for (int k = 0; k < nadj; ++k) {
        rev_inames.push_back("lambda_" + std::to_string(k));
    }

    // build dynamic output names
    std::vector<std::string> rev_onames;
    for (int k = 0; k < nadj; ++k) {
        rev_onames.push_back("adj_u_" + std::to_string(k));
    }

    // ensure expand= false
    casadi::Dict factory_opts = opts;
    if (factory_opts.find("expand") == factory_opts.end()) {
        factory_opts["expand"] = false;
    }

    // build the reverse Function
    casadi::Function rev_fun = rcb->factory(
        name.empty() ? "reverse_fun" : name,
        rev_inames,
        rev_onames,
        casadi::Callback::AuxOut(),
        factory_opts
    );

    return rev_fun;
}

casadi::Function BlackboxConstr::get_jacobian(const std::string& name,
                                             const std::vector<std::string>& /*inames*/,
                                             const std::vector<std::string>& /*onames*/,
                                             const casadi::Dict& opts) const {
    // Local callback that returns g and jacobian(g,x) via FD
    struct JacCallback : public casadi::Callback {
        const BlackboxConstr& parent;
        int nx;
        int nu;
        double eps;
        JacCallback(const BlackboxConstr& p, int nx_, int nu_, double eps_ = 1e-6)
            : Callback(), parent(p), nx(nx_), nu(nu_), eps(eps_)
        {
            construct("jac_cb", casadi::Dict());
        }

        casadi_int get_n_in() override { return 1; }   // u
        casadi_int get_n_out() override { return 2; }  // g, jac:g:x

        std::string get_name_in(casadi_int i) override {
            if (i == 0) return "u";
            return "in";
        }
        std::string get_name_out(casadi_int i) override {
            if (i == 0) return "g";
            if (i == 1) return "jac:g:u";
            return "out";
        }

        casadi::Sparsity get_sparsity_in(casadi_int i) override {
            if (i == 0) return casadi::Sparsity::dense(nu, 1);
            throw std::runtime_error("Bad input index");
        }
        casadi::Sparsity get_sparsity_out(casadi_int i) override {
            if (i == 0) return casadi::Sparsity::dense(nx, 1);   // g
            if (i == 1) return casadi::Sparsity::dense(nx, nu);  // jacobian matrix
            throw std::runtime_error("Bad output index");
        }

        // Evaluate g and jacobian(g,x) by FD
        std::vector<casadi::DM> eval(const std::vector<casadi::DM>& in) const override {
            std::cout << "Call for get_jacobian *.*.*.*.*.*.*.*.*.*.*.*.*.*.* " << std::endl;

            casadi::DM u0 = in[0]; // nu x 1

            // baseline g
            std::vector<casadi::DM> gvec = parent.eval({u0});
            casadi::DM g0 = gvec[0]; // nx x 1

            // jacobian matrix (nx x nx)
            casadi::DM J = casadi::DM::zeros(nx, nu);

            for (int i = 0; i < nu; ++i) {
                casadi::DM up = u0;
                up(i) = up(i) + eps;
                std::vector<casadi::DM> gpv = parent.eval({up});
                casadi::DM gp = gpv[0];
                casadi::DM col = (gp - g0) / eps; // nx x 1
                // set column i
                for (int r = 0; r < nx; ++r) J(r, i) = col(r);
            }

            return {g0, J};
        }
    };

    // Create instance and factory it
    double eps = 1e-6;
    JacCallback *jcb = new JacCallback(*this, nx, nu, eps);

    std::vector<std::string> jac_inames  = {"u"};
    std::vector<std::string> jac_onames  = {"g", "jac:g:u"};

    casadi::Dict factory_opts = opts;
    if (factory_opts.find("expand") == factory_opts.end()) factory_opts["expand"] = false;

    casadi::Function jac_fun = jcb->factory(
        name.empty() ? "jac_bb_fun" : name,
        jac_inames,
        jac_onames,
        casadi::Callback::AuxOut(),
        factory_opts
    );

    return jac_fun;
}


std::vector<casadi::DM> BlackboxConstr::eval(const std::vector<casadi::DM>& in) const {

    if (numberOfModelComputes % 10000 == 0) {
        std::cout << "Call for Eval *-*-*-*-*-*-*-*-*-*  "<< numberOfModelComputes << std::endl;
    }
    ++numberOfModelComputes;

    // Convert CasADi input to Eigen vector
    const std::vector<double> u_std = in[0].nonzeros();
    const Eigen::VectorXd u = Eigen::Map<const Eigen::VectorXd>(u_std.data(), u_std.size());

    // Run model
    if (ctrModel.Compute(u, opt_LOAD) < 0) {
        std::cout << "Error: ctrModel.Compute() failed\n";
        std::cout << "Applied u: " << u << std::endl;
        throw std::runtime_error("Model did not converge");
    }

    // Convert prediction back to CasADi
    const Eigen::VectorXd predicted = ctrModel.GetP();
    std::vector<double> predictedV(predicted.data(), predicted.data() + predicted.size());

    return { casadi::DM(predictedV) };
}

