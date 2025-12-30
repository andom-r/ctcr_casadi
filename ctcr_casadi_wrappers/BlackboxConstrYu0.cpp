#include "BlackboxConstrYu0.hpp"

// using namespace CtrLib;
// ------------------------------------------------------------
// Constructor
// ------------------------------------------------------------
BlackboxConstrYu0::BlackboxConstrYu0(const std::string& name,
                               CtrLib::CtrModel* model_ptr,
                               int nx_,
                               int nu_,
                               int bcNu_,
                               int unkNu_,
                               const CtrLib::computationOptions& opt_LOAD_)
    : casadi::Callback(),
      ctrModel(*model_ptr),
      numberOfModelComputes(0),
      nx(nx_),
      nu(nu_),
      bcNu(bcNu_),
      unkNu(unkNu_),
      opt_LOAD(opt_LOAD_) {
    construct(name, casadi::Dict());
}

// ------------------------------------------------------------
// Basic callback interface
// ------------------------------------------------------------

// Number of inputs: u, unk
casadi_int BlackboxConstrYu0::get_n_in() {
    return 2;  // 0: u, 1: unk
}

// Number of outputs: g
casadi_int BlackboxConstrYu0::get_n_out() {
    return 1;  // 0: g
}

std::string BlackboxConstrYu0::get_name_in(casadi_int i) {
    switch (i) {
        case 0: return "u";
        case 1: return "unk";
        default: throw std::runtime_error("Bad input index");
    }
}

std::string BlackboxConstrYu0::get_name_out(casadi_int i) {
    if (i == 0) return "g";
    throw std::runtime_error("Bad output index");
}

// ------------------------------------------------------------
// Sparsity information
// ------------------------------------------------------------

// Sparsity of inputs
casadi::Sparsity BlackboxConstrYu0::get_sparsity_in(casadi_int i) {
    switch (i) {
        case 0: return casadi::Sparsity::dense(nu, 1);      // u_k
        case 1: return casadi::Sparsity::dense(unkNu, 1);   // unk_k
        default: throw std::runtime_error("Bad input index");
    }
}

// Sparsity of outputs: g has size (nx + bcNu) x 1
casadi::Sparsity BlackboxConstrYu0::get_sparsity_out(casadi_int i) {
    if (i == 0) return casadi::Sparsity::dense(nx + bcNu, 1);
    throw std::runtime_error("Bad output index");
}

// Allow CasADi to use finite differences
bool BlackboxConstrYu0::has_forward(casadi_int) const {
    return true;
}

bool BlackboxConstrYu0::has_reverse(casadi_int) const {
    return true;
}

// Advertise Jacobian sparsity
bool BlackboxConstrYu0::has_jac_sparsity(casadi_int, casadi_int) const {
    return true;
}

casadi::Sparsity BlackboxConstrYu0::get_jac_sparsity(casadi_int i, casadi_int o) const {
    // Only one output g
    if (o != 0) {
        throw std::runtime_error("Bad output index in get_jac_sparsity");
    }

    // dg/du
    if (i == 0) return casadi::Sparsity::dense(nx + bcNu, nu);

    // dg/dunk
    if (i == 1) return casadi::Sparsity::dense(nx + bcNu, unkNu);

    throw std::runtime_error("Bad input index in get_jac_sparsity");
}

// ------------------------------------------------------------
// Forward-mode callback (FD-based)
// ------------------------------------------------------------
casadi::Function BlackboxConstrYu0::get_forward(
    casadi_int nfwd,
    const std::string& name,
    const std::vector<std::string>&,
    const std::vector<std::string>&,
    const casadi::Dict& opts) const {

    // Local callback used by CasADi for forward sensitivities
    struct ForwardCallback : public casadi::Callback {
        const BlackboxConstrYu0& parent;
        int ng, nu, nunk, nfwd_;
        double eps;

        ForwardCallback(const BlackboxConstrYu0& p,
                        int ng_, int nu_, int nunk_, int nfwd__,
                        double eps_ = 1e-6)
            : casadi::Callback(),
              parent(p),
              ng(ng_),
              nu(nu_),
              nunk(nunk_),
              nfwd_(nfwd__),
              eps(eps_) {
            construct("forward_cb", casadi::Dict());
        }

    
        casadi_int get_n_in() override  { return 5; } // u, unk, g, du, dunk
        casadi_int get_n_out() override { return 1; } // dg

        std::string get_name_in(casadi_int i) override {
            if (i == 0) return "u";
            if (i == 1) return "unk";
            if (i == 2) return "g";
            if (i == 3) return "du";
            if (i == 4) return "dunk";
            return "in";
        }
        std::string get_name_out(casadi_int) override { return "dg"; }

        casadi::Sparsity get_sparsity_in(casadi_int i) override {
            if (i == 0) return casadi::Sparsity::dense(nu, 1);         // u
            if (i == 1) return casadi::Sparsity::dense(nunk, 1);       // unk
            if (i == 2) return casadi::Sparsity::dense(ng, 1);         // g 
            if (i == 3) return casadi::Sparsity::dense(nu, nfwd_);     // du
            if (i == 4) return casadi::Sparsity::dense(nunk, nfwd_);   // dunk
            throw std::runtime_error("Bad input index");
        }

        casadi::Sparsity get_sparsity_out(casadi_int) override {
            return casadi::Sparsity::dense(ng, nfwd_);                 // dg
        }

        std::vector<casadi::DM> eval(const std::vector<casadi::DM>& in) const override {
            const casadi::DM& u0   = in.at(0);
            const casadi::DM& unk0 = in.at(1);
            const casadi::DM& g0   = in.at(2);
            const casadi::DM& du   = in.at(3);
            const casadi::DM& dunk = in.at(4);

            casadi::DM dg = casadi::DM::zeros(ng, nfwd_);

            for (int k = 0; k < nfwd_; ++k) {
                casadi::DM up   = u0;
                casadi::DM unkp = unk0;

                for (int j = 0; j < nu;   ++j) up(j)   += eps * du(j, k);
                for (int j = 0; j < nunk; ++j) unkp(j) += eps * dunk(j, k);

                casadi::DM gp = parent.eval({up, unkp}).at(0);
                casadi::DM col = (gp - g0) / eps;

                for (int r = 0; r < ng; ++r) dg(r, k) = col(r);
            }

            return {dg};
        }
    };

    const int ng = nx + bcNu;
    auto fcb = std::make_shared<ForwardCallback>(*this, ng, nu, unkNu, static_cast<int>(nfwd), 1e-8);
    owned_callbacks_.push_back(fcb);

    casadi::Dict factory_opts = opts;
    factory_opts["expand"] = false;

    return fcb->factory(
        name.empty() ? "forward_fun" : name,
        {"u", "unk", "g", "du", "dunk"},
        {"dg"},
        casadi::Callback::AuxOut(),
        factory_opts
    );
}

// ------------------------------------------------------------
// Reverse-mode callback (FD-based)
// ------------------------------------------------------------
casadi::Function BlackboxConstrYu0::get_reverse(
    casadi_int nadj,
    const std::string& name,
    const std::vector<std::string>&,
    const std::vector<std::string>&,
    const casadi::Dict& opts) const {

    struct ReverseCallback : public casadi::Callback {
        const BlackboxConstrYu0& parent;
        int ng, nu, nunk, nadj;
        double eps;

        ReverseCallback(const BlackboxConstrYu0& p,
                        int ng_, int nu_, int nunk_, int nadj_,
                        double eps_ = 1e-6)
            : casadi::Callback(),
              parent(p),
              ng(ng_),
              nu(nu_),
              nunk(nunk_),
              nadj(nadj_),
              eps(eps_) {
            construct("reverse_cb", casadi::Dict());
        }

        casadi_int get_n_in() override  { return 3 + nadj; }
        casadi_int get_n_out() override { return 2 * nadj; }

        std::string get_name_in(casadi_int i) override {
            if (i == 0) return "u";
            if (i == 1) return "unk";
            if (i == 2) return "g";
            if (i >= 3 && i < 3 + nadj) return "lambda_" + std::to_string(i - 3);
            return "in";
        }

        std::string get_name_out(casadi_int i) override {
            if (i >= 0 && i < nadj)           return "adj_u_"   + std::to_string(i);
            if (i >= nadj && i < 2 * nadj)    return "adj_unk_" + std::to_string(i - nadj);
            return "out";
        }

        casadi::Sparsity get_sparsity_in(casadi_int i) override {
            if (i == 0) return casadi::Sparsity::dense(nu, 1);     // u
            if (i == 1) return casadi::Sparsity::dense(nunk, 1);   // unk
            if (i == 2) return casadi::Sparsity::dense(ng, 1);     // g
            if (i >= 3 && i < 3 + nadj) return casadi::Sparsity::dense(ng, 1); // lambda_k
            throw std::runtime_error("Bad input index");
        }

        casadi::Sparsity get_sparsity_out(casadi_int i) override {
            if (i >= 0 && i < nadj)        return casadi::Sparsity::dense(nu, 1);    // adj_u_k
            if (i >= nadj && i < 2 * nadj) return casadi::Sparsity::dense(nunk, 1);  // adj_unk_k
            throw std::runtime_error("Bad output index");
        }

        std::vector<casadi::DM> eval(const std::vector<casadi::DM>& in) const override {
            const casadi::DM& u0   = in.at(0);
            const casadi::DM& unk0 = in.at(1);
            const casadi::DM& g0   = in.at(2);

            std::vector<casadi::DM> out(2 * nadj);

            for (int k = 0; k < nadj; ++k) {
                const casadi::DM& lambda = in.at(3 + k);

                casadi::DM adj_u   = casadi::DM::zeros(nu, 1);
                casadi::DM adj_unk = casadi::DM::zeros(nunk, 1);

                for (int j = 0; j < nu; ++j) {
                    casadi::DM up = u0;
                    up(j) += eps;
                    casadi::DM gp = parent.eval({up, unk0}).at(0);
                    casadi::DM diff = (gp - g0) / eps;

                    double s = 0.0;
                    for (int r = 0; r < ng; ++r)
                        s += double(diff(r)) * double(lambda(r));

                    adj_u(j) = s;
                }

                for (int j = 0; j < nunk; ++j) {
                    casadi::DM unkp = unk0;
                    unkp(j) += eps;
                    casadi::DM gp = parent.eval({u0, unkp}).at(0);
                    casadi::DM diff = (gp - g0) / eps;

                    double s = 0.0;
                    for (int r = 0; r < ng; ++r)
                        s += double(diff(r)) * double(lambda(r));

                    adj_unk(j) = s;
                }

                out[k]        = adj_u;
                out[nadj + k] = adj_unk;
            }

            return out;
        }
    };
    

    auto rcb = std::make_shared<ReverseCallback>(*this, nx + bcNu, nu, unkNu, nadj, 1e-8);
    owned_callbacks_.push_back(rcb);

    // Input names: u, unk, g, lambda_0..lambda_{nadj-1}
    std::vector<std::string> rev_inames;
    rev_inames.reserve(3 + nadj);
    rev_inames.push_back("u");
    rev_inames.push_back("unk");
    rev_inames.push_back("g");
    for (int k = 0; k < nadj; ++k) {
        rev_inames.push_back("lambda_" + std::to_string(k));
    }

    // Output names: adj_u_*, adj_unk_*
    std::vector<std::string> rev_onames;
    rev_onames.reserve(2 * nadj);
    for (int k = 0; k < nadj; ++k) {
        rev_onames.push_back("adj_u_" + std::to_string(k));
    }
    for (int k = 0; k < nadj; ++k) {
        rev_onames.push_back("adj_unk_" + std::to_string(k));
    }

    casadi::Dict factory_opts = opts;
    if (factory_opts.find("expand") == factory_opts.end()) {
        factory_opts["expand"] = false;
    }

    return rcb->factory(
        name.empty() ? "reverse_fun" : name,
        rev_inames,
        rev_onames,
        casadi::Callback::AuxOut(),
        factory_opts
    );
    // casadi::Dict factory_opts = opts;
    // factory_opts["expand"] = false;

    // return rcb->factory(
    //     name.empty() ? "reverse_fun" : name,
    //     {},
    //     {},
    //     casadi::Callback::AuxOut(),
    //     factory_opts
    // );
}


casadi::Function BlackboxConstrYu0::get_jacobian(const std::string& name,
                                             const std::vector<std::string>&,
                                             const std::vector<std::string>&,
                                             const casadi::Dict& opts) const {
  struct JacCallback : public casadi::Callback {
    const BlackboxConstrYu0& parent;
    int ng, nu, nunk;
    double eps;

    JacCallback(const BlackboxConstrYu0& p, int ng_, int nu_, int nunk_, double eps_ = 1e-8)
      : casadi::Callback(), parent(p), ng(ng_), nu(nu_), nunk(nunk_), eps(eps_) {
      construct("jac_cb", casadi::Dict());
    }

    casadi_int get_n_in() override  { return 2; }
    casadi_int get_n_out() override { return 3; }

    casadi::Sparsity get_sparsity_in(casadi_int i) override {
      if (i == 0) return casadi::Sparsity::dense(nu, 1);
      if (i == 1) return casadi::Sparsity::dense(nunk, 1);
      throw std::runtime_error("Bad input index");
    }

    casadi::Sparsity get_sparsity_out(casadi_int i) override {
      if (i == 0) return casadi::Sparsity::dense(ng, 1);
      if (i == 1) return casadi::Sparsity::dense(ng, nu);
      if (i == 2) return casadi::Sparsity::dense(ng, nunk);
      throw std::runtime_error("Bad output index");
    }

    std::vector<casadi::DM> eval(const std::vector<casadi::DM>& in) const override {
      const casadi::DM& u0   = in.at(0);
      const casadi::DM& unk0 = in.at(1);

      casadi::DM g0 = parent.eval({u0, unk0}).at(0);

      casadi::DM Ju   = casadi::DM::zeros(ng, nu);
      casadi::DM Junk = casadi::DM::zeros(ng, nunk);

      for (int i = 0; i < nu; ++i) {
        casadi::DM up = u0; up(i) = up(i) + eps;
        casadi::DM gp = parent.eval({up, unk0}).at(0);
        casadi::DM col = (gp - g0) / eps;
        for (int r = 0; r < ng; ++r) Ju(r, i) = col(r);
      }

      for (int i = 0; i < nunk; ++i) {
        casadi::DM unkp = unk0; unkp(i) = unkp(i) + eps;
        casadi::DM gp = parent.eval({u0, unkp}).at(0);
        casadi::DM col = (gp - g0) / eps;
        for (int r = 0; r < ng; ++r) Junk(r, i) = col(r);
      }

      return {g0, Ju, Junk};
    }
  };

  const int ng = nx + bcNu;
  const double eps = 1e-8;

  auto jcb = std::make_shared<JacCallback>(*this, ng, nu, unkNu, eps);
  owned_callbacks_.push_back(jcb);

  std::vector<std::string> jac_inames = {"u", "unk"};
  std::vector<std::string> jac_onames = {"g", "jac:g:u", "jac:g:unk"};

  casadi::Dict factory_opts = opts;
  if (factory_opts.find("expand") == factory_opts.end()) factory_opts["expand"] = false;

  return jcb->factory(name.empty() ? "jac_bb_fun" : name,
                      jac_inames, jac_onames,
                      casadi::Callback::AuxOut(),
                      factory_opts);
}

// ------------------------------------------------------------
// Main model evaluation
// ------------------------------------------------------------
std::vector<casadi::DM> BlackboxConstrYu0::eval(const std::vector<casadi::DM>& in) const {

    if (numberOfModelComputes % 1000000 == 0) {
        std::cout << "Eval call count: " << numberOfModelComputes << std::endl;
    }
    ++numberOfModelComputes;

    // Inputs
    std::vector<double> u_std   = in.at(0).nonzeros();
    std::vector<double> unk_std = in.at(1).nonzeros();

    Eigen::VectorXd u   = Eigen::Map<Eigen::VectorXd>(u_std.data(),   u_std.size());
    Eigen::VectorXd yu0 = Eigen::Map<Eigen::VectorXd>(unk_std.data(), unk_std.size());

    CtrLib::SingleIVPOut outG;
    if (ctrModel.ComputeIVP(u, yu0, opt_LOAD, outG) < 0) {
        throw std::runtime_error("Model did not converge");
    }

    Eigen::VectorXd g(nx + bcNu);
    g.head(nx) = ctrModel.GetP();
    g.tail(bcNu) = outG.b;

    return { casadi::DM(std::vector<double>(g.data(), g.data() + g.size())) };
}


BlackboxConstrYu0::~BlackboxConstrYu0() = default;
