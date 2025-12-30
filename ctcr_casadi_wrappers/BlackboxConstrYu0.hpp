// BlackboxConstrYu0.hpp
#pragma once

#include <casadi/casadi.hpp>
#include "CtrModel.h"         

class BlackboxConstrYu0 : public casadi::Callback {
public:
    // Constructor
    BlackboxConstrYu0(const std::string& name,
                CtrLib::CtrModel* model_ptr,
                int nx_,
                int nu_,
                int bcNu_,
                int unkNu_,
                const CtrLib::computationOptions& opt_LOAD_);

    // --- CasADi Callback interface ---
    casadi_int get_n_in() override;
    casadi_int get_n_out() override;

    std::string get_name_in(casadi_int i) override;
    std::string get_name_out(casadi_int i) override;

    casadi::Sparsity get_sparsity_in(casadi_int i) override;
    casadi::Sparsity get_sparsity_out(casadi_int i) override;

    bool has_forward(casadi_int nfwd) const override;
    bool has_reverse(casadi_int nadj) const override;

    bool has_jac_sparsity(casadi_int i, casadi_int o) const override;
    casadi::Sparsity get_jac_sparsity(casadi_int i, casadi_int o) const;

    casadi::Function get_forward(
        casadi_int nfwd,
        const std::string& name,
        const std::vector<std::string>& inames,
        const std::vector<std::string>& onames,
        const casadi::Dict& opts) const override;

    casadi::Function get_reverse(
        casadi_int nadj,
        const std::string& name,
        const std::vector<std::string>& inames,
        const std::vector<std::string>& onames,
        const casadi::Dict& opts) const override;

    casadi::Function get_jacobian(
        const std::string& name,
        const std::vector<std::string>& inames,
        const std::vector<std::string>& onames,
        const casadi::Dict& opts) const override;

    std::vector<casadi::DM> eval(
        const std::vector<casadi::DM>& in) const override;

    ~BlackboxConstrYu0() override;
private:
    CtrLib::CtrModel& ctrModel;
    
    // Model compute counter
    // mutable long numberOfModelComputes{0};
    mutable long numberOfModelComputes;

    /// Number of state variables.
    /// Examples:
    ///   - 3  → 3D position only
    ///   - 6  → 3D position + 3D orientation (rotation) (Not implemented yet)
    const int nx;

    /// Number of control/input variables.
    const int nu;

    /// Number of boundary condition constraints.
    const int bcNu;

    /// Number of unknown initial variables.
    /// These are typically estimated or optimized initial states.
    const int unkNu;

    /// Calculation/load option selector, see CtrLib doc.
    /// Used to control model evaluation behavior.
    /// (Reserved for future extensions.)
    const CtrLib::computationOptions& opt_LOAD;


    // CasADi Functions returned by get_forward/get_reverse/get_jacobian must keep
    // the callback objects alive. We store them here.
    mutable std::vector<std::shared_ptr<casadi::Callback>> owned_callbacks_;

};
