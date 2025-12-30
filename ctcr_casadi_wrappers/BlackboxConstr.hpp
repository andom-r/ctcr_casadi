// BlackboxConstr.hpp
#pragma once

#include <casadi/casadi.hpp>
#include "CtrModel.h"

// class CtrModel;
class BlackboxConstr : public casadi::Callback {
public:
    BlackboxConstr(
        const std::string& name, 
        CtrLib::CtrModel* model_ptr, 
        int nx_, 
        int nu_,
        const CtrLib::computationOptions& opt_LOAD_
    );

public:
    // Number of inputs (x and u)
    casadi_int get_n_in() override;

    // Number of outputs (g)
    casadi_int get_n_out() override;

    std::string get_name_in(casadi_int i) override;
    std::string get_name_out(casadi_int i) override;

    // Sparsity of inputs
    casadi::Sparsity get_sparsity_in(casadi_int i) override;

    // Sparsity of outputs
    casadi::Sparsity get_sparsity_out(casadi_int i) override;

    // bool has_jacobian() const override { return false; }
    // allow FD by advertising capability - DERIVATIVES IMPLEMENTED MANUALLY
    bool has_forward(casadi_int /*nfwd*/) const override;
    bool has_reverse(casadi_int /*nadj*/) const override;

    // Provide Jacobian sparsity
    bool has_jac_sparsity(casadi_int /*i*/, casadi_int /*o*/) const override;

    // Ensure has_forward returns true
    casadi::Function get_forward(casadi_int nfwd,
                                 const std::string& name,
                                 const std::vector<std::string>& inames,
                                 const std::vector<std::string>& onames,
                                 const casadi::Dict& opts) const override;

    casadi::Sparsity get_jac_sparsity(casadi_int i, casadi_int o) const;

    casadi::Function get_reverse(casadi_int nadj,
                                 const std::string& name,
                                 const std::vector<std::string>& inames,
                                 const std::vector<std::string>& onames,
                                 const casadi::Dict& opts) const override;

    casadi::Function get_jacobian(const std::string& name,
                                  const std::vector<std::string>& inames,
                                  const std::vector<std::string>& onames,
                                  const casadi::Dict& opts) const override;

    // using Jacobian for model compute
    std::vector<casadi::DM> eval(const std::vector<casadi::DM>& in) const override;

protected:
    CtrLib::CtrModel& ctrModel;
    
    /// Number of state variables.
    /// Examples:
    ///   - 3  → 3D position only
    ///   - 6  → 3D position + 3D orientation (rotation) (6 Not implemented yet)
    int nx;

    /// Number of control/input variables.
    int nu;

    /// Calculation/load option selector.
    /// Used to control model evaluation behavior.
    /// (Reserved for future extensions.)
    const CtrLib::computationOptions& opt_LOAD;

    /// Number of boundary condition constraints.
    mutable long numberOfModelComputes;    
};
