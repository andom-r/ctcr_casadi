// BlackboxConstrJacobian.cpp
#pragma once

#include "BlackboxConstr.hpp"

// This class just extends BlackboxConstr "as is"
// and overrides only eval(...)
class BlackboxConstrJacobian : public BlackboxConstr {
public:
    // Reuse base constructor
    using BlackboxConstr::BlackboxConstr;

    // Overwrite eval
    std::vector<casadi::DM> eval(const std::vector<casadi::DM>& in) const override;
};
