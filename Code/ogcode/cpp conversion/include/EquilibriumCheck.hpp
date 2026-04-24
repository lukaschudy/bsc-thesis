#pragma once

#include "globals.hpp"

namespace EquilibriumCheck {

void computeEqCheck(int iExperiment);

void computeEqCheckSession(
    const std::vector<std::vector<int>>& OptimalStrategy,
    int CycleLengthSession,
    const std::vector<int>& CycleStatesSession,
    std::vector<double>& freqBRAll,
    std::vector<double>& freqBROnPath,
    std::vector<double>& freqBROffPath,
    double& freqEQAll,
    double& freqEQOnPath,
    double& freqEQOffPath,
    std::vector<int>& flagBRAll,
    std::vector<int>& flagBROnPath,
    std::vector<int>& flagBROffPath,
    int& flagEQAll,
    int& flagEQOnPath,
    int& flagEQOffPath);

}  // namespace EquilibriumCheck
