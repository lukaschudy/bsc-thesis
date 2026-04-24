#pragma once

#include "globals.hpp"

namespace QGapToMaximum {

void computeQGapToMax(int iExperiment);

void computeQGapToMaxSession(
    const std::vector<std::vector<int>>& OptimalStrategy,
    int CycleLength,
    const std::vector<int>& CycleStates,
    std::vector<double>& QGapTot,
    std::vector<double>& QGapOnPath,
    std::vector<double>& QGapNotOnPath,
    std::vector<double>& QGapNotBRAllStates,
    std::vector<double>& QGapNotBRonPath,
    std::vector<double>& QGapNotEqAllStates,
    std::vector<double>& QGapNotEqonPath);

}  // namespace QGapToMaximum
