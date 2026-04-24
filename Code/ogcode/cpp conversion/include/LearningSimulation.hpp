#pragma once

#include "globals.hpp"

namespace LearningSimulation {

void computeExperiment(
    int iExperiment,
    int codExperiment,
    const std::vector<double>& alpha,
    const std::vector<double>& ExplorationParameters,
    double delta);

void computePPrime(
    const std::vector<double>& ExplorationParameters,
    const std::vector<std::vector<double>>& uExploration,
    const std::vector<std::vector<int>>& strategyPrime,
    int state,
    int iIters,
    std::vector<int>& pPrime,
    const std::vector<std::vector<std::vector<double>>>& Q,
    std::vector<double>& eps);

}  // namespace LearningSimulation
