#pragma once

#include "globals.hpp"

namespace LearningSimulationAsync {

// Drop-in replacement for LearningSimulation::computeExperiment.
// lockIn = 0  ->  identical behaviour to the baseline (both agents move
//                 every period, simultaneous Bertrand).
// lockIn = T  ->  strict deterministic alternation with T-period commitment:
//                 movingAgent(iter) = ((iter-1) / T) mod 2 + 1
//                 i.e. agent 1 moves for T consecutive periods, then
//                 agent 2 moves for T, then agent 1 again, and so on.
//                 The non-moving agent's price carries over unchanged.
//                 Both agents still Q-learn from the observed joint
//                 outcome each period.
void computeExperimentAsync(
    int iExperiment,
    int codExperiment,
    const std::vector<double>& alpha,
    const std::vector<double>& ExplorationParameters,
    double delta,
    int lockIn);

}  // namespace LearningSimulationAsync
