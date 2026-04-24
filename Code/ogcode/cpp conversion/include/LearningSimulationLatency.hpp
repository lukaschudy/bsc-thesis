#pragma once

#include "globals.hpp"

namespace LearningSimulationLatency {

// Drop-in replacement for LearningSimulation::computeExperiment.
// latency = 0  ->  identical behaviour to the baseline.
// latency = L  ->  each agent observes the opponent's price with a delay of L periods.
void computeExperimentLatency(
    int iExperiment,
    int codExperiment,
    const std::vector<double>& alpha,
    const std::vector<double>& ExplorationParameters,
    double delta,
    int latency);

}  // namespace LearningSimulationLatency
