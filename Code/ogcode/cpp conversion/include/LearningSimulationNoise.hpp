#pragma once

#include "globals.hpp"

namespace LearningSimulationNoise {

// Drop-in replacement for LearningSimulation::computeExperiment.
// sigma = 0  ->  identical behaviour to the baseline.
// sigma > 0  ->  each agent observes the opponent's price with i.i.d.
//                Gaussian noise on the price INDEX, rounded to the grid
//                and clamped to [1, numPrices].
void computeExperimentNoise(
    int iExperiment,
    int codExperiment,
    const std::vector<double>& alpha,
    const std::vector<double>& ExplorationParameters,
    double delta,
    double sigma);

}  // namespace LearningSimulationNoise
