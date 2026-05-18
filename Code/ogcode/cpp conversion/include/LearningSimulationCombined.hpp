#pragma once

#include "globals.hpp"

namespace LearningSimulationCombined {

// Drop-in replacement for LearningSimulation::computeExperiment that
// applies all three frictions simultaneously.
//
//   latency  (L)   : each agent observes the opponent's price with L
//                    periods of delay. L=0 => no delay.
//   sigma    (sigma) : Gaussian measurement error with std `sigma` on the
//                      price INDEX of the (possibly delayed) opponent
//                      observation, clamped to [1, numPrices]. sigma=0
//                      => no noise.
//   lockIn   (T)   : strict deterministic alternation. T=0 => both move
//                    every period (simultaneous Bertrand). T>=1 =>
//                    movingAgent(iter) = ((iter-1)/T) mod 2 + 1.
//
// At (L=0, sigma=0, T=0) this reproduces the baseline bit-for-bit.
void computeExperimentCombined(
    int iExperiment,
    int codExperiment,
    const std::vector<double>& alpha,
    const std::vector<double>& ExplorationParameters,
    double delta,
    int latency,
    double sigma,
    int lockIn,
    int playbackPeriods = 30);

}  // namespace LearningSimulationCombined
