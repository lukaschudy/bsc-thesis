#pragma once

#include "globals.hpp"

namespace LearningTrajectory {

void computeLearningTrajectory(
    int iExperiment,
    int codExperiment,
    const std::vector<double>& alpha,
    const std::vector<double>& ExplorationParameters,
    double delta);

}  // namespace LearningTrajectory
