#pragma once

#include "globals.hpp"

namespace ImpulseResponse {

void computeIRAnalysis(int iExperiment, int UnitNumber, int IRType);

void ComputeStaticBestResponse(
    const std::vector<std::vector<int>>& OptimalStrategy,
    int iState,
    int iAgent,
    int& IndexStaticBR,
    double& PIStaticBR);

void ComputeDynamicBestResponse(
    const std::vector<std::vector<int>>& OptimalStrategy,
    int iState,
    int iAgent,
    double delta,
    int& IndexDynamicBR,
    double& QDynamicBR);

void computeIndividualIR(
    const std::vector<std::vector<int>>& OptimalStrategy,
    int InitialState,
    int DevAgent,
    int DevPrice,
    int DevLength,
    int DevObsLength,
    int PreCycleLength,
    const std::vector<int>& PreCycleStates,
    std::vector<int>& ShockStates,
    std::vector<std::vector<int>>& ShockIndPrices,
    std::vector<std::vector<double>>& ShockPrices,
    std::vector<std::vector<double>>& ShockProfits,
    std::vector<double>& AvgPostPrices,
    std::vector<double>& AvgPostProfits,
    int& ShockLength,
    int& PunishmentStrategy,
    int& PostLength);

}  // namespace ImpulseResponse
