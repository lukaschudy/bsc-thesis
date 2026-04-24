#pragma once

#include "globals.hpp"

namespace QL_routines {

void initQMatrices(
    int iSession,
    int& idumQ,
    std::vector<int>& ivQ,
    int& iyQ,
    int& idum2Q,
    const std::vector<std::vector<double>>& PI,
    double delta,
    std::vector<std::vector<std::vector<double>>>& Q,
    std::vector<std::vector<double>>& maxValQ,
    std::vector<std::vector<int>>& maxLocQ);

void initState(
    const std::vector<std::vector<double>>& u,
    std::vector<std::vector<int>>& p,
    int& stateNumber,
    int& actionNumber);

void generate_uIniPrice(
    std::vector<std::vector<std::vector<double>>>& uIniPrice,
    int& idum,
    std::vector<int>& iv,
    int& iy,
    int& idum2);

void generateUExploration(
    std::vector<std::vector<double>>& uExploration,
    int& idum,
    std::vector<int>& iv,
    int& iy,
    int& idum2);

int computeStateNumber(const std::vector<std::vector<int>>& p);

int computeActionNumber(const std::vector<int>& p);

std::vector<std::string> computeStatesCodePrint();

std::vector<int> computeStrategyNumber(const std::vector<std::vector<int>>& maxLocQ);

void computeQCell(
    const std::vector<std::vector<int>>& OptimalStrategy,
    int iState,
    int iPrice,
    int iAgent,
    double delta,
    double& QCell,
    std::vector<int>& VisitedStates,
    int& PreCycleLength,
    int& CycleLength);

void ReadInfoExperiment();

}  // namespace QL_routines
