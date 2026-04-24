#pragma once

#include "fortran_runtime.hpp"

namespace generic_routines {

std::vector<std::vector<double>> ComputeRowSummaryStatistics(
    int m,
    int n,
    const std::vector<std::vector<double>>& x);

bool AreEqualReals(double a, double b);

std::vector<int> convertNumberBase(int n, int b, int l);

double ran2(int& idum, std::vector<int>& iv, int& iy, int& idum2);

void generateCombinations(
    int rows,
    int cols,
    const std::vector<int>& lengths,
    const std::vector<std::vector<int>>& x,
    int totrows,
    std::vector<std::vector<int>>& comb);

void updateVectorAverage(int m, int n, const std::vector<double>& x, std::vector<double>& xbar);

void updateScalarAverage(int n, double x, double& xbar);

void MaxLocBreakTies(
    int n,
    const std::vector<double>& x,
    int& idumQ,
    std::vector<int>& ivQ,
    int& iyQ,
    int& idum2Q,
    double& m,
    int& p);

}  // namespace generic_routines
