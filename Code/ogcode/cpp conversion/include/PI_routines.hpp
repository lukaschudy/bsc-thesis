#pragma once

#include "globals.hpp"

namespace PI_routines {

void computePIMatricesSinghVives(
    const std::vector<double>& DemandParameters,
    const std::vector<double>& NashPrices,
    const std::vector<double>& CoopPrices,
    std::vector<std::vector<double>>& PI,
    std::vector<double>& NashProfits,
    std::vector<double>& CoopProfits,
    std::vector<int>& indexNashPrices,
    std::vector<int>& indexCoopPrices,
    std::vector<double>& NashMarketShares,
    std::vector<double>& CoopMarketShares,
    std::vector<std::vector<double>>& PricesGrids);

void computePIMatricesLogit(
    const std::vector<double>& DemandParameters,
    const std::vector<double>& NashPrices,
    const std::vector<double>& CoopPrices,
    std::vector<std::vector<double>>& PI,
    std::vector<double>& NashProfits,
    std::vector<double>& CoopProfits,
    std::vector<int>& indexNashPrices,
    std::vector<int>& indexCoopPrices,
    std::vector<double>& NashMarketShares,
    std::vector<double>& CoopMarketShares,
    std::vector<std::vector<double>>& PricesGrids);

void computePIMatricesLogitMu0(
    const std::vector<double>& DemandParameters,
    const std::vector<double>& NashPrices,
    const std::vector<double>& CoopPrices,
    std::vector<std::vector<double>>& PI,
    std::vector<double>& NashProfits,
    std::vector<double>& CoopProfits,
    std::vector<int>& indexNashPrices,
    std::vector<int>& indexCoopPrices,
    std::vector<double>& NashMarketShares,
    std::vector<double>& CoopMarketShares,
    std::vector<std::vector<double>>& PricesGrids);

std::vector<double> logitDemands(
    double a0,
    const std::vector<double>& a,
    const std::vector<double>& c,
    double mu,
    const std::vector<double>& p);

std::vector<double> linearDemands(double gamma, const std::vector<double>& p);

}  // namespace PI_routines
