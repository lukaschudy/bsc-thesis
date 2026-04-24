#include "PI_routines.hpp"

#include "generic_routines.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

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
    std::vector<std::vector<double>>& PricesGrids) {
    using namespace globals;

    const double gamma = DemandParameters[1];
    std::vector<double> extend = runtime::make1<double>(2, 0.0);
    extend[1] = DemandParameters[2];
    extend[2] = DemandParameters[3];

    NashMarketShares = linearDemands(gamma, NashPrices);
    for (int i = 1; i <= numAgents; ++i) {
        NashProfits[i] = NashPrices[i] * NashMarketShares[i];
    }

    CoopMarketShares = linearDemands(gamma, CoopPrices);
    for (int i = 1; i <= numAgents; ++i) {
        CoopProfits[i] = CoopPrices[i] * CoopMarketShares[i];
    }

    const bool all_pos = (extend[1] > 0.0 && extend[2] > 0.0);
    if (all_pos) {
        for (int iAgent = 1; iAgent <= numAgents; ++iAgent) {
            PricesGrids[1][iAgent] =
                std::max(0.0, NashPrices[iAgent] - extend[1] * (CoopPrices[iAgent] - NashPrices[iAgent]));
            PricesGrids[numPrices][iAgent] =
                std::max(0.0, CoopPrices[iAgent] + extend[2] * (CoopPrices[iAgent] - NashPrices[iAgent]));
        }
    } else if ((extend[1] < 0.0) && (extend[2] >= -std::numeric_limits<double>::epsilon())) {
        for (int iAgent = 1; iAgent <= numAgents; ++iAgent) {
            PricesGrids[1][iAgent] = 0.0;
            PricesGrids[numPrices][iAgent] = std::max(0.0, (1.0 + extend[2]) * CoopPrices[iAgent]);
        }
    }

    std::vector<double> stepPrices = runtime::make1<double>(numAgents, 0.0);
    for (int iAgent = 1; iAgent <= numAgents; ++iAgent) {
        stepPrices[iAgent] = (PricesGrids[numPrices][iAgent] - PricesGrids[1][iAgent]) / static_cast<double>(numPrices - 1);
    }

    for (int i = 2; i <= numPrices - 1; ++i) {
        for (int iAgent = 1; iAgent <= numAgents; ++iAgent) {
            PricesGrids[i][iAgent] = PricesGrids[i - 1][iAgent] + stepPrices[iAgent];
        }
    }

    std::vector<double> prices = runtime::make1<double>(numAgents, 0.0);
    for (int i = 1; i <= numActions; ++i) {
        for (int j = 1; j <= numAgents; ++j) {
            prices[j] = PricesGrids[indexActions[i][j]][j];
        }

        const auto d = linearDemands(gamma, prices);
        for (int j = 1; j <= numAgents; ++j) {
            PI[i][j] = prices[j] * d[j];
        }
    }

    for (int i = 1; i <= numAgents; ++i) {
        indexNashPrices[i] = 0;
        indexCoopPrices[i] = 0;
    }
}

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
    std::vector<std::vector<double>>& PricesGrids) {
    using namespace globals;

    const double a0 = DemandParameters[1];
    std::vector<double> a = runtime::make1<double>(numAgents, 0.0);
    std::vector<double> c = runtime::make1<double>(numAgents, 0.0);

    for (int i = 1; i <= numAgents; ++i) {
        a[i] = DemandParameters[1 + i];
        c[i] = DemandParameters[1 + numAgents + i];
    }

    const double mu = DemandParameters[2 + 2 * numAgents];
    std::vector<double> extend = runtime::make1<double>(2, 0.0);
    extend[1] = DemandParameters[3 + 2 * numAgents];
    extend[2] = DemandParameters[4 + 2 * numAgents];

    NashMarketShares = logitDemands(a0, a, c, mu, NashPrices);
    for (int i = 1; i <= numAgents; ++i) {
        NashProfits[i] = (NashPrices[i] - c[i]) * NashMarketShares[i];
    }

    CoopMarketShares = logitDemands(a0, a, c, mu, CoopPrices);
    for (int i = 1; i <= numAgents; ++i) {
        CoopProfits[i] = (CoopPrices[i] - c[i]) * CoopMarketShares[i];
    }

    const bool all_pos = (extend[1] > 0.0 && extend[2] > 0.0);
    if (all_pos) {
        for (int i = 1; i <= numAgents; ++i) {
            PricesGrids[1][i] = NashPrices[i] - extend[1] * (CoopPrices[i] - NashPrices[i]);
            PricesGrids[numPrices][i] = CoopPrices[i] + extend[2] * (CoopPrices[i] - NashPrices[i]);
        }
    } else if ((extend[1] < 0.0) && (extend[2] >= -std::numeric_limits<double>::epsilon())) {
        for (int i = 1; i <= numAgents; ++i) {
            PricesGrids[1][i] = c[i] + extend[1] * c[i];
            PricesGrids[numPrices][i] = CoopPrices[i] + extend[2] * (CoopPrices[i] - NashPrices[i]);
        }
    }

    std::vector<double> stepPrices = runtime::make1<double>(numAgents, 0.0);
    for (int i = 1; i <= numAgents; ++i) {
        stepPrices[i] = (PricesGrids[numPrices][i] - PricesGrids[1][i]) / static_cast<double>(numPrices - 1);
    }
    for (int i = 2; i <= numPrices - 1; ++i) {
        for (int j = 1; j <= numAgents; ++j) {
            PricesGrids[i][j] = PricesGrids[i - 1][j] + stepPrices[j];
        }
    }

    std::vector<double> prices = runtime::make1<double>(numAgents, 0.0);
    for (int i = 1; i <= numActions; ++i) {
        for (int j = 1; j <= numAgents; ++j) {
            prices[j] = PricesGrids[indexActions[i][j]][j];
        }

        const auto d = logitDemands(a0, a, c, mu, prices);
        for (int j = 1; j <= numAgents; ++j) {
            PI[i][j] = (prices[j] - c[j]) * d[j];
        }
    }

    for (int i = 1; i <= numAgents; ++i) {
        indexNashPrices[i] = 0;
        indexCoopPrices[i] = 0;
    }
}

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
    std::vector<std::vector<double>>& PricesGrids) {
    using namespace globals;

    if (numAgents > 2) {
        throw std::runtime_error("Perfect competition only works with two agents");
    }

    const double a0 = DemandParameters[1];
    (void)a0;
    std::vector<double> a = runtime::make1<double>(numAgents, 0.0);
    std::vector<double> c = runtime::make1<double>(numAgents, 0.0);
    for (int i = 1; i <= numAgents; ++i) {
        a[i] = DemandParameters[1 + i];
        c[i] = DemandParameters[1 + numAgents + i];
    }

    std::vector<double> extend = runtime::make1<double>(2, 0.0);
    extend[1] = DemandParameters[3 + 2 * numAgents];
    extend[2] = DemandParameters[4 + 2 * numAgents];

    for (int i = 1; i <= numAgents; ++i) {
        NashMarketShares[i] = 0.5;
        CoopMarketShares[i] = 0.5;
        NashProfits[i] = (NashPrices[i] - c[i]) * NashMarketShares[i];
        CoopProfits[i] = (CoopPrices[i] - c[i]) * CoopMarketShares[i];
    }

    for (int i = 1; i <= numAgents; ++i) {
        PricesGrids[1][i] = NashPrices[i] - extend[1] * (CoopPrices[i] - NashPrices[i]);
        PricesGrids[numPrices][i] = CoopPrices[i] + extend[2] * (CoopPrices[i] - NashPrices[i]);
    }

    std::vector<double> stepPrices = runtime::make1<double>(numAgents, 0.0);
    for (int i = 1; i <= numAgents; ++i) {
        stepPrices[i] = (PricesGrids[numPrices][i] - PricesGrids[1][i]) / static_cast<double>(numPrices - 1);
    }

    for (int i = 2; i <= numPrices - 1; ++i) {
        for (int j = 1; j <= numAgents; ++j) {
            PricesGrids[i][j] = PricesGrids[i - 1][j] + stepPrices[j];
        }
    }

    std::vector<double> pp = runtime::make1<double>(numAgents, 0.0);
    std::vector<double> d = runtime::make1<double>(numAgents, 0.0);

    for (int i = 1; i <= numActions; ++i) {
        for (int j = 1; j <= numAgents; ++j) {
            pp[j] = PricesGrids[indexActions[i][j]][j];
        }

        if ((pp[1] < a[1]) && (pp[2] < a[2]) && (pp[1] == pp[2])) {
            d[1] = 0.5;
        } else if ((pp[1] < a[1]) && (pp[1] < pp[2])) {
            d[1] = 1.0;
        } else if ((pp[1] > pp[2]) && (pp[2] < a[2])) {
            d[1] = 0.0;
        } else if (generic_routines::AreEqualReals(pp[1], a[1]) && generic_routines::AreEqualReals(pp[2], a[2])) {
            d[1] = 1.0 / 3.0;
        } else if (generic_routines::AreEqualReals(pp[1], a[1]) && (pp[2] > a[2])) {
            d[1] = 0.5;
        } else if ((pp[1] > a[1]) && (pp[2] >= a[2])) {
            d[1] = 0.0;
        }

        if ((pp[2] < a[2]) && (pp[1] < a[1]) && (pp[2] == pp[1])) {
            d[2] = 0.5;
        } else if ((pp[2] < a[2]) && (pp[2] < pp[1])) {
            d[2] = 1.0;
        } else if ((pp[2] > pp[1]) && (pp[1] < a[1])) {
            d[2] = 0.0;
        } else if (generic_routines::AreEqualReals(pp[2], a[2]) && generic_routines::AreEqualReals(pp[1], a[1])) {
            d[2] = 1.0 / 3.0;
        } else if (generic_routines::AreEqualReals(pp[2], a[2]) && (pp[1] > a[1])) {
            d[2] = 0.5;
        } else if ((pp[2] > a[2]) && (pp[1] >= a[1])) {
            d[2] = 0.0;
        }

        for (int j = 1; j <= numAgents; ++j) {
            PI[i][j] = (pp[j] - c[j]) * d[j];
        }
    }

    for (int i = 1; i <= numAgents; ++i) {
        indexNashPrices[i] = 0;
        indexCoopPrices[i] = 0;
    }
}

std::vector<double> logitDemands(
    double a0,
    const std::vector<double>& a,
    const std::vector<double>& c,
    double mu,
    const std::vector<double>& p) {
    using namespace globals;
    (void)c;

    auto out = runtime::make1<double>(numAgents, 0.0);
    double den = std::exp(a0 / mu);
    for (int i = 1; i <= numAgents; ++i) {
        out[i] = std::exp((a[i] - p[i]) / mu);
        den += out[i];
    }
    for (int i = 1; i <= numAgents; ++i) {
        out[i] /= den;
    }
    return out;
}

std::vector<double> linearDemands(double gamma, const std::vector<double>& p) {
    using namespace globals;

    auto out = runtime::make1<double>(numAgents, 0.0);
    out[1] = std::max(0.0, std::min(1.0 - p[1], (1.0 - gamma - p[1] + gamma * p[2]) / (1.0 - gamma * gamma)));
    out[2] = std::max(0.0, std::min(1.0 - p[2], (1.0 - gamma - p[2] + gamma * p[1]) / (1.0 - gamma * gamma)));
    return out;
}

}  // namespace PI_routines
