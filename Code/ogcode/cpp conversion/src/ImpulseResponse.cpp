#include "ImpulseResponse.hpp"

#include "EquilibriumCheck.hpp"
#include "QL_routines.hpp"
#include "generic_routines.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>

namespace ImpulseResponse {

namespace {

int first_repeat_index(const std::vector<int>& v, int upto, int value) {
    for (int i = 1; i <= upto; ++i) {
        if (v[i] == value) {
            return i;
        }
    }
    return 1;
}

std::vector<int> flatten_depth_agent_no_transpose(const std::vector<std::vector<int>>& p, int depth, int agents) {
    auto out = runtime::make1<int>(depth * agents, 0);
    int k = 1;
    for (int a = 1; a <= agents; ++a) {
        for (int d = 1; d <= depth; ++d) {
            out[k++] = p[d][a];
        }
    }
    return out;
}

int argmin_squared_distance_to_value(const std::vector<std::vector<double>>& grid, int agent, double target, int nPrices) {
    int idx = 1;
    double best = (grid[1][agent] - target) * (grid[1][agent] - target);
    for (int i = 2; i <= nPrices; ++i) {
        const double d = (grid[i][agent] - target) * (grid[i][agent] - target);
        if (d < best) {
            best = d;
            idx = i;
        }
    }
    return idx;
}

}  // namespace

void ComputeStaticBestResponse(
    const std::vector<std::vector<int>>& OptimalStrategy,
    int iState,
    int iAgent,
    int& IndexStaticBR,
    double& PIStaticBR) {
    using namespace globals;

    auto pPrime = runtime::make1<int>(numAgents, 0);
    for (int a = 1; a <= numAgents; ++a) {
        pPrime[a] = OptimalStrategy[iState][a];
    }

    auto selProfits = runtime::make1<double>(numPrices, 0.0);
    for (int iPrice = 1; iPrice <= numPrices; ++iPrice) {
        pPrime[iAgent] = iPrice;
        selProfits[iPrice] = PI[QL_routines::computeActionNumber(pPrime)][iAgent];
    }

    double mx = selProfits[1];
    int arg = 1;
    for (int iPrice = 2; iPrice <= numPrices; ++iPrice) {
        if (selProfits[iPrice] > mx) {
            mx = selProfits[iPrice];
            arg = iPrice;
        }
    }

    IndexStaticBR = arg;
    PIStaticBR = mx;
}

void ComputeDynamicBestResponse(
    const std::vector<std::vector<int>>& OptimalStrategy,
    int iState,
    int iAgent,
    double delta,
    int& IndexDynamicBR,
    double& QDynamicBR) {
    using namespace globals;

    auto selQ = runtime::make1<double>(numPrices, 0.0);
    for (int iPrice = 1; iPrice <= numPrices; ++iPrice) {
        auto VisitedStates = runtime::make1<int>(numPeriods, 0);
        int PreCycleLength = 0;
        int CycleLengthLocal = 0;
        QL_routines::computeQCell(
            OptimalStrategy,
            iState,
            iPrice,
            iAgent,
            delta,
            selQ[iPrice],
            VisitedStates,
            PreCycleLength,
            CycleLengthLocal);
    }

    double mx = selQ[1];
    int arg = 1;
    for (int iPrice = 2; iPrice <= numPrices; ++iPrice) {
        if (selQ[iPrice] > mx) {
            mx = selQ[iPrice];
            arg = iPrice;
        }
    }

    IndexDynamicBR = arg;
    QDynamicBR = mx;
}

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
    int& PostLength) {
    using namespace globals;

    auto p = runtime::reshape_to_depth_agent(
        generic_routines::convertNumberBase(InitialState - 1, numPrices, numAgents * DepthState),
        DepthState,
        numAgents);
    auto pPrime = runtime::make1<int>(numAgents, 0);
    for (int a = 1; a <= numAgents; ++a) {
        pPrime[a] = OptimalStrategy[InitialState][a];
    }

    pPrime[DevAgent] = DevPrice;

    const int loopLength = std::max(DevObsLength, numPeriods);
    auto VisitedStates = runtime::make1<int>(loopLength, 0);

    for (int i = 1; i <= DevObsLength; ++i) {
        ShockStates[i] = 0;
        for (int a = 1; a <= numAgents; ++a) {
            ShockIndPrices[i][a] = 0;
            ShockPrices[i][a] = 0.0;
            ShockProfits[i][a] = 0.0;
        }
    }

    bool FlagReturnedToState = false;
    auto indexShockState = runtime::make1<int>(LengthStates, 0);

    for (int iPeriod = 1; iPeriod <= loopLength; ++iPeriod) {
        if (DepthState > 1) {
            for (int d = DepthState; d >= 2; --d) {
                for (int a = 1; a <= numAgents; ++a) {
                    p[d][a] = p[d - 1][a];
                }
            }
        }

        for (int a = 1; a <= numAgents; ++a) {
            p[1][a] = pPrime[a];
        }

        VisitedStates[iPeriod] = QL_routines::computeStateNumber(p);

        for (int jAgent = 1; jAgent <= numAgents; ++jAgent) {
            if (iPeriod <= DevObsLength) {
                ShockStates[iPeriod] = VisitedStates[iPeriod];
                ShockIndPrices[iPeriod][jAgent] = pPrime[jAgent];
                ShockPrices[iPeriod][jAgent] = PricesGrids[pPrime[jAgent]][jAgent];
                ShockProfits[iPeriod][jAgent] = PI[QL_routines::computeActionNumber(pPrime)][jAgent];
            }
        }

        bool onPrePath = false;
        for (int k = 1; k <= PreCycleLength; ++k) {
            if (PreCycleStates[k] == VisitedStates[iPeriod]) {
                onPrePath = true;
                break;
            }
        }

        if (!FlagReturnedToState && onPrePath) {
            ShockLength = iPeriod;
            PunishmentStrategy = iPeriod;
            indexShockState = flatten_depth_agent_no_transpose(p, DepthState, numAgents);
            FlagReturnedToState = true;
        }

        bool repeated = false;
        if (iPeriod >= 2 && !FlagReturnedToState) {
            for (int t = 1; t <= iPeriod - 1; ++t) {
                if (VisitedStates[t] == VisitedStates[iPeriod]) {
                    repeated = true;
                    break;
                }
            }
        }

        if (repeated) {
            ShockLength = first_repeat_index(VisitedStates, iPeriod - 1, VisitedStates[iPeriod]);
            PunishmentStrategy = 0;
            indexShockState = flatten_depth_agent_no_transpose(p, DepthState, numAgents);
            FlagReturnedToState = true;
        }

        for (int a = 1; a <= numAgents; ++a) {
            pPrime[a] = OptimalStrategy[VisitedStates[iPeriod]][a];
        }

        if (DevLength == 1000) {
            pPrime[DevAgent] = DevPrice;
        }
        if (DevLength > iPeriod) {
            pPrime[DevAgent] = DevPrice;
        }
    }

    auto VisitedStatesPost = runtime::make1<int>(numPeriods, 0);
    auto VisitedPrices = runtime::make2<double>(numPeriods, numAgents, 0.0);
    auto VisitedProfits = runtime::make2<double>(numPeriods, numAgents, 0.0);

    p = runtime::reshape_to_depth_agent(indexShockState, DepthState, numAgents);
    {
        const int s = QL_routines::computeStateNumber(p);
        for (int a = 1; a <= numAgents; ++a) {
            pPrime[a] = OptimalStrategy[s][a];
        }
    }

    int iPeriod = 1;
    for (iPeriod = 1; iPeriod <= numPeriods; ++iPeriod) {
        if (DepthState > 1) {
            for (int d = DepthState; d >= 2; --d) {
                for (int a = 1; a <= numAgents; ++a) {
                    p[d][a] = p[d - 1][a];
                }
            }
        }

        for (int a = 1; a <= numAgents; ++a) {
            p[1][a] = pPrime[a];
        }

        VisitedStatesPost[iPeriod] = QL_routines::computeStateNumber(p);
        for (int jAgent = 1; jAgent <= numAgents; ++jAgent) {
            VisitedPrices[iPeriod][jAgent] = PricesGrids[pPrime[jAgent]][jAgent];
            VisitedProfits[iPeriod][jAgent] = PI[QL_routines::computeActionNumber(pPrime)][jAgent];
        }

        bool repeated = false;
        if (iPeriod >= 2) {
            for (int t = 1; t <= iPeriod - 1; ++t) {
                if (VisitedStatesPost[t] == VisitedStatesPost[iPeriod]) {
                    repeated = true;
                    break;
                }
            }
        }

        if (repeated) {
            break;
        }

        for (int a = 1; a <= numAgents; ++a) {
            pPrime[a] = OptimalStrategy[VisitedStatesPost[iPeriod]][a];
        }
    }

    PostLength = iPeriod - first_repeat_index(VisitedStatesPost, iPeriod - 1, VisitedStatesPost[iPeriod]);

    for (int a = 1; a <= numAgents; ++a) {
        double sPrice = 0.0;
        double sProf = 0.0;
        for (int t = iPeriod - PostLength + 1; t <= iPeriod; ++t) {
            sPrice += VisitedPrices[t][a];
            sProf += VisitedProfits[t][a];
        }
        AvgPostPrices[a] = sPrice / static_cast<double>(PostLength);
        AvgPostProfits[a] = sProf / static_cast<double>(PostLength);
    }
}

void computeIRAnalysis(int iExperiment, int UnitNumber, int IRType) {
    using namespace globals;

    constexpr int numThresPeriodsLength = 10;
    constexpr int numThresPeriodsLength0 = numThresPeriodsLength + 1;
    constexpr int numShockPeriodsPrint = 25;

    std::cout << "Computing Impulse Responses\n";

    QL_routines::ReadInfoExperiment();

    auto ThresPeriodsLength = runtime::make1<int>(numThresPeriodsLength, 0);
    auto ThresPeriodsLength0 = runtime::make1<int>(numThresPeriodsLength0, 0);
    for (int i = 1; i <= numThresPeriodsLength; ++i) {
        ThresPeriodsLength[i] = i;
        ThresPeriodsLength0[i + 1] = i;
    }

    auto FreqPreLength = runtime::make1<int>(numThresPeriodsLength, 0);
    auto AvgPrePrices = runtime::make1<double>(numAgents, 0.0);
    auto AvgPreProfits = runtime::make1<double>(numAgents, 0.0);
    auto AvgPrePricesQ = runtime::make1<double>(numAgents, 0.0);
    auto AvgPreProfitsQ = runtime::make1<double>(numAgents, 0.0);

    auto FreqShockLength = runtime::make2<int>(numAgents, numThresPeriodsLength, 0);
    auto FreqPunishmentStrategy = runtime::make2<int>(numAgents, numThresPeriodsLength, 0);
    auto FreqPostLength = runtime::make2<int>(numAgents, numThresPeriodsLength, 0);

    auto AvgShockPrices = runtime::make3<double>(numShockPeriodsPrint, numAgents, numAgents, 0.0);
    auto AvgShockProfits = runtime::make3<double>(numShockPeriodsPrint, numAgents, numAgents, 0.0);
    auto AvgShockPricesQ = runtime::make3<double>(numShockPeriodsPrint, numAgents, numAgents, 0.0);
    auto AvgShockProfitsQ = runtime::make3<double>(numShockPeriodsPrint, numAgents, numAgents, 0.0);

    auto AvgPostPrices = runtime::make2<double>(numAgents, numAgents, 0.0);
    auto AvgPostProfits = runtime::make2<double>(numAgents, numAgents, 0.0);
    auto AvgPostPricesQ = runtime::make2<double>(numAgents, numAgents, 0.0);
    auto AvgPostProfitsQ = runtime::make2<double>(numAgents, numAgents, 0.0);

    for (int iSession = 1; iSession <= numSessions; ++iSession) {
        std::cout << "iSession = " << iSession << "\n";

        auto OptimalStrategyVec = runtime::make1<int>(lengthStrategies, 0);
        for (int k = 1; k <= lengthStrategies; ++k) {
            OptimalStrategyVec[k] = indexStrategies[k][iSession];
        }

        auto OptimalStrategy = runtime::make2<int>(numStates, numAgents, 0);
        for (int iAgent = 1; iAgent <= numAgents; ++iAgent) {
            for (int iState = 1; iState <= numStates; ++iState) {
                OptimalStrategy[iState][iAgent] = OptimalStrategyVec[(iAgent - 1) * numStates + iState];
            }
        }

        const int PeriodsLengthPre = CycleLength[iSession];
        const int PosThres = std::min(numThresPeriodsLength, PeriodsLengthPre);
        FreqPreLength[PosThres] += 1;

        auto VisitedStatesPre = runtime::make1<int>(numPeriods, 0);
        auto PrePrices = runtime::make2<double>(numPeriods, numAgents, 0.0);
        auto PreProfits = runtime::make2<double>(numPeriods, numAgents, 0.0);

        for (int iPeriod = 1; iPeriod <= PeriodsLengthPre; ++iPeriod) {
            VisitedStatesPre[iPeriod] = CycleStates[iPeriod][iSession];
            for (int iAgent = 1; iAgent <= numAgents; ++iAgent) {
                PrePrices[iPeriod][iAgent] = PricesGrids[CyclePrices[iAgent][iPeriod][iSession]][iAgent];
                PreProfits[iPeriod][iAgent] = CycleProfits[iAgent][iPeriod][iSession];
            }
        }

        for (int iAgent = 1; iAgent <= numAgents; ++iAgent) {
            double sP = 0.0;
            double sPi = 0.0;
            for (int iPeriod = 1; iPeriod <= PeriodsLengthPre; ++iPeriod) {
                sP += PrePrices[iPeriod][iAgent];
                sPi += PreProfits[iPeriod][iAgent];
            }
            const double mP = sP / static_cast<double>(PeriodsLengthPre);
            const double mPi = sPi / static_cast<double>(PeriodsLengthPre);
            AvgPrePrices[iAgent] += mP;
            AvgPrePricesQ[iAgent] += mP * mP;
            AvgPreProfits[iAgent] += mPi;
            AvgPreProfitsQ[iAgent] += mPi * mPi;
        }

        for (int iAgent = 1; iAgent <= numAgents; ++iAgent) {
            auto AvgShockPricesTmp = runtime::make2<double>(numShockPeriodsPrint, numAgents, 0.0);
            auto AvgShockProfitsTmp = runtime::make2<double>(numShockPeriodsPrint, numAgents, 0.0);
            auto AvgPostPricesTmp = runtime::make1<double>(numAgents, 0.0);
            auto AvgPostProfitsTmp = runtime::make1<double>(numAgents, 0.0);

            for (int iStatePre = 1; iStatePre <= PeriodsLengthPre; ++iStatePre) {
                int DevPrice = 1;
                int DevLength = 1;
                double r_tmp = 0.0;

                if (IRType <= -1) {
                    DevPrice = -IRType;
                    DevLength = 1;
                } else if (IRType == 0) {
                    ComputeStaticBestResponse(OptimalStrategy, iStatePre, iAgent, DevPrice, r_tmp);
                    DevLength = 1;
                } else if (IRType >= 1) {
                    DevPrice = argmin_squared_distance_to_value(PricesGrids, iAgent, NashPrices[iAgent], numPrices);
                    DevLength = IRType;
                }

                auto ShockStates = runtime::make1<int>(numShockPeriodsPrint, 0);
                auto ShockIndPrices = runtime::make2<int>(numShockPeriodsPrint, numAgents, 0);
                auto ShockPrices = runtime::make2<double>(numShockPeriodsPrint, numAgents, 0.0);
                auto ShockProfits = runtime::make2<double>(numShockPeriodsPrint, numAgents, 0.0);
                auto PostPrices = runtime::make1<double>(numAgents, 0.0);
                auto PostProfits = runtime::make1<double>(numAgents, 0.0);
                int ShockLength = 0;
                int PunishmentStrategy = 0;
                int PostLength = 0;

                auto PreCycleStates = runtime::make1<int>(PeriodsLengthPre, 0);
                for (int k = 1; k <= PeriodsLengthPre; ++k) {
                    PreCycleStates[k] = VisitedStatesPre[k];
                }

                computeIndividualIR(
                    OptimalStrategy,
                    VisitedStatesPre[iStatePre],
                    iAgent,
                    DevPrice,
                    DevLength,
                    numShockPeriodsPrint,
                    PeriodsLengthPre,
                    PreCycleStates,
                    ShockStates,
                    ShockIndPrices,
                    ShockPrices,
                    ShockProfits,
                    PostPrices,
                    PostProfits,
                    ShockLength,
                    PunishmentStrategy,
                    PostLength);

                const double nn = static_cast<double>(iStatePre);
                for (int iPeriod = 1; iPeriod <= numShockPeriodsPrint; ++iPeriod) {
                    for (int jAgent = 1; jAgent <= numAgents; ++jAgent) {
                        AvgShockPricesTmp[iPeriod][jAgent] =
                            (nn - 1.0) / nn * AvgShockPricesTmp[iPeriod][jAgent] + ShockPrices[iPeriod][jAgent] / nn;
                        AvgShockProfitsTmp[iPeriod][jAgent] =
                            (nn - 1.0) / nn * AvgShockProfitsTmp[iPeriod][jAgent] + ShockProfits[iPeriod][jAgent] / nn;
                    }
                }

                for (int jAgent = 1; jAgent <= numAgents; ++jAgent) {
                    AvgPostPricesTmp[jAgent] = (nn - 1.0) / nn * AvgPostPricesTmp[jAgent] + PostPrices[jAgent] / nn;
                    AvgPostProfitsTmp[jAgent] = (nn - 1.0) / nn * AvgPostProfitsTmp[jAgent] + PostProfits[jAgent] / nn;
                }

                FreqShockLength[iAgent][std::min(numThresPeriodsLength, ShockLength)] += 1;
                FreqPunishmentStrategy[iAgent][std::min(numThresPeriodsLength, PunishmentStrategy)] += 1;
                FreqPostLength[iAgent][std::min(numThresPeriodsLength, PostLength)] += 1;
            }

            for (int iPeriod = 1; iPeriod <= numShockPeriodsPrint; ++iPeriod) {
                for (int jAgent = 1; jAgent <= numAgents; ++jAgent) {
                    AvgShockPrices[iPeriod][iAgent][jAgent] += AvgShockPricesTmp[iPeriod][jAgent];
                    AvgShockPricesQ[iPeriod][iAgent][jAgent] +=
                        AvgShockPricesTmp[iPeriod][jAgent] * AvgShockPricesTmp[iPeriod][jAgent];
                    AvgShockProfits[iPeriod][iAgent][jAgent] += AvgShockProfitsTmp[iPeriod][jAgent];
                    AvgShockProfitsQ[iPeriod][iAgent][jAgent] +=
                        AvgShockProfitsTmp[iPeriod][jAgent] * AvgShockProfitsTmp[iPeriod][jAgent];
                }
            }

            for (int jAgent = 1; jAgent <= numAgents; ++jAgent) {
                AvgPostPrices[iAgent][jAgent] += AvgPostPricesTmp[jAgent];
                AvgPostPricesQ[iAgent][jAgent] += AvgPostPricesTmp[jAgent] * AvgPostPricesTmp[jAgent];
                AvgPostProfits[iAgent][jAgent] += AvgPostProfitsTmp[jAgent];
                AvgPostProfitsQ[iAgent][jAgent] += AvgPostProfitsTmp[jAgent] * AvgPostProfitsTmp[jAgent];
            }
        }
    }

    for (int iAgent = 1; iAgent <= numAgents; ++iAgent) {
        AvgPrePrices[iAgent] /= static_cast<double>(numSessions);
        AvgPreProfits[iAgent] /= static_cast<double>(numSessions);
        AvgPrePricesQ[iAgent] /= static_cast<double>(numSessions);
        AvgPreProfitsQ[iAgent] /= static_cast<double>(numSessions);

        for (int jAgent = 1; jAgent <= numAgents; ++jAgent) {
            AvgPostPrices[iAgent][jAgent] /= static_cast<double>(numSessions);
            AvgPostProfits[iAgent][jAgent] /= static_cast<double>(numSessions);
            AvgPostPricesQ[iAgent][jAgent] /= static_cast<double>(numSessions);
            AvgPostProfitsQ[iAgent][jAgent] /= static_cast<double>(numSessions);
        }
    }

    for (int iPeriod = 1; iPeriod <= numShockPeriodsPrint; ++iPeriod) {
        for (int iAgent = 1; iAgent <= numAgents; ++iAgent) {
            for (int jAgent = 1; jAgent <= numAgents; ++jAgent) {
                AvgShockPrices[iPeriod][iAgent][jAgent] /= static_cast<double>(numSessions);
                AvgShockProfits[iPeriod][iAgent][jAgent] /= static_cast<double>(numSessions);
                AvgShockPricesQ[iPeriod][iAgent][jAgent] /= static_cast<double>(numSessions);
                AvgShockProfitsQ[iPeriod][iAgent][jAgent] /= static_cast<double>(numSessions);
            }
        }
    }

    double AggrPrePrices = 0.0;
    double AggrPreProfits = 0.0;
    double AggrPrePricesQ = 0.0;
    double AggrPreProfitsQ = 0.0;

    for (int iAgent = 1; iAgent <= numAgents; ++iAgent) {
        AggrPrePrices += AvgPrePrices[iAgent];
        AggrPreProfits += AvgPreProfits[iAgent];
        AggrPrePricesQ += AvgPrePricesQ[iAgent];
        AggrPreProfitsQ += AvgPreProfitsQ[iAgent];
    }
    AggrPrePrices /= static_cast<double>(numAgents);
    AggrPreProfits /= static_cast<double>(numAgents);
    AggrPrePricesQ /= static_cast<double>(numAgents);
    AggrPreProfitsQ /= static_cast<double>(numAgents);

    auto AggrDevShockPrices = runtime::make1<double>(numShockPeriodsPrint, 0.0);
    auto AggrNonDevShockPrices = runtime::make1<double>(numShockPeriodsPrint, 0.0);
    auto AggrDevShockProfits = runtime::make1<double>(numShockPeriodsPrint, 0.0);
    auto AggrNonDevShockProfits = runtime::make1<double>(numShockPeriodsPrint, 0.0);
    auto AggrDevShockPricesQ = runtime::make1<double>(numShockPeriodsPrint, 0.0);
    auto AggrNonDevShockPricesQ = runtime::make1<double>(numShockPeriodsPrint, 0.0);
    auto AggrDevShockProfitsQ = runtime::make1<double>(numShockPeriodsPrint, 0.0);
    auto AggrNonDevShockProfitsQ = runtime::make1<double>(numShockPeriodsPrint, 0.0);

    for (int iPeriod = 1; iPeriod <= numShockPeriodsPrint; ++iPeriod) {
        double devP = 0.0;
        double devPi = 0.0;
        double devPQ = 0.0;
        double devPiQ = 0.0;
        double allP = 0.0;
        double allPi = 0.0;
        double allPQ = 0.0;
        double allPiQ = 0.0;

        for (int iAgent = 1; iAgent <= numAgents; ++iAgent) {
            for (int jAgent = 1; jAgent <= numAgents; ++jAgent) {
                allP += AvgShockPrices[iPeriod][iAgent][jAgent];
                allPi += AvgShockProfits[iPeriod][iAgent][jAgent];
                allPQ += AvgShockPricesQ[iPeriod][iAgent][jAgent];
                allPiQ += AvgShockProfitsQ[iPeriod][iAgent][jAgent];
            }
            devP += AvgShockPrices[iPeriod][iAgent][iAgent];
            devPi += AvgShockProfits[iPeriod][iAgent][iAgent];
            devPQ += AvgShockPricesQ[iPeriod][iAgent][iAgent];
            devPiQ += AvgShockProfitsQ[iPeriod][iAgent][iAgent];
        }

        AggrDevShockPrices[iPeriod] = devP / static_cast<double>(numAgents);
        AggrDevShockProfits[iPeriod] = devPi / static_cast<double>(numAgents);
        AggrDevShockPricesQ[iPeriod] = devPQ / static_cast<double>(numAgents);
        AggrDevShockProfitsQ[iPeriod] = devPiQ / static_cast<double>(numAgents);

        AggrNonDevShockPrices[iPeriod] =
            (allP - devP) / static_cast<double>(numAgents * std::max(1, numAgents - 1));
        AggrNonDevShockProfits[iPeriod] =
            (allPi - devPi) / static_cast<double>(numAgents * std::max(1, numAgents - 1));
        AggrNonDevShockPricesQ[iPeriod] =
            (allPQ - devPQ) / static_cast<double>(numAgents * std::max(1, numAgents - 1));
        AggrNonDevShockProfitsQ[iPeriod] =
            (allPiQ - devPiQ) / static_cast<double>(numAgents * std::max(1, numAgents - 1));
    }

    double AggrDevPostPrices = 0.0;
    double AggrNonDevPostPrices = 0.0;
    double AggrDevPostProfits = 0.0;
    double AggrNonDevPostProfits = 0.0;
    double AggrDevPostPricesQ = 0.0;
    double AggrNonDevPostPricesQ = 0.0;
    double AggrDevPostProfitsQ = 0.0;
    double AggrNonDevPostProfitsQ = 0.0;

    double allPostP = 0.0;
    double allPostPi = 0.0;
    double allPostPQ = 0.0;
    double allPostPiQ = 0.0;

    for (int iAgent = 1; iAgent <= numAgents; ++iAgent) {
        for (int jAgent = 1; jAgent <= numAgents; ++jAgent) {
            allPostP += AvgPostPrices[iAgent][jAgent];
            allPostPi += AvgPostProfits[iAgent][jAgent];
            allPostPQ += AvgPostPricesQ[iAgent][jAgent];
            allPostPiQ += AvgPostProfitsQ[iAgent][jAgent];
        }
        AggrDevPostPrices += AvgPostPrices[iAgent][iAgent];
        AggrDevPostProfits += AvgPostProfits[iAgent][iAgent];
        AggrDevPostPricesQ += AvgPostPricesQ[iAgent][iAgent];
        AggrDevPostProfitsQ += AvgPostProfitsQ[iAgent][iAgent];
    }

    AggrNonDevPostPrices =
        (allPostP - AggrDevPostPrices) / static_cast<double>(numAgents * std::max(1, numAgents - 1));
    AggrNonDevPostProfits =
        (allPostPi - AggrDevPostProfits) / static_cast<double>(numAgents * std::max(1, numAgents - 1));
    AggrNonDevPostPricesQ =
        (allPostPQ - AggrDevPostPricesQ) / static_cast<double>(numAgents * std::max(1, numAgents - 1));
    AggrNonDevPostProfitsQ =
        (allPostPiQ - AggrDevPostProfitsQ) / static_cast<double>(numAgents * std::max(1, numAgents - 1));

    AggrDevPostPrices /= static_cast<double>(numAgents);
    AggrDevPostProfits /= static_cast<double>(numAgents);
    AggrDevPostPricesQ /= static_cast<double>(numAgents);
    AggrDevPostProfitsQ /= static_cast<double>(numAgents);

    auto se = [](double mean, double meanSq) { return std::sqrt(std::abs(meanSq - mean * mean)); };

    for (int iAgent = 1; iAgent <= numAgents; ++iAgent) {
        AvgPrePricesQ[iAgent] = se(AvgPrePrices[iAgent], AvgPrePricesQ[iAgent]);
        AvgPreProfitsQ[iAgent] = se(AvgPreProfits[iAgent], AvgPreProfitsQ[iAgent]);

        for (int jAgent = 1; jAgent <= numAgents; ++jAgent) {
            AvgPostPricesQ[iAgent][jAgent] = se(AvgPostPrices[iAgent][jAgent], AvgPostPricesQ[iAgent][jAgent]);
            AvgPostProfitsQ[iAgent][jAgent] = se(AvgPostProfits[iAgent][jAgent], AvgPostProfitsQ[iAgent][jAgent]);
        }
    }

    for (int iPeriod = 1; iPeriod <= numShockPeriodsPrint; ++iPeriod) {
        for (int iAgent = 1; iAgent <= numAgents; ++iAgent) {
            for (int jAgent = 1; jAgent <= numAgents; ++jAgent) {
                AvgShockPricesQ[iPeriod][iAgent][jAgent] =
                    se(AvgShockPrices[iPeriod][iAgent][jAgent], AvgShockPricesQ[iPeriod][iAgent][jAgent]);
                AvgShockProfitsQ[iPeriod][iAgent][jAgent] =
                    se(AvgShockProfits[iPeriod][iAgent][jAgent], AvgShockProfitsQ[iPeriod][iAgent][jAgent]);
            }
        }

        AggrNonDevShockPricesQ[iPeriod] = se(AggrNonDevShockPrices[iPeriod], AggrNonDevShockPricesQ[iPeriod]);
        AggrDevShockPricesQ[iPeriod] = se(AggrDevShockPrices[iPeriod], AggrDevShockPricesQ[iPeriod]);
        AggrNonDevShockProfitsQ[iPeriod] = se(AggrNonDevShockProfits[iPeriod], AggrNonDevShockProfitsQ[iPeriod]);
        AggrDevShockProfitsQ[iPeriod] = se(AggrDevShockProfits[iPeriod], AggrDevShockProfitsQ[iPeriod]);
    }

    AggrPrePricesQ = se(AggrPrePrices, AggrPrePricesQ);
    AggrPreProfitsQ = se(AggrPreProfits, AggrPreProfitsQ);
    AggrNonDevPostPricesQ = se(AggrNonDevPostPrices, AggrNonDevPostPricesQ);
    AggrDevPostPricesQ = se(AggrDevPostPrices, AggrDevPostPricesQ);
    AggrNonDevPostProfitsQ = se(AggrNonDevPostProfits, AggrNonDevPostProfitsQ);
    AggrDevPostProfitsQ = se(AggrDevPostProfits, AggrDevPostProfitsQ);

    std::fstream& out = runtime::io::unit(UnitNumber);
    if (iExperiment == 1) {
        out << "ImpulseResponse\n";
    }

    out << codExperiment << ' ' << IRType << ' ';
    for (int i = 1; i <= numAgents; ++i) {
        out << alpha[i] << ' ';
    }
    for (int i = 1; i <= numExplorationParameters; ++i) {
        out << MExpl[i] << ' ';
    }
    out << delta << ' ';

    for (int i = 1; i <= numAgents; ++i) {
        out << typeQInitialization[i] << ' ';
        for (int j = 1; j <= numAgents; ++j) {
            out << parQInitialization[i][j] << ' ';
        }
    }

    for (int i = 1; i <= numDemandParameters; ++i) {
        out << DemandParameters[i] << ' ';
    }
    for (int i = 1; i <= numAgents; ++i) {
        out << NashPrices[i] << ' ';
    }
    for (int i = 1; i <= numAgents; ++i) {
        out << CoopPrices[i] << ' ';
    }
    for (int i = 1; i <= numAgents; ++i) {
        out << NashProfits[i] << ' ';
    }
    for (int i = 1; i <= numAgents; ++i) {
        out << CoopProfits[i] << ' ';
    }
    for (int i = 1; i <= numAgents; ++i) {
        out << NashMarketShares[i] << ' ';
    }
    for (int i = 1; i <= numAgents; ++i) {
        out << CoopMarketShares[i] << ' ';
    }
    for (int i = 1; i <= numAgents; ++i) {
        for (int j = 1; j <= numPrices; ++j) {
            out << PricesGrids[j][i] << ' ';
        }
    }

    out << AggrPrePrices << ' ';
    for (int i = 1; i <= numShockPeriodsPrint; ++i) {
        out << AggrDevShockPrices[i] << ' ';
    }
    out << AggrDevPostPrices << ' ';
    for (int i = 1; i <= numShockPeriodsPrint; ++i) {
        out << AggrNonDevShockPrices[i] << ' ';
    }
    out << AggrNonDevPostPrices << ' ';

    out << AggrPrePricesQ << ' ';
    for (int i = 1; i <= numShockPeriodsPrint; ++i) {
        out << AggrDevShockPricesQ[i] << ' ';
    }
    out << AggrDevPostPricesQ << ' ';
    for (int i = 1; i <= numShockPeriodsPrint; ++i) {
        out << AggrNonDevShockPricesQ[i] << ' ';
    }
    out << AggrNonDevPostPricesQ << ' ';

    out << AggrPreProfits << ' ';
    for (int i = 1; i <= numShockPeriodsPrint; ++i) {
        out << AggrDevShockProfits[i] << ' ';
    }
    out << AggrDevPostProfits << ' ';
    for (int i = 1; i <= numShockPeriodsPrint; ++i) {
        out << AggrNonDevShockProfits[i] << ' ';
    }
    out << AggrNonDevPostProfits << ' ';

    out << AggrPreProfitsQ << ' ';
    for (int i = 1; i <= numShockPeriodsPrint; ++i) {
        out << AggrDevShockProfitsQ[i] << ' ';
    }
    out << AggrDevPostProfitsQ << ' ';
    for (int i = 1; i <= numShockPeriodsPrint; ++i) {
        out << AggrNonDevShockProfitsQ[i] << ' ';
    }
    out << AggrNonDevPostProfitsQ << ' ';

    for (int i = 1; i <= numThresPeriodsLength; ++i) {
        out << FreqPreLength[i] << ' ';
    }
    for (int iAgent = 1; iAgent <= numAgents; ++iAgent) {
        for (int i = 1; i <= numThresPeriodsLength; ++i) {
            out << FreqPostLength[iAgent][i] << ' ';
        }
    }
    for (int iAgent = 1; iAgent <= numAgents; ++iAgent) {
        for (int i = 1; i <= numThresPeriodsLength; ++i) {
            out << FreqShockLength[iAgent][i] << ' ';
        }
    }
    for (int iAgent = 1; iAgent <= numAgents; ++iAgent) {
        for (int i = 0; i <= numThresPeriodsLength; ++i) {
            out << FreqPunishmentStrategy[iAgent][i] << ' ';
        }
    }

    for (int iAgent = 1; iAgent <= numAgents; ++iAgent) {
        for (int jAgent = 1; jAgent <= numAgents; ++jAgent) {
            out << AvgPrePrices[jAgent] << ' ';
            for (int iPeriod = 1; iPeriod <= numShockPeriodsPrint; ++iPeriod) {
                out << AvgShockPrices[iPeriod][iAgent][jAgent] << ' ';
            }
            out << AvgPostPrices[iAgent][jAgent] << ' ';
        }
    }

    for (int iAgent = 1; iAgent <= numAgents; ++iAgent) {
        for (int jAgent = 1; jAgent <= numAgents; ++jAgent) {
            out << AvgPrePricesQ[jAgent] << ' ';
            for (int iPeriod = 1; iPeriod <= numShockPeriodsPrint; ++iPeriod) {
                out << AvgShockPricesQ[iPeriod][iAgent][jAgent] << ' ';
            }
            out << AvgPostPricesQ[iAgent][jAgent] << ' ';
        }
    }

    for (int iAgent = 1; iAgent <= numAgents; ++iAgent) {
        for (int jAgent = 1; jAgent <= numAgents; ++jAgent) {
            out << AvgPreProfits[jAgent] << ' ';
            for (int iPeriod = 1; iPeriod <= numShockPeriodsPrint; ++iPeriod) {
                out << AvgShockProfits[iPeriod][iAgent][jAgent] << ' ';
            }
            out << AvgPostProfits[iAgent][jAgent] << ' ';
        }
    }

    for (int iAgent = 1; iAgent <= numAgents; ++iAgent) {
        for (int jAgent = 1; jAgent <= numAgents; ++jAgent) {
            out << AvgPreProfitsQ[jAgent] << ' ';
            for (int iPeriod = 1; iPeriod <= numShockPeriodsPrint; ++iPeriod) {
                out << AvgShockProfitsQ[iPeriod][iAgent][jAgent] << ' ';
            }
            out << AvgPostProfitsQ[iAgent][jAgent] << ' ';
        }
    }

    out << '\n';
}

}  // namespace ImpulseResponse
