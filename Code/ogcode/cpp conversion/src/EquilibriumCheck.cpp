#include "EquilibriumCheck.hpp"

#include "QL_routines.hpp"
#include "generic_routines.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>

namespace EquilibriumCheck {

namespace {

double safe_div(double num, double den) {
    if (den == 0.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return num / den;
}

}  // namespace

void computeEqCheckSession(
    const std::vector<std::vector<int>>& OptimalStrategy,
    int CycleLengthSession,
    const std::vector<int>& CycleStatesSession,
    std::vector<double>& freqBRAll,
    std::vector<double>& freqBROnPath,
    std::vector<double>& freqBROffPath,
    double& freqEQAll,
    double& freqEQOnPath,
    double& freqEQOffPath,
    std::vector<int>& flagBRAll,
    std::vector<int>& flagBROnPath,
    std::vector<int>& flagBROffPath,
    int& flagEQAll,
    int& flagEQOnPath,
    int& flagEQOffPath) {
    using namespace globals;

    auto IsBestReply = runtime::make2<int>(numStates, numAgents, 0);

    for (int iState = 1; iState <= numStates; ++iState) {
        for (int iAgent = 1; iAgent <= numAgents; ++iAgent) {
            auto StateValueFunction = runtime::make1<double>(numPrices, 0.0);

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
                    StateValueFunction[iPrice],
                    VisitedStates,
                    PreCycleLength,
                    CycleLengthLocal);
            }

            double MaxStateValueFunction = StateValueFunction[1];
            for (int iPrice = 2; iPrice <= numPrices; ++iPrice) {
                MaxStateValueFunction = std::max(MaxStateValueFunction, StateValueFunction[iPrice]);
            }

            const int StrategyPrice = OptimalStrategy[iState][iAgent];
            bool strategyIsBest = false;
            for (int iPrice = 1; iPrice <= numPrices; ++iPrice) {
                const double TestDiff = std::abs(StateValueFunction[iPrice] - MaxStateValueFunction);
                if (TestDiff <= 0.0 && iPrice == StrategyPrice) {
                    strategyIsBest = true;
                }
            }
            if (strategyIsBest) {
                IsBestReply[iState][iAgent] = 1;
            }
        }
    }

    auto numStatesBRAll = runtime::make1<int>(numAgents, 0);
    auto numStatesBROnPath = runtime::make1<int>(numAgents, 0);
    auto numStatesBROffPath = runtime::make1<int>(numAgents, 0);

    for (int iAgent = 1; iAgent <= numAgents; ++iAgent) {
        int sAll = 0;
        for (int iState = 1; iState <= numStates; ++iState) {
            sAll += IsBestReply[iState][iAgent];
        }
        numStatesBRAll[iAgent] = sAll;

        int sOnPath = 0;
        for (int k = 1; k <= CycleLengthSession; ++k) {
            sOnPath += IsBestReply[CycleStatesSession[k]][iAgent];
        }
        numStatesBROnPath[iAgent] = sOnPath;
        numStatesBROffPath[iAgent] = sAll - sOnPath;

        flagBRAll[iAgent] = (numStatesBRAll[iAgent] == numStates) ? 1 : 0;
        flagBROnPath[iAgent] = (numStatesBROnPath[iAgent] == CycleLengthSession) ? 1 : 0;
        flagBROffPath[iAgent] = (numStatesBROffPath[iAgent] == (numStates - CycleLengthSession)) ? 1 : 0;
    }

    int numStatesEQAll = 0;
    int numStatesEQOnPath = 0;
    int numStatesEQOffPath = 0;

    for (int iState = 1; iState <= numStates; ++iState) {
        bool allBR = true;
        for (int iAgent = 1; iAgent <= numAgents; ++iAgent) {
            if (IsBestReply[iState][iAgent] != 1) {
                allBR = false;
                break;
            }
        }

        if (allBR) {
            ++numStatesEQAll;
            bool onPath = false;
            for (int k = 1; k <= CycleLengthSession; ++k) {
                if (CycleStatesSession[k] == iState) {
                    onPath = true;
                    break;
                }
            }
            if (onPath) {
                ++numStatesEQOnPath;
            } else {
                ++numStatesEQOffPath;
            }
        }
    }

    flagEQAll = (numStatesEQAll == numStates) ? 1 : 0;
    flagEQOnPath = (numStatesEQOnPath == CycleLengthSession) ? 1 : 0;
    flagEQOffPath = (numStatesEQOffPath == (numStates - CycleLengthSession)) ? 1 : 0;

    for (int iAgent = 1; iAgent <= numAgents; ++iAgent) {
        freqBRAll[iAgent] = safe_div(static_cast<double>(numStatesBRAll[iAgent]), static_cast<double>(numStates));
        freqBROnPath[iAgent] =
            safe_div(static_cast<double>(numStatesBROnPath[iAgent]), static_cast<double>(CycleLengthSession));
        freqBROffPath[iAgent] =
            safe_div(static_cast<double>(numStatesBROffPath[iAgent]), static_cast<double>(numStates - CycleLengthSession));
    }

    freqEQAll = safe_div(static_cast<double>(numStatesEQAll), static_cast<double>(numStates));
    freqEQOnPath = safe_div(static_cast<double>(numStatesEQOnPath), static_cast<double>(CycleLengthSession));
    freqEQOffPath = safe_div(static_cast<double>(numStatesEQOffPath), static_cast<double>(numStates - CycleLengthSession));
}

void computeEqCheck(int iExperiment) {
    using namespace globals;

    std::cout << "Computing equilibrium checks\n";

    auto ThresCycleLength = runtime::make1<int>(numThresCycleLength, 0);
    for (int i = 1; i <= numThresCycleLength; ++i) {
        ThresCycleLength[i] = i;
    }

    QL_routines::ReadInfoExperiment();

    auto freqBRAll = runtime::make2<double>(numAgents, numSessions, 0.0);
    auto freqBROnPath = runtime::make2<double>(numAgents, numSessions, 0.0);
    auto freqBROffPath = runtime::make2<double>(numAgents, numSessions, 0.0);
    auto freqEQAll = runtime::make1<double>(numSessions, 0.0);
    auto freqEQOnPath = runtime::make1<double>(numSessions, 0.0);
    auto freqEQOffPath = runtime::make1<double>(numSessions, 0.0);

    auto flagBRAll = runtime::make2<int>(numAgents, numSessions, 0);
    auto flagBROnPath = runtime::make2<int>(numAgents, numSessions, 0);
    auto flagBROffPath = runtime::make2<int>(numAgents, numSessions, 0);
    auto flagEQAll = runtime::make1<int>(numSessions, 0);
    auto flagEQOnPath = runtime::make1<int>(numSessions, 0);
    auto flagEQOffPath = runtime::make1<int>(numSessions, 0);

    for (int iSession = 1; iSession <= numSessions; ++iSession) {
        std::cout << "iSession = " << iSession << "\n";

        auto OptimalStrategyVec = runtime::make1<int>(lengthStrategies, 0);
        for (int k = 1; k <= lengthStrategies; ++k) {
            OptimalStrategyVec[k] = indexStrategies[k][iSession];
        }

        const int CycleLengthSession = CycleLength[iSession];
        auto CycleStatesSession = runtime::make1<int>(CycleLengthSession, 0);
        for (int k = 1; k <= CycleLengthSession; ++k) {
            CycleStatesSession[k] = CycleStates[k][iSession];
        }

        auto OptimalStrategy = runtime::make2<int>(numStates, numAgents, 0);
        for (int iAgent = 1; iAgent <= numAgents; ++iAgent) {
            for (int iState = 1; iState <= numStates; ++iState) {
                OptimalStrategy[iState][iAgent] = OptimalStrategyVec[(iAgent - 1) * numStates + iState];
            }
        }

        auto freqBRAllSession = runtime::make1<double>(numAgents, 0.0);
        auto freqBROnPathSession = runtime::make1<double>(numAgents, 0.0);
        auto freqBROffPathSession = runtime::make1<double>(numAgents, 0.0);
        double freqEQAllSession = 0.0;
        double freqEQOnPathSession = 0.0;
        double freqEQOffPathSession = 0.0;
        auto flagBRAllSession = runtime::make1<int>(numAgents, 0);
        auto flagBROnPathSession = runtime::make1<int>(numAgents, 0);
        auto flagBROffPathSession = runtime::make1<int>(numAgents, 0);
        int flagEQAllSession = 0;
        int flagEQOnPathSession = 0;
        int flagEQOffPathSession = 0;

        computeEqCheckSession(
            OptimalStrategy,
            CycleLengthSession,
            CycleStatesSession,
            freqBRAllSession,
            freqBROnPathSession,
            freqBROffPathSession,
            freqEQAllSession,
            freqEQOnPathSession,
            freqEQOffPathSession,
            flagBRAllSession,
            flagBROnPathSession,
            flagBROffPathSession,
            flagEQAllSession,
            flagEQOnPathSession,
            flagEQOffPathSession);

        for (int iAgent = 1; iAgent <= numAgents; ++iAgent) {
            freqBRAll[iAgent][iSession] = freqBRAllSession[iAgent];
            freqBROnPath[iAgent][iSession] = freqBROnPathSession[iAgent];
            freqBROffPath[iAgent][iSession] = freqBROffPathSession[iAgent];
            flagBRAll[iAgent][iSession] = flagBRAllSession[iAgent];
            flagBROnPath[iAgent][iSession] = flagBROnPathSession[iAgent];
            flagBROffPath[iAgent][iSession] = flagBROffPathSession[iAgent];
        }

        freqEQAll[iSession] = freqEQAllSession;
        freqEQOnPath[iSession] = freqEQOnPathSession;
        freqEQOffPath[iSession] = freqEQOffPathSession;
        flagEQAll[iSession] = flagEQAllSession;
        flagEQOnPath[iSession] = flagEQOnPathSession;
        flagEQOffPath[iSession] = flagEQOffPathSession;
    }

    auto AvgFreqBRAll = runtime::make2<double>(numAgents, numThresCycleLength, 0.0);
    auto AvgFreqBROnPath = runtime::make2<double>(numAgents, numThresCycleLength, 0.0);
    auto AvgFreqBROffPath = runtime::make2<double>(numAgents, numThresCycleLength, 0.0);
    auto AvgFreqEQAll = runtime::make1<double>(numThresCycleLength, 0.0);
    auto AvgFreqEQOnPath = runtime::make1<double>(numThresCycleLength, 0.0);
    auto AvgFreqEQOffPath = runtime::make1<double>(numThresCycleLength, 0.0);
    auto AvgFlagBRAll = runtime::make2<double>(numAgents, numThresCycleLength, 0.0);
    auto AvgFlagBROnPath = runtime::make2<double>(numAgents, numThresCycleLength, 0.0);
    auto AvgFlagBROffPath = runtime::make2<double>(numAgents, numThresCycleLength, 0.0);
    auto AvgFlagEQAll = runtime::make1<double>(numThresCycleLength, 0.0);
    auto AvgFlagEQOnPath = runtime::make1<double>(numThresCycleLength, 0.0);
    auto AvgFlagEQOffPath = runtime::make1<double>(numThresCycleLength, 0.0);

    auto numCycleLength = runtime::make1<int>(numThresCycleLength, 0);
    numCycleLength[0] = numSessions;

    {
        const double r_num_ag = static_cast<double>(numAgents * numCycleLength[0]);
        const double r_num = static_cast<double>(numCycleLength[0]);

        double sFreqBRAll = 0.0;
        double sFreqBROnPath = 0.0;
        double sFreqBROffPath = 0.0;
        double sFlagBRAll = 0.0;
        double sFlagBROnPath = 0.0;
        double sFlagBROffPath = 0.0;

        for (int a = 1; a <= numAgents; ++a) {
            for (int s = 1; s <= numSessions; ++s) {
                sFreqBRAll += freqBRAll[a][s];
                sFreqBROnPath += freqBROnPath[a][s];
                sFreqBROffPath += freqBROffPath[a][s];
                sFlagBRAll += static_cast<double>(flagBRAll[a][s]);
                sFlagBROnPath += static_cast<double>(flagBROnPath[a][s]);
                sFlagBROffPath += static_cast<double>(flagBROffPath[a][s]);
            }
        }

        AvgFreqBRAll[0][0] = safe_div(sFreqBRAll, r_num_ag);
        AvgFreqBROnPath[0][0] = safe_div(sFreqBROnPath, r_num_ag);
        AvgFreqBROffPath[0][0] = safe_div(sFreqBROffPath, r_num_ag);
        AvgFlagBRAll[0][0] = safe_div(sFlagBRAll, r_num_ag);
        AvgFlagBROnPath[0][0] = safe_div(sFlagBROnPath, r_num_ag);
        AvgFlagBROffPath[0][0] = safe_div(sFlagBROffPath, r_num_ag);

        double sFreqEQAll = 0.0;
        double sFreqEQOnPath = 0.0;
        double sFreqEQOffPath = 0.0;
        double sFlagEQAll = 0.0;
        double sFlagEQOnPath = 0.0;
        double sFlagEQOffPath = 0.0;

        for (int s = 1; s <= numSessions; ++s) {
            sFreqEQAll += freqEQAll[s];
            sFreqEQOnPath += freqEQOnPath[s];
            sFreqEQOffPath += freqEQOffPath[s];
            sFlagEQAll += static_cast<double>(flagEQAll[s]);
            sFlagEQOnPath += static_cast<double>(flagEQOnPath[s]);
            sFlagEQOffPath += static_cast<double>(flagEQOffPath[s]);
        }

        AvgFreqEQAll[0] = safe_div(sFreqEQAll, r_num);
        AvgFreqEQOnPath[0] = safe_div(sFreqEQOnPath, r_num);
        AvgFreqEQOffPath[0] = safe_div(sFreqEQOffPath, r_num);
        AvgFlagEQAll[0] = safe_div(sFlagEQAll, r_num);
        AvgFlagEQOnPath[0] = safe_div(sFlagEQOnPath, r_num);
        AvgFlagEQOffPath[0] = safe_div(sFlagEQOffPath, r_num);

        for (int iAgent = 1; iAgent <= numAgents; ++iAgent) {
            double sbra = 0.0;
            double sbron = 0.0;
            double sbrof = 0.0;
            double fb = 0.0;
            double fbon = 0.0;
            double fbof = 0.0;
            for (int s = 1; s <= numSessions; ++s) {
                sbra += freqBRAll[iAgent][s];
                sbron += freqBROnPath[iAgent][s];
                sbrof += freqBROffPath[iAgent][s];
                fb += static_cast<double>(flagBRAll[iAgent][s]);
                fbon += static_cast<double>(flagBROnPath[iAgent][s]);
                fbof += static_cast<double>(flagBROffPath[iAgent][s]);
            }
            AvgFreqBRAll[iAgent][0] = safe_div(sbra, r_num);
            AvgFreqBROnPath[iAgent][0] = safe_div(sbron, r_num);
            AvgFreqBROffPath[iAgent][0] = safe_div(sbrof, r_num);
            AvgFlagBRAll[iAgent][0] = safe_div(fb, r_num);
            AvgFlagBROnPath[iAgent][0] = safe_div(fbon, r_num);
            AvgFlagBROffPath[iAgent][0] = safe_div(fbof, r_num);
        }
    }

    for (int iThres = 1; iThres <= numThresCycleLength; ++iThres) {
        std::vector<int> selected;
        selected.reserve(numSessions);

        if (iThres < numThresCycleLength) {
            for (int s = 1; s <= numSessions; ++s) {
                if (CycleLength[s] == ThresCycleLength[iThres]) {
                    selected.push_back(s);
                }
            }
        } else {
            for (int s = 1; s <= numSessions; ++s) {
                if (CycleLength[s] >= ThresCycleLength[iThres]) {
                    selected.push_back(s);
                }
            }
        }

        numCycleLength[iThres] = static_cast<int>(selected.size());
        if (numCycleLength[iThres] == 0) {
            continue;
        }

        const double r_num_ag = static_cast<double>(numAgents * numCycleLength[iThres]);
        const double r_num = static_cast<double>(numCycleLength[iThres]);

        double sFreqBRAll = 0.0;
        double sFreqBROnPath = 0.0;
        double sFreqBROffPath = 0.0;
        double sFlagBRAll = 0.0;
        double sFlagBROnPath = 0.0;
        double sFlagBROffPath = 0.0;

        for (int a = 1; a <= numAgents; ++a) {
            for (int idx : selected) {
                sFreqBRAll += freqBRAll[a][idx];
                sFreqBROnPath += freqBROnPath[a][idx];
                sFreqBROffPath += freqBROffPath[a][idx];
                sFlagBRAll += static_cast<double>(flagBRAll[a][idx]);
                sFlagBROnPath += static_cast<double>(flagBROnPath[a][idx]);
                sFlagBROffPath += static_cast<double>(flagBROffPath[a][idx]);
            }
        }

        AvgFreqBRAll[0][iThres] = safe_div(sFreqBRAll, r_num_ag);
        AvgFreqBROnPath[0][iThres] = safe_div(sFreqBROnPath, r_num_ag);
        AvgFreqBROffPath[0][iThres] = safe_div(sFreqBROffPath, r_num_ag);
        AvgFlagBRAll[0][iThres] = safe_div(sFlagBRAll, r_num_ag);
        AvgFlagBROnPath[0][iThres] = safe_div(sFlagBROnPath, r_num_ag);
        AvgFlagBROffPath[0][iThres] = safe_div(sFlagBROffPath, r_num_ag);

        double sFreqEQAll = 0.0;
        double sFreqEQOnPath = 0.0;
        double sFreqEQOffPath = 0.0;
        double sFlagEQAll = 0.0;
        double sFlagEQOnPath = 0.0;
        double sFlagEQOffPath = 0.0;

        for (int idx : selected) {
            sFreqEQAll += freqEQAll[idx];
            sFreqEQOnPath += freqEQOnPath[idx];
            sFreqEQOffPath += freqEQOffPath[idx];
            sFlagEQAll += static_cast<double>(flagEQAll[idx]);
            sFlagEQOnPath += static_cast<double>(flagEQOnPath[idx]);
            sFlagEQOffPath += static_cast<double>(flagEQOffPath[idx]);
        }

        AvgFreqEQAll[iThres] = safe_div(sFreqEQAll, r_num);
        AvgFreqEQOnPath[iThres] = safe_div(sFreqEQOnPath, r_num);
        AvgFreqEQOffPath[iThres] = safe_div(sFreqEQOffPath, r_num);
        AvgFlagEQAll[iThres] = safe_div(sFlagEQAll, r_num);
        AvgFlagEQOnPath[iThres] = safe_div(sFlagEQOnPath, r_num);
        AvgFlagEQOffPath[iThres] = safe_div(sFlagEQOffPath, r_num);

        for (int iAgent = 1; iAgent <= numAgents; ++iAgent) {
            double sbra = 0.0;
            double sbron = 0.0;
            double sbrof = 0.0;
            double fb = 0.0;
            double fbon = 0.0;
            double fbof = 0.0;
            for (int idx : selected) {
                sbra += freqBRAll[iAgent][idx];
                sbron += freqBROnPath[iAgent][idx];
                sbrof += freqBROffPath[iAgent][idx];
                fb += static_cast<double>(flagBRAll[iAgent][idx]);
                fbon += static_cast<double>(flagBROnPath[iAgent][idx]);
                fbof += static_cast<double>(flagBROffPath[iAgent][idx]);
            }

            AvgFreqBRAll[iAgent][iThres] = safe_div(sbra, r_num);
            AvgFreqBROnPath[iAgent][iThres] = safe_div(sbron, r_num);
            AvgFreqBROffPath[iAgent][iThres] = safe_div(sbrof, r_num);
            AvgFlagBRAll[iAgent][iThres] = safe_div(fb, r_num);
            AvgFlagBROnPath[iAgent][iThres] = safe_div(fbon, r_num);
            AvgFlagBROffPath[iAgent][iThres] = safe_div(fbof, r_num);
        }
    }

    std::fstream& out = runtime::io::unit(10004);
    if (iExperiment == 1) {
        out << "EquilibriumCheck\n";
    }

    out << codExperiment << ' ';
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

    for (int iThres = 0; iThres <= numThresCycleLength; ++iThres) {
        out << numCycleLength[iThres] << ' ';
    }

    for (int iThres = 0; iThres <= numThresCycleLength; ++iThres) {
        out << AvgFlagEQAll[iThres] << ' ' << AvgFlagEQOnPath[iThres] << ' ' << AvgFlagEQOffPath[iThres] << ' '
            << AvgFreqEQAll[iThres] << ' ' << AvgFreqEQOnPath[iThres] << ' ' << AvgFreqEQOffPath[iThres] << ' '
            << AvgFlagBRAll[0][iThres] << ' ' << AvgFlagBROnPath[0][iThres] << ' ' << AvgFlagBROffPath[0][iThres]
            << ' ' << AvgFreqBRAll[0][iThres] << ' ' << AvgFreqBROnPath[0][iThres] << ' '
            << AvgFreqBROffPath[0][iThres] << ' ';
    }

    for (int iAgent = 1; iAgent <= numAgents; ++iAgent) {
        for (int iThres = 0; iThres <= numThresCycleLength; ++iThres) {
            out << AvgFlagBRAll[iAgent][iThres] << ' ' << AvgFlagBROnPath[iAgent][iThres] << ' '
                << AvgFlagBROffPath[iAgent][iThres] << ' ' << AvgFreqBRAll[iAgent][iThres] << ' '
                << AvgFreqBROnPath[iAgent][iThres] << ' ' << AvgFreqBROffPath[iAgent][iThres] << ' ';
        }
    }

    out << '\n';
}

}  // namespace EquilibriumCheck
