#include "QGapToMaximum.hpp"

#include "EquilibriumCheck.hpp"
#include "QL_routines.hpp"
#include "generic_routines.hpp"

#include <cmath>
#include <iostream>
#include <limits>

namespace QGapToMaximum {

namespace {

double masked_average(
    const std::vector<std::vector<double>>& x,
    const std::vector<std::vector<bool>>& mask,
    int s1,
    int e1,
    int s2,
    int e2) {
    double sum = 0.0;
    int cnt = 0;
    for (int i = s1; i <= e1; ++i) {
        for (int j = s2; j <= e2; ++j) {
            if (mask[i][j]) {
                sum += x[i][j];
                ++cnt;
            }
        }
    }
    if (cnt == 0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return sum / static_cast<double>(cnt);
}

bool any_le_zero(const std::vector<double>& v, int hi) {
    for (int i = 0; i <= hi; ++i) {
        if (v[i] <= 0.0) {
            return true;
        }
    }
    return false;
}

}  // namespace

void computeQGapToMaxSession(
    const std::vector<std::vector<int>>& OptimalStrategy,
    int CycleLength,
    const std::vector<int>& CycleStates,
    std::vector<double>& QGapTot,
    std::vector<double>& QGapOnPath,
    std::vector<double>& QGapNotOnPath,
    std::vector<double>& QGapNotBRAllStates,
    std::vector<double>& QGapNotBRonPath,
    std::vector<double>& QGapNotEqAllStates,
    std::vector<double>& QGapNotEqonPath) {
    using namespace globals;

    auto QTrue = runtime::make3<double>(numStates, numPrices, numAgents, 0.0);
    auto MaxQTrue = runtime::make2<double>(numStates, numAgents, 0.0);
    auto QGap = runtime::make2<double>(numStates, numAgents, 0.0);

    for (int iState = 1; iState <= numStates; ++iState) {
        for (int iAgent = 1; iAgent <= numAgents; ++iAgent) {
            for (int iPrice = 1; iPrice <= numPrices; ++iPrice) {
                auto CellVisitedStates = runtime::make1<int>(numPeriods, 0);
                int CellPreCycleLength = 0;
                int CellCycleLength = 0;
                QL_routines::computeQCell(
                    OptimalStrategy,
                    iState,
                    iPrice,
                    iAgent,
                    delta,
                    QTrue[iState][iPrice][iAgent],
                    CellVisitedStates,
                    CellPreCycleLength,
                    CellCycleLength);
            }

            double m = QTrue[iState][1][iAgent];
            for (int iPrice = 2; iPrice <= numPrices; ++iPrice) {
                m = std::max(m, QTrue[iState][iPrice][iAgent]);
            }
            MaxQTrue[iState][iAgent] = m;

            const double denom = std::abs(m);
            QGap[iState][iAgent] = (m - QTrue[iState][OptimalStrategy[iState][iAgent]][iAgent]) / denom;
        }
    }

    auto IsOnPath = runtime::make2<bool>(numStates, numAgents, false);
    auto IsNotOnPath = runtime::make2<bool>(numStates, numAgents, false);
    auto IsNotBRAllStates = runtime::make2<bool>(numStates, numAgents, false);
    auto IsNotBROnPath = runtime::make2<bool>(numStates, numAgents, false);
    auto IsNotEqAllStates = runtime::make2<bool>(numStates, numAgents, false);
    auto IsNotEqOnPath = runtime::make2<bool>(numStates, numAgents, false);

    for (int iState = 1; iState <= numStates; ++iState) {
        bool onPath = false;
        for (int k = 1; k <= CycleLength; ++k) {
            if (CycleStates[k] == iState) {
                onPath = true;
                break;
            }
        }

        for (int iAgent = 1; iAgent <= numAgents; ++iAgent) {
            IsOnPath[iState][iAgent] = onPath;
            IsNotOnPath[iState][iAgent] = !onPath;
        }

        auto IsBR = runtime::make1<bool>(numAgents, false);
        for (int iAgent = 1; iAgent <= numAgents; ++iAgent) {
            if (generic_routines::AreEqualReals(
                    QTrue[iState][OptimalStrategy[iState][iAgent]][iAgent],
                    MaxQTrue[iState][iAgent])) {
                IsBR[iAgent] = true;
            } else {
                IsNotBRAllStates[iState][iAgent] = true;
                if (onPath) {
                    IsNotBROnPath[iState][iAgent] = true;
                }
            }
        }

        bool allBR = true;
        for (int iAgent = 1; iAgent <= numAgents; ++iAgent) {
            if (!IsBR[iAgent]) {
                allBR = false;
                break;
            }
        }

        if (!allBR) {
            for (int iAgent = 1; iAgent <= numAgents; ++iAgent) {
                IsNotEqAllStates[iState][iAgent] = true;
                if (onPath) {
                    IsNotEqOnPath[iState][iAgent] = true;
                }
            }
        }
    }

    double sumAll = 0.0;
    for (int iState = 1; iState <= numStates; ++iState) {
        for (int iAgent = 1; iAgent <= numAgents; ++iAgent) {
            sumAll += QGap[iState][iAgent];
        }
    }

    QGapTot[0] = sumAll / static_cast<double>(numAgents * numStates);
    QGapOnPath[0] = masked_average(QGap, IsOnPath, 1, numStates, 1, numAgents);
    QGapNotOnPath[0] = masked_average(QGap, IsNotOnPath, 1, numStates, 1, numAgents);
    QGapNotBRAllStates[0] = masked_average(QGap, IsNotBRAllStates, 1, numStates, 1, numAgents);
    QGapNotBRonPath[0] = masked_average(QGap, IsNotBROnPath, 1, numStates, 1, numAgents);
    QGapNotEqAllStates[0] = masked_average(QGap, IsNotEqAllStates, 1, numStates, 1, numAgents);
    QGapNotEqonPath[0] = masked_average(QGap, IsNotEqOnPath, 1, numStates, 1, numAgents);

    for (int iAgent = 1; iAgent <= numAgents; ++iAgent) {
        double s = 0.0;
        for (int iState = 1; iState <= numStates; ++iState) {
            s += QGap[iState][iAgent];
        }
        QGapTot[iAgent] = s / static_cast<double>(numStates);

        auto maskOnPath = runtime::make2<bool>(numStates, 1, false);
        auto maskNotOnPath = runtime::make2<bool>(numStates, 1, false);
        auto maskNotBRAll = runtime::make2<bool>(numStates, 1, false);
        auto maskNotBROn = runtime::make2<bool>(numStates, 1, false);
        auto maskNotEqAll = runtime::make2<bool>(numStates, 1, false);
        auto maskNotEqOn = runtime::make2<bool>(numStates, 1, false);
        auto qcol = runtime::make2<double>(numStates, 1, 0.0);

        for (int iState = 1; iState <= numStates; ++iState) {
            qcol[iState][1] = QGap[iState][iAgent];
            maskOnPath[iState][1] = IsOnPath[iState][iAgent];
            maskNotOnPath[iState][1] = IsNotOnPath[iState][iAgent];
            maskNotBRAll[iState][1] = IsNotBRAllStates[iState][iAgent];
            maskNotBROn[iState][1] = IsNotBROnPath[iState][iAgent];
            maskNotEqAll[iState][1] = IsNotEqAllStates[iState][iAgent];
            maskNotEqOn[iState][1] = IsNotEqOnPath[iState][iAgent];
        }

        QGapOnPath[iAgent] = masked_average(qcol, maskOnPath, 1, numStates, 1, 1);
        QGapNotOnPath[iAgent] = masked_average(qcol, maskNotOnPath, 1, numStates, 1, 1);
        QGapNotBRAllStates[iAgent] = masked_average(qcol, maskNotBRAll, 1, numStates, 1, 1);
        QGapNotBRonPath[iAgent] = masked_average(qcol, maskNotBROn, 1, numStates, 1, 1);
        QGapNotEqAllStates[iAgent] = masked_average(qcol, maskNotEqAll, 1, numStates, 1, 1);
        QGapNotEqonPath[iAgent] = masked_average(qcol, maskNotEqOn, 1, numStates, 1, 1);
    }

    for (int i = 0; i <= numAgents; ++i) {
        if (std::isnan(QGapTot[i])) {
            QGapTot[i] = -999.999;
        }
        if (std::isnan(QGapOnPath[i])) {
            QGapOnPath[i] = -999.999;
        }
        if (std::isnan(QGapNotOnPath[i])) {
            QGapNotOnPath[i] = -999.999;
        }
        if (std::isnan(QGapNotBRAllStates[i])) {
            QGapNotBRAllStates[i] = -999.999;
        }
        if (std::isnan(QGapNotBRonPath[i])) {
            QGapNotBRonPath[i] = -999.999;
        }
        if (std::isnan(QGapNotEqAllStates[i])) {
            QGapNotEqAllStates[i] = -999.999;
        }
        if (std::isnan(QGapNotEqonPath[i])) {
            QGapNotEqonPath[i] = -999.999;
        }
    }
}

void computeQGapToMax(int iExperiment) {
    using namespace globals;

    constexpr int numThresPathCycleLength = 10;
    auto ThresPathCycleLength = runtime::make1<int>(numThresPathCycleLength, 0);
    for (int i = 1; i <= numThresPathCycleLength; ++i) {
        ThresPathCycleLength[i] = i;
    }

    std::cout << "Computing Q gaps\n";

    auto SumQGapTot = runtime::make2<double>(numThresPathCycleLength, numAgents, 0.0);
    auto SumQGapOnPath = runtime::make2<double>(numThresPathCycleLength, numAgents, 0.0);
    auto SumQGapNotOnPath = runtime::make2<double>(numThresPathCycleLength, numAgents, 0.0);
    auto SumQGapNotBRAllStates = runtime::make2<double>(numThresPathCycleLength, numAgents, 0.0);
    auto SumQGapNotBRonPath = runtime::make2<double>(numThresPathCycleLength, numAgents, 0.0);
    auto SumQGapNotEqAllStates = runtime::make2<double>(numThresPathCycleLength, numAgents, 0.0);
    auto SumQGapNotEqonPath = runtime::make2<double>(numThresPathCycleLength, numAgents, 0.0);

    auto NumQGapTot = runtime::make2<int>(numThresPathCycleLength, numAgents, 0);
    auto NumQGapOnPath = runtime::make2<int>(numThresPathCycleLength, numAgents, 0);
    auto NumQGapNotOnPath = runtime::make2<int>(numThresPathCycleLength, numAgents, 0);
    auto NumQGapNotBRAllStates = runtime::make2<int>(numThresPathCycleLength, numAgents, 0);
    auto NumQGapNotBRonPath = runtime::make2<int>(numThresPathCycleLength, numAgents, 0);
    auto NumQGapNotEqAllStates = runtime::make2<int>(numThresPathCycleLength, numAgents, 0);
    auto NumQGapNotEqonPath = runtime::make2<int>(numThresPathCycleLength, numAgents, 0);

    QL_routines::ReadInfoExperiment();

    for (int iSession = 1; iSession <= numSessions; ++iSession) {
        std::cout << "iSession = " << iSession << "\n";

        auto OptimalStrategyVec = runtime::make1<int>(lengthStrategies, 0);
        for (int k = 1; k <= lengthStrategies; ++k) {
            OptimalStrategyVec[k] = indexStrategies[k][iSession];
        }

        int CycleLengthSession = CycleLength[iSession];
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

        auto QGapTotSession = runtime::make1<double>(numAgents, 0.0);
        auto QGapOnPathSession = runtime::make1<double>(numAgents, 0.0);
        auto QGapNotOnPathSession = runtime::make1<double>(numAgents, 0.0);
        auto QGapNotBRAllStatesSession = runtime::make1<double>(numAgents, 0.0);
        auto QGapNotBRonPathSession = runtime::make1<double>(numAgents, 0.0);
        auto QGapNotEqAllStatesSession = runtime::make1<double>(numAgents, 0.0);
        auto QGapNotEqonPathSession = runtime::make1<double>(numAgents, 0.0);

        computeQGapToMaxSession(
            OptimalStrategy,
            CycleLengthSession,
            CycleStatesSession,
            QGapTotSession,
            QGapOnPathSession,
            QGapNotOnPathSession,
            QGapNotBRAllStatesSession,
            QGapNotBRonPathSession,
            QGapNotEqAllStatesSession,
            QGapNotEqonPathSession);

        const int iThres = std::min(CycleLength[iSession], ThresPathCycleLength[numThresPathCycleLength]);

        if (!any_le_zero(QGapTotSession, numAgents)) {
            for (int j = 0; j <= numAgents; ++j) {
                SumQGapTot[0][j] += QGapTotSession[j];
                SumQGapTot[iThres][j] += QGapTotSession[j];
                NumQGapTot[0][j] += 1;
                NumQGapTot[iThres][j] += 1;
            }
        }
        if (!any_le_zero(QGapOnPathSession, numAgents)) {
            for (int j = 0; j <= numAgents; ++j) {
                SumQGapOnPath[0][j] += QGapOnPathSession[j];
                SumQGapOnPath[iThres][j] += QGapOnPathSession[j];
                NumQGapOnPath[0][j] += 1;
                NumQGapOnPath[iThres][j] += 1;
            }
        }
        if (!any_le_zero(QGapNotOnPathSession, numAgents)) {
            for (int j = 0; j <= numAgents; ++j) {
                SumQGapNotOnPath[0][j] += QGapNotOnPathSession[j];
                SumQGapNotOnPath[iThres][j] += QGapNotOnPathSession[j];
                NumQGapNotOnPath[0][j] += 1;
                NumQGapNotOnPath[iThres][j] += 1;
            }
        }
        if (!any_le_zero(QGapNotBRAllStatesSession, numAgents)) {
            for (int j = 0; j <= numAgents; ++j) {
                SumQGapNotBRAllStates[0][j] += QGapNotBRAllStatesSession[j];
                SumQGapNotBRAllStates[iThres][j] += QGapNotBRAllStatesSession[j];
                NumQGapNotBRAllStates[0][j] += 1;
                NumQGapNotBRAllStates[iThres][j] += 1;
            }
        }
        if (!any_le_zero(QGapNotBRonPathSession, numAgents)) {
            for (int j = 0; j <= numAgents; ++j) {
                SumQGapNotBRonPath[0][j] += QGapNotBRonPathSession[j];
                SumQGapNotBRonPath[iThres][j] += QGapNotBRonPathSession[j];
                NumQGapNotBRonPath[0][j] += 1;
                NumQGapNotBRonPath[iThres][j] += 1;
            }
        }
        if (!any_le_zero(QGapNotEqAllStatesSession, numAgents)) {
            for (int j = 0; j <= numAgents; ++j) {
                SumQGapNotEqAllStates[0][j] += QGapNotEqAllStatesSession[j];
                SumQGapNotEqAllStates[iThres][j] += QGapNotEqAllStatesSession[j];
                NumQGapNotEqAllStates[0][j] += 1;
                NumQGapNotEqAllStates[iThres][j] += 1;
            }
        }
        if (!any_le_zero(QGapNotEqonPathSession, numAgents)) {
            for (int j = 0; j <= numAgents; ++j) {
                SumQGapNotEqonPath[0][j] += QGapNotEqonPathSession[j];
                SumQGapNotEqonPath[iThres][j] += QGapNotEqonPathSession[j];
                NumQGapNotEqonPath[0][j] += 1;
                NumQGapNotEqonPath[iThres][j] += 1;
            }
        }
    }

    auto finalize = [](std::vector<std::vector<double>>& sum, const std::vector<std::vector<int>>& num, int n1, int n2) {
        for (int i = 0; i <= n1; ++i) {
            for (int j = 0; j <= n2; ++j) {
                if (num[i][j] > 0) {
                    sum[i][j] /= static_cast<double>(num[i][j]);
                } else {
                    sum[i][j] = -999.999;
                }
                if (std::isnan(sum[i][j])) {
                    sum[i][j] = -999.999;
                }
            }
        }
    };

    finalize(SumQGapTot, NumQGapTot, numThresPathCycleLength, numAgents);
    finalize(SumQGapOnPath, NumQGapOnPath, numThresPathCycleLength, numAgents);
    finalize(SumQGapNotOnPath, NumQGapNotOnPath, numThresPathCycleLength, numAgents);
    finalize(SumQGapNotBRAllStates, NumQGapNotBRAllStates, numThresPathCycleLength, numAgents);
    finalize(SumQGapNotBRonPath, NumQGapNotBRonPath, numThresPathCycleLength, numAgents);
    finalize(SumQGapNotEqAllStates, NumQGapNotEqAllStates, numThresPathCycleLength, numAgents);
    finalize(SumQGapNotEqonPath, NumQGapNotEqonPath, numThresPathCycleLength, numAgents);

    std::fstream& out = runtime::io::unit(10006);
    if (iExperiment == 1) {
        out << "QGapToMaximum\n";
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

    out << SumQGapTot[0][0] << ' ' << SumQGapOnPath[0][0] << ' ' << SumQGapNotOnPath[0][0] << ' '
        << SumQGapNotBRAllStates[0][0] << ' ' << SumQGapNotBRonPath[0][0] << ' ' << SumQGapNotEqAllStates[0][0]
        << ' ' << SumQGapNotEqonPath[0][0] << ' ';

    for (int i = 1; i <= numAgents; ++i) {
        out << SumQGapTot[0][i] << ' ' << SumQGapOnPath[0][i] << ' ' << SumQGapNotOnPath[0][i] << ' '
            << SumQGapNotBRAllStates[0][i] << ' ' << SumQGapNotBRonPath[0][i] << ' '
            << SumQGapNotEqAllStates[0][i] << ' ' << SumQGapNotEqonPath[0][i] << ' ';
    }

    for (int j = 1; j <= numThresPathCycleLength; ++j) {
        out << SumQGapTot[j][0] << ' ' << SumQGapOnPath[j][0] << ' ' << SumQGapNotOnPath[j][0] << ' '
            << SumQGapNotBRAllStates[j][0] << ' ' << SumQGapNotBRonPath[j][0] << ' '
            << SumQGapNotEqAllStates[j][0] << ' ' << SumQGapNotEqonPath[j][0] << ' ';

        for (int i = 1; i <= numAgents; ++i) {
            out << SumQGapTot[j][i] << ' ' << SumQGapOnPath[j][i] << ' ' << SumQGapNotOnPath[j][i] << ' '
                << SumQGapNotBRAllStates[j][i] << ' ' << SumQGapNotBRonPath[j][i] << ' '
                << SumQGapNotEqAllStates[j][i] << ' ' << SumQGapNotEqonPath[j][i] << ' ';
        }
    }

    out << '\n';
}

}  // namespace QGapToMaximum
