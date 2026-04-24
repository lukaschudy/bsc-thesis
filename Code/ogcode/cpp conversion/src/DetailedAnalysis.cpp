#include "DetailedAnalysis.hpp"

#include "EquilibriumCheck.hpp"
#include "ImpulseResponse.hpp"
#include "QGapToMaximum.hpp"
#include "QL_routines.hpp"
#include "generic_routines.hpp"
#include "globals.hpp"

#include <fstream>
#include <iostream>

namespace DetailedAnalysis {

void ComputeDetailedAnalysis(int iExperiment) {
    using namespace globals;

    (void)iExperiment;

    std::cout << "Computing Detailed Analysis\n";

    const std::string FileName = "A_det_" + ExperimentNumber;
    std::ofstream out(FileName, std::ios::trunc);

    QL_routines::ReadInfoExperiment();

    out << "Session DevToPrice ShockAgent ObsAgent ...\n";

    for (int iSession = 1; iSession <= numSessions; ++iSession) {
        std::cout << "Session = " << iSession << " started\n";

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
        auto VisitedStatesPre = runtime::make1<int>(numPeriods, 0);
        auto PrePrices = runtime::make2<double>(numPeriods, numAgents, 0.0);
        auto PreProfits = runtime::make2<double>(numPeriods, numAgents, 0.0);
        auto IndPrePrices = runtime::make2<int>(numPeriods, numAgents, 0);

        for (int iPeriod = 1; iPeriod <= PeriodsLengthPre; ++iPeriod) {
            VisitedStatesPre[iPeriod] = CycleStates[iPeriod][iSession];
            for (int iAgent = 1; iAgent <= numAgents; ++iAgent) {
                IndPrePrices[iPeriod][iAgent] = CyclePrices[iAgent][iPeriod][iSession];
                PrePrices[iPeriod][iAgent] = PricesGrids[IndPrePrices[iPeriod][iAgent]][iAgent];
                PreProfits[iPeriod][iAgent] = CycleProfits[iAgent][iPeriod][iSession];
            }
        }

        auto AvgPrePrices = runtime::make1<double>(numAgents, 0.0);
        auto AvgPreProfits = runtime::make1<double>(numAgents, 0.0);
        for (int iAgent = 1; iAgent <= numAgents; ++iAgent) {
            double sP = 0.0;
            double sPi = 0.0;
            for (int iPeriod = 1; iPeriod <= PeriodsLengthPre; ++iPeriod) {
                sP += PrePrices[iPeriod][iAgent];
                sPi += PreProfits[iPeriod][iAgent];
            }
            AvgPrePrices[iAgent] = sP / static_cast<double>(PeriodsLengthPre);
            AvgPreProfits[iAgent] = sPi / static_cast<double>(PeriodsLengthPre);
        }

        auto ProfitGains = runtime::make1<double>(numAgents, 0.0);
        for (int iAgent = 1; iAgent <= numAgents; ++iAgent) {
            ProfitGains[iAgent] = (AvgPreProfits[iAgent] - NashProfits[iAgent]) / (CoopProfits[iAgent] - NashProfits[iAgent]);
        }

        auto freqBRAll = runtime::make1<double>(numAgents, 0.0);
        auto freqBROnPath = runtime::make1<double>(numAgents, 0.0);
        auto freqBROffPath = runtime::make1<double>(numAgents, 0.0);
        auto flagBRAll = runtime::make1<int>(numAgents, 0);
        auto flagBROnPath = runtime::make1<int>(numAgents, 0);
        auto flagBROffPath = runtime::make1<int>(numAgents, 0);
        double freqEQAll = 0.0;
        double freqEQOnPath = 0.0;
        double freqEQOffPath = 0.0;
        int flagEQAll = 0;
        int flagEQOnPath = 0;
        int flagEQOffPath = 0;

        auto CycleStatesSession = runtime::make1<int>(PeriodsLengthPre, 0);
        for (int i = 1; i <= PeriodsLengthPre; ++i) {
            CycleStatesSession[i] = VisitedStatesPre[i];
        }

        EquilibriumCheck::computeEqCheckSession(
            OptimalStrategy,
            PeriodsLengthPre,
            CycleStatesSession,
            freqBRAll,
            freqBROnPath,
            freqBROffPath,
            freqEQAll,
            freqEQOnPath,
            freqEQOffPath,
            flagBRAll,
            flagBROnPath,
            flagBROffPath,
            flagEQAll,
            flagEQOnPath,
            flagEQOffPath);

        auto QGapTotSession = runtime::make1<double>(numAgents, 0.0);
        auto QGapOnPathSession = runtime::make1<double>(numAgents, 0.0);
        auto QGapNotOnPathSession = runtime::make1<double>(numAgents, 0.0);
        auto QGapNotBRAllStatesSession = runtime::make1<double>(numAgents, 0.0);
        auto QGapNotBRonPathSession = runtime::make1<double>(numAgents, 0.0);
        auto QGapNotEqAllStatesSession = runtime::make1<double>(numAgents, 0.0);
        auto QGapNotEqonPathSession = runtime::make1<double>(numAgents, 0.0);

        QGapToMaximum::computeQGapToMaxSession(
            OptimalStrategy,
            PeriodsLengthPre,
            CycleStatesSession,
            QGapTotSession,
            QGapOnPathSession,
            QGapNotOnPathSession,
            QGapNotBRAllStatesSession,
            QGapNotBRonPathSession,
            QGapNotEqAllStatesSession,
            QGapNotEqonPathSession);

        for (int iPrice = 1; iPrice <= numPrices; ++iPrice) {
            for (int iStatePre = 1; iStatePre <= PeriodsLengthPre; ++iStatePre) {
                for (int iAgent = 1; iAgent <= numAgents; ++iAgent) {
                    auto ShockPrices = runtime::make2<int>(numShockPeriodsPrint, numAgents, 0);
                    auto ShockRealPrices = runtime::make2<double>(numShockPeriodsPrint, numAgents, 0.0);
                    auto ShockProfits = runtime::make2<double>(numShockPeriodsPrint, numAgents, 0.0);
                    auto StaticBRPrices = runtime::make2<int>(numShockPeriodsPrint, numAgents, 0);
                    auto DynamicBRPrices = runtime::make2<int>(numShockPeriodsPrint, numAgents, 0);
                    auto OptStratQ = runtime::make2<double>(numShockPeriodsPrint, numAgents, 0.0);
                    auto DynamicBRQ = runtime::make2<double>(numShockPeriodsPrint, numAgents, 0.0);
                    auto DeviationQ = runtime::make1<double>(numAgents, 0.0);
                    auto AvgPostPrices = runtime::make1<double>(numAgents, 0.0);
                    auto AvgPostProfits = runtime::make1<double>(numAgents, 0.0);

                    auto pPrime = runtime::make1<int>(numAgents, 0);
                    for (int a = 1; a <= numAgents; ++a) {
                        pPrime[a] = OptimalStrategy[VisitedStatesPre[iStatePre]][a];
                    }
                    pPrime[iAgent] = iPrice;

                    for (int jAgent = 1; jAgent <= numAgents; ++jAgent) {
                        auto VisitedStatesTMP = runtime::make1<int>(numPeriods, 0);
                        int PreCycleLength = 0;
                        int QCellCycleLength = 0;
                        QL_routines::computeQCell(
                            OptimalStrategy,
                            VisitedStatesPre[iStatePre],
                            pPrime[jAgent],
                            jAgent,
                            delta,
                            DeviationQ[jAgent],
                            VisitedStatesTMP,
                            PreCycleLength,
                            QCellCycleLength);
                    }

                    auto ShockStates = runtime::make1<int>(numShockPeriodsPrint, 0);
                    auto ShockIndPrices = runtime::make2<int>(numShockPeriodsPrint, numAgents, 0);
                    int ShockLength = 0;
                    int SameCyclePrePost = 0;
                    int PostLength = 0;

                    ImpulseResponse::computeIndividualIR(
                        OptimalStrategy,
                        VisitedStatesPre[iStatePre],
                        iAgent,
                        iPrice,
                        1,
                        numShockPeriodsPrint,
                        PeriodsLengthPre,
                        CycleStatesSession,
                        ShockStates,
                        ShockIndPrices,
                        ShockRealPrices,
                        ShockProfits,
                        AvgPostPrices,
                        AvgPostProfits,
                        ShockLength,
                        SameCyclePrePost,
                        PostLength);

                    for (int iPeriod = 1; iPeriod <= numShockPeriodsPrint; ++iPeriod) {
                        auto p = runtime::reshape_to_depth_agent(
                            generic_routines::convertNumberBase(ShockStates[iPeriod] - 1, numPrices, numAgents * DepthState),
                            DepthState,
                            numAgents);

                        for (int a = 1; a <= numAgents; ++a) {
                            ShockPrices[iPeriod][a] = p[1][a];
                        }

                        const int iPeriodState = (iPeriod == 1) ? VisitedStatesPre[iStatePre] : ShockStates[iPeriod - 1];

                        for (int jAgent = 1; jAgent <= numAgents; ++jAgent) {
                            ImpulseResponse::ComputeDynamicBestResponse(
                                OptimalStrategy,
                                iPeriodState,
                                jAgent,
                                delta,
                                DynamicBRPrices[iPeriod][jAgent],
                                DynamicBRQ[iPeriod][jAgent]);

                            auto VisitedStatesTMP = runtime::make1<int>(numPeriods, 0);
                            int PreCycleLength = 0;
                            int QCellCycleLength = 0;
                            QL_routines::computeQCell(
                                OptimalStrategy,
                                iPeriodState,
                                OptimalStrategy[iPeriodState][jAgent],
                                jAgent,
                                delta,
                                OptStratQ[iPeriod][jAgent],
                                VisitedStatesTMP,
                                PreCycleLength,
                                QCellCycleLength);

                            double PIStaticBR = 0.0;
                            ImpulseResponse::ComputeStaticBestResponse(
                                OptimalStrategy,
                                VisitedStatesPre[iStatePre],
                                jAgent,
                                StaticBRPrices[iPeriod][jAgent],
                                PIStaticBR);
                        }
                    }

                    for (int jAgent = 1; jAgent <= numAgents; ++jAgent) {
                        out << iSession << ' ' << iPrice << ' ';
                        for (int a = 1; a <= numAgents; ++a) {
                            out << NashProfits[a] << ' ';
                        }
                        for (int a = 1; a <= numAgents; ++a) {
                            out << CoopProfits[a] << ' ';
                        }
                        out << PeriodsLengthPre << ' ';
                        for (int a = 1; a <= numAgents; ++a) {
                            out << AvgPrePrices[a] << ' ';
                        }
                        for (int a = 1; a <= numAgents; ++a) {
                            out << AvgPreProfits[a] << ' ';
                        }
                        for (int a = 1; a <= numAgents; ++a) {
                            out << ProfitGains[a] << ' ';
                        }

                        out << converged[iSession] << ' ' << timeToConvergence[iSession] << ' ' << iStatePre << ' ';
                        out << flagEQAll << ' ' << flagEQOnPath << ' ' << flagEQOffPath << ' ';
                        out << freqEQAll << ' ' << freqEQOnPath << ' ' << freqEQOffPath << ' ';

                        for (int a = 1; a <= numAgents; ++a) {
                            out << flagBRAll[a] << ' ';
                        }
                        for (int a = 1; a <= numAgents; ++a) {
                            out << flagBROnPath[a] << ' ';
                        }
                        for (int a = 1; a <= numAgents; ++a) {
                            out << flagBROffPath[a] << ' ';
                        }
                        for (int a = 1; a <= numAgents; ++a) {
                            out << freqBRAll[a] << ' ';
                        }
                        for (int a = 1; a <= numAgents; ++a) {
                            out << freqBROnPath[a] << ' ';
                        }
                        for (int a = 1; a <= numAgents; ++a) {
                            out << freqBROffPath[a] << ' ';
                        }

                        out << QGapTotSession[0] << ' ' << QGapOnPathSession[0] << ' ' << QGapNotOnPathSession[0] << ' '
                            << QGapNotBRAllStatesSession[0] << ' ' << QGapNotBRonPathSession[0] << ' '
                            << QGapNotEqAllStatesSession[0] << ' ' << QGapNotEqonPathSession[0] << ' ';

                        for (int a = 1; a <= numAgents; ++a) {
                            out << QGapTotSession[a] << ' ';
                        }
                        for (int a = 1; a <= numAgents; ++a) {
                            out << QGapOnPathSession[a] << ' ';
                        }
                        for (int a = 1; a <= numAgents; ++a) {
                            out << QGapNotOnPathSession[a] << ' ';
                        }
                        for (int a = 1; a <= numAgents; ++a) {
                            out << QGapNotBRAllStatesSession[a] << ' ';
                        }
                        for (int a = 1; a <= numAgents; ++a) {
                            out << QGapNotBRonPathSession[a] << ' ';
                        }
                        for (int a = 1; a <= numAgents; ++a) {
                            out << QGapNotEqAllStatesSession[a] << ' ';
                        }
                        for (int a = 1; a <= numAgents; ++a) {
                            out << QGapNotEqonPathSession[a] << ' ';
                        }

                        for (int a = 1; a <= numAgents; ++a) {
                            out << IndPrePrices[iStatePre][a] << ' ';
                        }
                        for (int a = 1; a <= numAgents; ++a) {
                            out << PreProfits[iStatePre][a] << ' ';
                        }

                        out << iAgent << ' ' << jAgent << ' ' << DeviationQ[jAgent] << ' ' << ShockLength << ' '
                            << SameCyclePrePost << ' ' << StaticBRPrices[1][iAgent] << ' ';

                        for (int t = 1; t <= numShockPeriodsPrint; ++t) {
                            out << ShockPrices[t][jAgent] << ' ';
                        }
                        for (int t = 1; t <= numShockPeriodsPrint; ++t) {
                            out << ShockProfits[t][jAgent] << ' ';
                        }
                        for (int t = 1; t <= numShockPeriodsPrint; ++t) {
                            out << StaticBRPrices[t][jAgent] << ' ';
                        }
                        for (int t = 1; t <= numShockPeriodsPrint; ++t) {
                            out << DynamicBRPrices[t][jAgent] << ' ';
                        }
                        for (int t = 1; t <= numShockPeriodsPrint; ++t) {
                            out << OptStratQ[t][jAgent] << ' ';
                        }
                        for (int t = 1; t <= numShockPeriodsPrint; ++t) {
                            out << DynamicBRQ[t][jAgent] << ' ';
                        }

                        out << PostLength << ' ';
                        for (int a = 1; a <= numAgents; ++a) {
                            out << AvgPostPrices[a] << ' ';
                        }
                        for (int a = 1; a <= numAgents; ++a) {
                            out << AvgPostProfits[a] << (a == numAgents ? '\n' : ' ');
                        }
                    }
                }
            }
        }

        std::cout << "Session = " << iSession << " completed\n";
    }
}

}  // namespace DetailedAnalysis
