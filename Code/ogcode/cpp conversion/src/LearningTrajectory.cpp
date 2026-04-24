#include "LearningTrajectory.hpp"

#include "ImpulseResponse.hpp"
#include "LearningSimulation.hpp"
#include "QL_routines.hpp"
#include "generic_routines.hpp"

#include <fstream>
#include <iostream>

namespace LearningTrajectory {

void computeLearningTrajectory(
    int iExperiment,
    int codExperiment,
    const std::vector<double>& alpha,
    const std::vector<double>& ExplorationParameters,
    double delta) {
    using namespace globals;

    (void)iExperiment;
    (void)codExperiment;

    QL_routines::ReadInfoExperiment();

    auto PGmat = runtime::make2<double>(ParamsLearningTrajectory[1], numSessions, 0.0);
    auto ICmat = runtime::make2<double>(ParamsLearningTrajectory[1], numSessions, 0.0);
    auto IRmat = runtime::make2<double>(ParamsLearningTrajectory[1], numSessions, 0.0);

    int idumIP = -1;
    int idum2IP = 123456789;
    std::vector<int> ivIP(33, 0);
    int iyIP = 0;
    auto uIniPrice = runtime::make3<double>(DepthState, numAgents, numSessions, 0.0);
    QL_routines::generate_uIniPrice(uIniPrice, idumIP, ivIP, iyIP, idum2IP);

    for (int iSession = 1; iSession <= numSessions; ++iSession) {
        std::cout << "Session = " << iSession << " started\n";

        const int CycleLengthSession = CycleLength[iSession];
        auto CycleStatesSession = runtime::make1<int>(numPeriods, 0);
        for (int k = 1; k <= CycleLengthSession; ++k) {
            CycleStatesSession[k] = CycleStates[k][iSession];
        }

        int idum = -iSession;
        int idum2 = 123456789;
        std::vector<int> iv(33, 0);
        int iy = 0;

        int idumQ = -iSession;
        int idum2Q = 123456789;
        std::vector<int> ivQ(33, 0);
        int iyQ = 0;

        auto Q = runtime::make3<double>(numStates, numPrices, numAgents, 0.0);
        auto maxValQLocal = runtime::make2<double>(numStates, numAgents, 0.0);
        auto strategyPrime = runtime::make2<int>(numStates, numAgents, 0);

        QL_routines::initQMatrices(iSession, idumQ, ivQ, iyQ, idum2Q, PI, delta, Q, maxValQLocal, strategyPrime);
        auto strategy = strategyPrime;

        auto p = runtime::make2<int>(DepthState, numAgents, 0);
        int statePrime = 0;
        int actionPrime = 0;

        auto uIniSlice = runtime::make2<double>(DepthState, numAgents, 0.0);
        for (int d = 1; d <= DepthState; ++d) {
            for (int a = 1; a <= numAgents; ++a) {
                uIniSlice[d][a] = uIniPrice[d][a][iSession];
            }
        }
        QL_routines::initState(uIniSlice, p, statePrime, actionPrime);
        int state = statePrime;

        auto eps = runtime::make1<double>(numAgents, 0.0);
        if (typeExplorationMechanism == 1) {
            for (int iAgent = 1; iAgent <= numAgents; ++iAgent) {
                eps[iAgent] = 1.0;
            }
        }
        if (typeExplorationMechanism == 2) {
            for (int iAgent = 1; iAgent <= numAgents; ++iAgent) {
                eps[iAgent] = 1.0e3;
            }
        }

        auto uExploration = runtime::make2<double>(2, numAgents, 0.0);
        auto pPrime = runtime::make1<int>(numAgents, 0);
        auto profitgain = runtime::make1<double>(numAgents, 0.0);

        double PGsum = 0.0;
        double ICsum = 0.0;
        double IRsum = 0.0;

        const int totalIters = ParamsLearningTrajectory[1] * ParamsLearningTrajectory[2];
        for (int iIters = 1; iIters <= totalIters; ++iIters) {
            QL_routines::generateUExploration(uExploration, idum, iv, iy, idum2);

            LearningSimulation::computePPrime(
                ExplorationParameters,
                uExploration,
                strategyPrime,
                state,
                iIters,
                pPrime,
                Q,
                eps);

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

            statePrime = QL_routines::computeStateNumber(p);
            actionPrime = QL_routines::computeActionNumber(pPrime);

            for (int iAgent = 1; iAgent <= numAgents; ++iAgent) {
                profitgain[iAgent] = PG[actionPrime][iAgent];
            }

            double pgAvg = 0.0;
            for (int iAgent = 1; iAgent <= numAgents; ++iAgent) {
                pgAvg += profitgain[iAgent];
            }
            pgAvg /= static_cast<double>(numAgents);
            PGsum += pgAvg;

            if ((iIters % ParamsLearningTrajectory[2]) == 0) {
                const int idx = iIters / ParamsLearningTrajectory[2];

                PGmat[idx][iSession] += PGsum / static_cast<double>(ParamsLearningTrajectory[2]);
                PGsum = 0.0;

                for (int iCycle = 1; iCycle <= CycleLengthSession; ++iCycle) {
                    const int ss1 = CycleStatesSession[iCycle];
                    for (int iAgent = 1; iAgent <= numAgents; ++iAgent) {
                        auto QRowValues = runtime::make1<double>(numPrices, 0.0);
                        for (int iPrice = 1; iPrice <= numPrices; ++iPrice) {
                            auto NUVisitedStates = runtime::make1<int>(numPeriods, 0);
                            int NUPreCycleLength = 0;
                            int NUCycleLength = 0;
                            QL_routines::computeQCell(
                                strategy,
                                ss1,
                                iPrice,
                                iAgent,
                                delta,
                                QRowValues[iPrice],
                                NUVisitedStates,
                                NUPreCycleLength,
                                NUCycleLength);
                        }

                        double maxQ = QRowValues[1];
                        for (int iPrice = 2; iPrice <= numPrices; ++iPrice) {
                            maxQ = std::max(maxQ, QRowValues[iPrice]);
                        }

                        if (generic_routines::AreEqualReals(maxQ, QRowValues[strategy[ss1][iAgent]])) {
                            ICsum += 1.0 / static_cast<double>(numAgents * CycleLengthSession);
                        }
                    }
                }

                ICmat[idx][iSession] = ICsum;
                ICsum = 0.0;

                for (int iCycle = 1; iCycle <= CycleLengthSession; ++iCycle) {
                    for (int iAgent = 1; iAgent <= numAgents; ++iAgent) {
                        const int ss0 = CycleStatesSession[iCycle];

                        auto pLT1 = runtime::reshape_to_depth_agent(
                            generic_routines::convertNumberBase(ss0 - 1, numPrices, LengthStates),
                            DepthState,
                            numAgents);

                        auto pPrimeLT1 = runtime::make1<int>(numAgents, 0);
                        for (int a = 1; a <= numAgents; ++a) {
                            pPrimeLT1[a] = strategy[ss0][a];
                        }

                        if (DepthState > 1) {
                            for (int d = DepthState; d >= 2; --d) {
                                for (int a = 1; a <= numAgents; ++a) {
                                    pLT1[d][a] = pLT1[d - 1][a];
                                }
                            }
                        }
                        for (int a = 1; a <= numAgents; ++a) {
                            pLT1[1][a] = pPrimeLT1[a];
                        }

                        double NUPIStaticBR = 0.0;
                        ImpulseResponse::ComputeStaticBestResponse(strategy, ss0, iAgent, pLT1[1][iAgent], NUPIStaticBR);

                        const int ss1 = QL_routines::computeStateNumber(pLT1);

                        auto pPrimeLT2 = runtime::make1<int>(numAgents, 0);
                        for (int a = 1; a <= numAgents; ++a) {
                            pPrimeLT2[a] = strategy[ss1][a];
                        }

                        double avgPRatio = 0.0;
                        for (int jAgent = 1; jAgent <= numAgents; ++jAgent) {
                            if (jAgent == iAgent) {
                                continue;
                            }
                            avgPRatio +=
                                PricesGrids[pPrimeLT2[jAgent]][jAgent] / PricesGrids[pPrimeLT1[jAgent]][jAgent];
                        }
                        avgPRatio /= static_cast<double>(numAgents - 1);

                        IRsum += avgPRatio / static_cast<double>(numAgents * CycleLengthSession);
                    }
                }

                IRmat[idx][iSession] = IRsum;
                IRsum = 0.0;
            }

            for (int iAgent = 1; iAgent <= numAgents; ++iAgent) {
                const double oldq = Q[state][pPrime[iAgent]][iAgent];
                const double newq =
                    oldq + alpha[iAgent] * (PI[actionPrime][iAgent] + delta * maxValQLocal[statePrime][iAgent] - oldq);

                Q[state][pPrime[iAgent]][iAgent] = newq;

                if (newq > maxValQLocal[state][iAgent]) {
                    maxValQLocal[state][iAgent] = newq;
                    if (strategyPrime[state][iAgent] != pPrime[iAgent]) {
                        strategyPrime[state][iAgent] = pPrime[iAgent];
                    }
                }

                if ((newq < maxValQLocal[state][iAgent]) && (strategyPrime[state][iAgent] == pPrime[iAgent])) {
                    auto row = runtime::make1<double>(numPrices, 0.0);
                    for (int iPrice = 1; iPrice <= numPrices; ++iPrice) {
                        row[iPrice] = Q[state][iPrice][iAgent];
                    }
                    generic_routines::MaxLocBreakTies(
                        numPrices,
                        row,
                        idumQ,
                        ivQ,
                        iyQ,
                        idum2Q,
                        maxValQLocal[state][iAgent],
                        strategyPrime[state][iAgent]);
                }
            }

            for (int iAgent = 1; iAgent <= numAgents; ++iAgent) {
                strategy[state][iAgent] = strategyPrime[state][iAgent];
            }
            state = statePrime;
        }
    }

    const auto PGss = generic_routines::ComputeRowSummaryStatistics(ParamsLearningTrajectory[1], numSessions, PGmat);
    const auto ICss = generic_routines::ComputeRowSummaryStatistics(ParamsLearningTrajectory[1], numSessions, ICmat);
    const auto IRss = generic_routines::ComputeRowSummaryStatistics(ParamsLearningTrajectory[1], numSessions, IRmat);

    const std::string LTrajectoryFileName = "LTrajectories_" + ExperimentNumber;
    std::ofstream out(LTrajectoryFileName, std::ios::trunc);

    out << "iter "
        << "AvgPG SdPG MinPG q0025PG q025PG q05PG q075PG q0975PG MaxPG "
        << "AvgIC SdIC MinIC q0025IC q025IC q05IC q075IC q0975IC MaxIC "
        << "AvgIR SdIR MinIR q0025IR q025IR q05IR q075IR q0975IR MaxIR\n";

    for (int i = 1; i <= ParamsLearningTrajectory[1]; ++i) {
        out << i * ParamsLearningTrajectory[2] << ' ';
        for (int k = 1; k <= 9; ++k) {
            out << PGss[i][k] << ' ';
        }
        for (int k = 1; k <= 9; ++k) {
            out << ICss[i][k] << ' ';
        }
        for (int k = 1; k <= 9; ++k) {
            out << IRss[i][k] << (k == 9 ? '\n' : ' ');
        }
    }
}

}  // namespace LearningTrajectory
