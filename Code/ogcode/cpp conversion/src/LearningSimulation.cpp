#include "LearningSimulation.hpp"

#include "QL_routines.hpp"
#include "generic_routines.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

namespace {

void write_int_field(std::ostream& os, int value, int width) {
    os << std::setw(width) << std::setfill(' ') << value;
}

void write_char_field(std::ostream& os, char value, int width) {
    os << std::setw(width) << std::setfill(' ') << value;
}

void write_real_field(std::ostream& os, double value, int width, int precision) {
    os << std::setw(width) << std::setfill(' ') << std::fixed << std::setprecision(precision) << value;
}

}  // namespace

namespace LearningSimulation {

void computePPrime(
    const std::vector<double>& ExplorationParameters,
    const std::vector<std::vector<double>>& uExploration,
    const std::vector<std::vector<int>>& strategyPrime,
    int state,
    int iIters,
    std::vector<int>& pPrime,
    const std::vector<std::vector<std::vector<double>>>& Q,
    std::vector<double>& eps) {
    using namespace globals;
    (void)iIters;

    if (typeExplorationMechanism == 1) {
        for (int iAgent = 1; iAgent <= numAgents; ++iAgent) {
            if (MExpl[iAgent] < 0.0) {
                pPrime[iAgent] = strategyPrime[state][iAgent];
            } else {
                const double u1 = uExploration[1][iAgent];
                const double u2 = uExploration[2][iAgent];
                if (u1 <= eps[iAgent]) {
                    pPrime[iAgent] = 1 + static_cast<int>(static_cast<double>(numPrices) * u2);
                } else {
                    pPrime[iAgent] = strategyPrime[state][iAgent];
                }
                eps[iAgent] = eps[iAgent] * ExplorationParameters[iAgent];
            }
        }
    }

    if (typeExplorationMechanism == 2) {
        for (int iAgent = 1; iAgent <= numAgents; ++iAgent) {
            double maxQ = Q[state][1][iAgent];
            for (int iPrice = 2; iPrice <= numPrices; ++iPrice) {
                maxQ = std::max(maxQ, Q[state][iPrice][iAgent]);
            }

            std::vector<double> probs = runtime::make1<double>(numPrices, 0.0);
            double sumProbs = 0.0;
            for (int iPrice = 1; iPrice <= numPrices; ++iPrice) {
                probs[iPrice] = std::exp((Q[state][iPrice][iAgent] - maxQ) / eps[iAgent]);
                sumProbs += probs[iPrice];
            }

            const double u1 = uExploration[1][iAgent] * sumProbs;
            double cdf = 0.0;
            for (int iPrice = 1; iPrice <= numPrices; ++iPrice) {
                cdf += probs[iPrice];
                if (u1 <= cdf) {
                    pPrime[iAgent] = iPrice;
                    break;
                }
            }

            eps[iAgent] = eps[iAgent] * ExplorationParameters[iAgent];
        }
    }
}

void computeExperiment(
    int iExperiment,
    int codExperiment,
    const std::vector<double>& alpha,
    const std::vector<double>& ExplorationParameters,
    double delta) {
    using namespace globals;

    for (int i = 1; i <= numSessions; ++i) {
        converged[i] = 0;
        timeToConvergence[i] = 0.0;
    }
    for (int i = 1; i <= lengthStrategies; ++i) {
        for (int j = 1; j <= numSessions; ++j) {
            indexStrategies[i][j] = 0;
        }
    }
    for (int i = 1; i <= LengthStates; ++i) {
        for (int j = 1; j <= numSessions; ++j) {
            indexLastState[i][j] = 0;
        }
    }

    const std::string codExperimentChar = [] (int value, int width) {
        std::ostringstream oss;
        oss << std::setw(width) << std::setfill('0') << value;
        return oss.str();
    }(codExperiment, LengthFormatTotExperimentsPrint);

    int idumIP = -1;
    int idum2IP = 123456789;
    std::vector<int> ivIP(33, 0);
    int iyIP = 0;
    auto uIniPrice = runtime::make3<double>(DepthState, numAgents, numSessions, 0.0);
    QL_routines::generate_uIniPrice(uIniPrice, idumIP, ivIP, iyIP, idum2IP);

    for (int iSession = 1; iSession <= numSessions; ++iSession) {
        std::cout << "Session = " << iSession << " started\n";

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

        int iIters = 0;
        int iItersFix = 0;
        int iItersInStrategy = 0;
        int convergedSession = -1;

        auto strategyFix = runtime::make2<int>(numStates, numAgents, 0);
        int stateFix = state;
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

        while (true) {
            ++iIters;

            QL_routines::generateUExploration(uExploration, idum, iv, iy, idum2);

            computePPrime(
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
                    std::vector<double> row = runtime::make1<double>(numPrices, 0.0);
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

            bool equalOnState = true;
            for (int a = 1; a <= numAgents; ++a) {
                if (strategyPrime[state][a] != strategy[state][a]) {
                    equalOnState = false;
                    break;
                }
            }
            if (equalOnState) {
                ++iItersInStrategy;
            } else {
                iItersInStrategy = 1;
            }

            if (convergedSession == -1) {
                if (iIters > maxIters) {
                    convergedSession = 0;
                    strategyFix = strategy;
                    stateFix = state;
                    iItersFix = iIters;
                }

                if (iItersInStrategy == itersInPerfMeasPeriod) {
                    convergedSession = 1;
                    strategyFix = strategy;
                    stateFix = state;
                    iItersFix = iIters;
                }
            }

            if (convergedSession != -1) {
                break;
            }

            for (int a = 1; a <= numAgents; ++a) {
                strategy[state][a] = strategyPrime[state][a];
            }
            state = statePrime;
        }

        if (printQ == 1) {
            std::ostringstream sss;
            sss << std::setw(LengthFormatNumSessionsPrint) << std::setfill('0') << iSession;
            const std::string QFileName = "Q_" + codExperimentChar + "_" + sss.str() + ".txt";

            std::ofstream qout(QFileName);
            for (int iAgent = 1; iAgent <= numAgents; ++iAgent) {
                for (int iState = 1; iState <= numStates; ++iState) {
                    for (int iPrice = 1; iPrice <= numPrices; ++iPrice) {
                        qout << Q[iState][iPrice][iAgent] << (iPrice == numPrices ? '\n' : ' ');
                    }
                }
            }
        }

        converged[iSession] = convergedSession;
        timeToConvergence[iSession] =
            static_cast<double>(iItersFix - itersInPerfMeasPeriod) / static_cast<double>(itersPerEpisode);

        const auto lastState = generic_routines::convertNumberBase(stateFix - 1, numPrices, LengthStates);
        for (int k = 1; k <= LengthStates; ++k) {
            indexLastState[k][iSession] = lastState[k];
        }

        const auto strategyNumber = QL_routines::computeStrategyNumber(strategyFix);
        for (int k = 1; k <= lengthStrategies; ++k) {
            indexStrategies[k][iSession] = strategyNumber[k];
        }

        if (convergedSession == 1) {
            std::cout << "Session = " << iSession << " converged\n";
        }
        if (convergedSession == 0) {
            std::cout << "Session = " << iSession << " did not converge\n";
        }
    }

    {
        std::ofstream out(FileNameInfoExperiment, std::ios::trunc);
        for (int iSession = 1; iSession <= numSessions; ++iSession) {
            out << iSession << '\n';
            out << converged[iSession] << '\n';
            out << timeToConvergence[iSession] << '\n';

            for (int i = 1; i <= LengthStates; ++i) {
                out << indexLastState[i][iSession] << (i == LengthStates ? '\n' : ' ');
            }

            for (int iState = 1; iState <= numStates; ++iState) {
                for (int iAgent = 1; iAgent <= numAgents; ++iAgent) {
                    out << indexStrategies[(iAgent - 1) * numStates + iState][iSession]
                        << (iAgent == numAgents ? '\n' : ' ');
                }
            }
        }
    }

    int numSessionsConverged = 0;
    for (int i = 1; i <= numSessions; ++i) {
        numSessionsConverged += converged[i];
    }

    meanNashProfit = 0.0;
    meanCoopProfit = 0.0;
    for (int i = 1; i <= numAgents; ++i) {
        meanNashProfit += NashProfits[i];
        meanCoopProfit += CoopProfits[i];
    }
    meanNashProfit /= static_cast<double>(numAgents);
    meanCoopProfit /= static_cast<double>(numAgents);

    double meanTimeToConvergence = 0.0;
    double seTimeToConvergence = 0.0;
    double medianTimeToConvergence = 0.0;

    if (numSessionsConverged > 0) {
        for (int i = 1; i <= numSessions; ++i) {
            if (converged[i] == 1) {
                meanTimeToConvergence += timeToConvergence[i];
            }
        }
        meanTimeToConvergence /= static_cast<double>(numSessionsConverged);

        double sq = 0.0;
        for (int i = 1; i <= numSessions; ++i) {
            if (converged[i] == 1) {
                sq += timeToConvergence[i] * timeToConvergence[i];
            }
        }
        seTimeToConvergence =
            std::sqrt(std::max(0.0, sq / static_cast<double>(numSessionsConverged) - meanTimeToConvergence * meanTimeToConvergence));
    }

    std::vector<double> ttc;
    ttc.reserve(numSessions);
    for (int i = 1; i <= numSessions; ++i) {
        ttc.push_back(timeToConvergence[i]);
    }
    std::sort(ttc.begin(), ttc.end());
    int med_idx = static_cast<int>(std::floor(0.5 * static_cast<double>(numSessions) + 0.5));
    med_idx = std::max(1, std::min(numSessions, med_idx));
    medianTimeToConvergence = ttc[med_idx - 1];

    std::fstream& res = runtime::io::unit(10002);
    if (iExperiment == 1) {
        res << "Experiment ";
        for (int i = 1; i <= numAgents; ++i) {
            res << "    alpha" << i << ' ';
        }
        for (int i = 1; i <= numExplorationParameters; ++i) {
            res << "     beta" << i << ' ';
        }
        res << "     delta ";
        for (int i = 1; i <= numAgents; ++i) {
            res << "typeQini" << i << ' ';
            for (int j = 1; j <= numAgents; ++j) {
                res << "par" << j << "Qini" << i << ' ';
            }
        }
        for (int i = 1; i <= numDemandParameters; ++i) {
            res << "  DemPar" << std::setw(2) << std::setfill('0') << i << std::setfill(' ') << ' ';
        }
        for (int i = 1; i <= numAgents; ++i) {
            res << "NashPrice" << i << ' ';
        }
        for (int i = 1; i <= numAgents; ++i) {
            res << "CoopPrice" << i << ' ';
        }
        for (int i = 1; i <= numAgents; ++i) {
            res << "NashProft" << i << ' ';
        }
        for (int i = 1; i <= numAgents; ++i) {
            res << "CoopProft" << i << ' ';
        }
        for (int i = 1; i <= numAgents; ++i) {
            res << "NashMktSh" << i << ' ';
        }
        for (int i = 1; i <= numAgents; ++i) {
            res << "CoopMktSh" << i << ' ';
        }
        for (int i = 1; i <= numAgents; ++i) {
            for (int j = 1; j <= numPrices; ++j) {
                res << "Ag" << i << "Price" << std::setw(2) << std::setfill('0') << j << std::setfill(' ') << ' ';
            }
        }
        res << "   numConv     avgTTC      seTTC     medTTC \n";
    }

    write_int_field(res, codExperiment, 10);
    res << ' ';
    for (int i = 1; i <= numAgents; ++i) {
        write_real_field(res, alpha[i], 10, 5);
        res << ' ';
    }
    for (int i = 1; i <= numExplorationParameters; ++i) {
        write_real_field(res, MExpl[i], 10, 5);
        res << ' ';
    }
    write_real_field(res, delta, 10, 5);
    res << ' ';

    for (int i = 1; i <= numAgents; ++i) {
        write_char_field(res, typeQInitialization[i], 9);
        res << ' ';
        for (int j = 1; j <= numAgents; ++j) {
            write_real_field(res, parQInitialization[i][j], 9, 2);
            res << ' ';
        }
    }

    for (int i = 1; i <= numDemandParameters; ++i) {
        write_real_field(res, DemandParameters[i], 10, 5);
        res << ' ';
    }

    for (int i = 1; i <= numAgents; ++i) {
        write_real_field(res, NashPrices[i], 10, 5);
        res << ' ';
    }
    for (int i = 1; i <= numAgents; ++i) {
        write_real_field(res, CoopPrices[i], 10, 5);
        res << ' ';
    }
    for (int i = 1; i <= numAgents; ++i) {
        write_real_field(res, NashProfits[i], 10, 5);
        res << ' ';
    }
    for (int i = 1; i <= numAgents; ++i) {
        write_real_field(res, CoopProfits[i], 10, 5);
        res << ' ';
    }
    for (int i = 1; i <= numAgents; ++i) {
        write_real_field(res, NashMarketShares[i], 10, 5);
        res << ' ';
    }
    for (int i = 1; i <= numAgents; ++i) {
        write_real_field(res, CoopMarketShares[i], 10, 5);
        res << ' ';
    }
    for (int i = 1; i <= numAgents; ++i) {
        for (int j = 1; j <= numPrices; ++j) {
            write_real_field(res, PricesGrids[j][i], 10, 7);
            res << ' ';
        }
    }

    write_int_field(res, numSessionsConverged, 10);
    res << ' ';
    write_real_field(res, meanTimeToConvergence, 10, 2);
    res << ' ';
    write_real_field(res, seTimeToConvergence, 10, 2);
    res << ' ';
    write_real_field(res, medianTimeToConvergence, 10, 2);
    res << '\n';
}

}  // namespace LearningSimulation
