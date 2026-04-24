#include "ConvergenceResults.hpp"

#include "QL_routines.hpp"
#include "generic_routines.hpp"
#include "globals.hpp"

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

namespace ConvergenceResults {

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

void read_nonempty_line(std::istream& in, std::string& line) {
    while (std::getline(in, line)) {
        if (!line.empty()) {
            return;
        }
    }
}

int first_repeat_index(const std::vector<int>& v, int upto, int value) {
    for (int i = 1; i <= upto; ++i) {
        if (v[i] == value) {
            return i;
        }
    }
    return 1;
}

}  // namespace

void ComputeConvResults(int iExperiment) {
    using namespace globals;

    std::cout << "Computing convergence results (average profits and frequency of prices)\n";

    auto Profits = runtime::make2<double>(numSessions, numAgents, 0.0);
    auto FreqStates = runtime::make2<double>(numSessions, numStates, 0.0);

    {
        std::ifstream in(FileNameInfoExperiment);
        std::string line;
        for (int iSession = 1; iSession <= numSessions; ++iSession) {
            if ((iSession % 100) == 0) {
                std::cout << "Read " << iSession << " strategies\n";
            }

            int rSession = iSession;
            read_nonempty_line(in, line);
            {
                std::istringstream iss(line);
                iss >> rSession;
            }

            read_nonempty_line(in, line);
            {
                std::istringstream iss(line);
                iss >> converged[rSession];
            }

            read_nonempty_line(in, line);
            {
                std::istringstream iss(line);
                iss >> timeToConvergence[rSession];
            }

            read_nonempty_line(in, line);
            {
                std::istringstream iss(line);
                for (int k = 1; k <= LengthStates; ++k) {
                    iss >> indexLastState[k][rSession];
                }
            }

            for (int iState = 1; iState <= numStates; ++iState) {
                read_nonempty_line(in, line);
                std::istringstream iss(line);
                for (int iAgent = 1; iAgent <= numAgents; ++iAgent) {
                    iss >> indexStrategies[(iAgent - 1) * numStates + iState][rSession];
                }
            }
        }
    }

    std::ofstream out(FileNameInfoExperiment, std::ios::trunc);

    for (int iSession = 1; iSession <= numSessions; ++iSession) {
        std::cout << "iSession = " << iSession << "\n";

        auto OptimalStrategyVec = runtime::make1<int>(lengthStrategies, 0);
        auto LastStateVec = runtime::make1<int>(LengthStates, 0);
        for (int k = 1; k <= lengthStrategies; ++k) {
            OptimalStrategyVec[k] = indexStrategies[k][iSession];
        }
        for (int k = 1; k <= LengthStates; ++k) {
            LastStateVec[k] = indexLastState[k][iSession];
        }

        auto OptimalStrategy = runtime::make2<int>(numStates, numAgents, 0);
        for (int iAgent = 1; iAgent <= numAgents; ++iAgent) {
            for (int iState = 1; iState <= numStates; ++iState) {
                OptimalStrategy[iState][iAgent] = OptimalStrategyVec[(iAgent - 1) * numStates + iState];
            }
        }

        auto LastObservedPrices = runtime::make2<int>(DepthState, numAgents, 0);
        if (DepthState0 == 0) {
            for (int iAgent = 1; iAgent <= numAgents; ++iAgent) {
                LastObservedPrices[1][iAgent] = OptimalStrategy[1][iAgent];
            }
        } else {
            LastObservedPrices = runtime::reshape_to_depth_agent(LastStateVec, DepthState, numAgents);
        }

        auto VisitedStates = runtime::make1<int>(numPeriods, 0);
        auto VisitedProfits = runtime::make2<double>(numPeriods, numAgents, 0.0);
        auto pHist = runtime::make2<int>(numPeriods, numAgents, 0);

        auto p = LastObservedPrices;
        auto pPrime = runtime::make1<int>(numAgents, 0);
        {
            const int s0 = QL_routines::computeStateNumber(p);
            for (int iAgent = 1; iAgent <= numAgents; ++iAgent) {
                pPrime[iAgent] = OptimalStrategy[s0][iAgent];
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
                pHist[iPeriod][a] = pPrime[a];
            }

            VisitedStates[iPeriod] = QL_routines::computeStateNumber(p);

            for (int iAgent = 1; iAgent <= numAgents; ++iAgent) {
                VisitedProfits[iPeriod][iAgent] = PI[QL_routines::computeActionNumber(pPrime)][iAgent];
            }

            bool repeated = false;
            if (iPeriod >= 2) {
                for (int t = 1; t <= iPeriod - 1; ++t) {
                    if (VisitedStates[t] == VisitedStates[iPeriod]) {
                        repeated = true;
                        break;
                    }
                }
            }
            if (repeated) {
                break;
            }

            for (int a = 1; a <= numAgents; ++a) {
                pPrime[a] = OptimalStrategy[VisitedStates[iPeriod]][a];
            }
        }

        const int cycleLength = iPeriod - first_repeat_index(VisitedStates, iPeriod - 1, VisitedStates[iPeriod]);

        for (int iAgent = 1; iAgent <= numAgents; ++iAgent) {
            double s = 0.0;
            for (int t = iPeriod - cycleLength + 1; t <= iPeriod; ++t) {
                s += VisitedProfits[t][iAgent];
            }
            Profits[iSession][iAgent] = s / static_cast<double>(cycleLength);
        }

        for (int t = iPeriod - cycleLength + 1; t <= iPeriod; ++t) {
            FreqStates[iSession][VisitedStates[t]] = 1.0 / static_cast<double>(cycleLength);
        }

        for (int i = 1; i <= cycleLength; ++i) {
            for (int iAgent = 1; iAgent <= numAgents; ++iAgent) {
                pHist[i][iAgent] = pHist[iPeriod - cycleLength + i][iAgent];
                VisitedProfits[i][iAgent] = VisitedProfits[iPeriod - cycleLength + i][iAgent];
            }
            VisitedStates[i] = VisitedStates[iPeriod - cycleLength + i];
        }

        for (int i = cycleLength + 1; i <= numPeriods; ++i) {
            VisitedStates[i] = 0;
            for (int iAgent = 1; iAgent <= numAgents; ++iAgent) {
                pHist[i][iAgent] = 0;
                VisitedProfits[i][iAgent] = 0.0;
            }
        }

        out << ' ';
        write_int_field(out, iSession, 8);
        out << '\n';

        out << ' ';
        write_int_field(out, converged[iSession], 1);
        out << '\n';

        out << ' ';
        write_real_field(out, timeToConvergence[iSession], 9, 2);
        out << '\n';

        out << ' ';
        write_int_field(out, cycleLength, 8);
        out << '\n';

        for (int i = 1; i <= cycleLength; ++i) {
            out << ' ';
            write_int_field(out, VisitedStates[i], LengthFormatStatesPrint);
        }
        out << '\n';

        for (int iAgent = 1; iAgent <= numAgents; ++iAgent) {
            for (int i = 1; i <= cycleLength; ++i) {
                out << ' ';
                write_int_field(out, pHist[i][iAgent], LengthFormatActionPrint);
            }
        }
        out << '\n';

        for (int iAgent = 1; iAgent <= numAgents; ++iAgent) {
            for (int i = 1; i <= cycleLength; ++i) {
                out << ' ';
                write_real_field(out, VisitedProfits[i][iAgent], 8, 5);
            }
        }
        out << '\n';

        for (int iState = 1; iState <= numStates; ++iState) {
            for (int iAgent = 1; iAgent <= numAgents; ++iAgent) {
                out << ' ';
                write_int_field(out, OptimalStrategy[iState][iAgent], LengthFormatActionPrint);
            }
            out << '\n';
        }
        out << '\n';
    }

    auto meanProfit = runtime::make1<double>(numAgents, 0.0);
    auto seProfit = runtime::make1<double>(numAgents, 0.0);
    auto meanProfitGain = runtime::make1<double>(numAgents, 0.0);
    auto seProfitGain = runtime::make1<double>(numAgents, 0.0);

    for (int iAgent = 1; iAgent <= numAgents; ++iAgent) {
        double m = 0.0;
        double q = 0.0;
        for (int iSession = 1; iSession <= numSessions; ++iSession) {
            m += Profits[iSession][iAgent];
            q += Profits[iSession][iAgent] * Profits[iSession][iAgent];
        }
        m /= static_cast<double>(numSessions);
        q /= static_cast<double>(numSessions);

        meanProfit[iAgent] = m;
        seProfit[iAgent] = std::sqrt(std::abs(q - m * m));

        meanProfitGain[iAgent] = (meanProfit[iAgent] - NashProfits[iAgent]) / (CoopProfits[iAgent] - NashProfits[iAgent]);
        seProfitGain[iAgent] = seProfit[iAgent] / (CoopProfits[iAgent] - NashProfits[iAgent]);
    }

    auto AvgProfits = runtime::make1<double>(numSessions, 0.0);
    for (int iSession = 1; iSession <= numSessions; ++iSession) {
        double s = 0.0;
        for (int iAgent = 1; iAgent <= numAgents; ++iAgent) {
            s += Profits[iSession][iAgent];
        }
        AvgProfits[iSession] = s / static_cast<double>(numAgents);
    }

    double meanAvgProfit = 0.0;
    double meanAvgProfitQ = 0.0;
    for (int iSession = 1; iSession <= numSessions; ++iSession) {
        meanAvgProfit += AvgProfits[iSession];
        meanAvgProfitQ += AvgProfits[iSession] * AvgProfits[iSession];
    }
    meanAvgProfit /= static_cast<double>(numSessions);
    meanAvgProfitQ /= static_cast<double>(numSessions);
    const double seAvgProfit = std::sqrt(std::abs(meanAvgProfitQ - meanAvgProfit * meanAvgProfit));

    meanNashProfit = 0.0;
    meanCoopProfit = 0.0;
    for (int iAgent = 1; iAgent <= numAgents; ++iAgent) {
        meanNashProfit += NashProfits[iAgent];
        meanCoopProfit += CoopProfits[iAgent];
    }
    meanNashProfit /= static_cast<double>(numAgents);
    meanCoopProfit /= static_cast<double>(numAgents);

    const double meanAvgProfitGain = (meanAvgProfit - meanNashProfit) / (meanCoopProfit - meanNashProfit);
    const double seAvgProfitGain = seAvgProfit / (meanCoopProfit - meanNashProfit);

    auto meanFreqStates = runtime::make1<double>(numStates, 0.0);
    for (int i = 1; i <= numStates; ++i) {
        double s = 0.0;
        for (int ss = 1; ss <= numSessions; ++ss) {
            s += FreqStates[ss][i];
        }
        meanFreqStates[i] = s / static_cast<double>(numSessions);
    }

    std::fstream& conv = runtime::io::unit(100022);
    if (iExperiment == 1) {
        conv << "Experiment ";
        for (int i = 1; i <= numAgents; ++i) {
            conv << "    alpha" << i << ' ';
        }
        for (int i = 1; i <= numExplorationParameters; ++i) {
            conv << "     beta" << i << ' ';
        }
        conv << "     delta ";
        for (int i = 1; i <= numAgents; ++i) {
            conv << "typeQini" << i << ' ';
            for (int j = 1; j <= numAgents; ++j) {
                conv << "par" << j << "Qini" << i << ' ';
            }
        }
        for (int i = 1; i <= numDemandParameters; ++i) {
            conv << "  DemPar" << std::setw(2) << std::setfill('0') << i << std::setfill(' ') << ' ';
        }
        for (int i = 1; i <= numAgents; ++i) {
            conv << "NashPrice" << i << ' ';
        }
        for (int i = 1; i <= numAgents; ++i) {
            conv << "CoopPrice" << i << ' ';
        }
        for (int i = 1; i <= numAgents; ++i) {
            conv << "NashProft" << i << ' ';
        }
        for (int i = 1; i <= numAgents; ++i) {
            conv << "CoopProft" << i << ' ';
        }
        for (int i = 1; i <= numAgents; ++i) {
            conv << "NashMktSh" << i << ' ';
        }
        for (int i = 1; i <= numAgents; ++i) {
            conv << "CoopMktSh" << i << ' ';
        }
        for (int i = 1; i <= numAgents; ++i) {
            for (int j = 1; j <= numPrices; ++j) {
                conv << "Ag" << i << "Price" << std::setw(2) << std::setfill('0') << j << std::setfill(' ') << ' ';
            }
        }
        for (int i = 1; i <= numAgents; ++i) {
            conv << "  avgProf" << i << " " << "   seProf" << i << ' ';
        }
        conv << "   avgProf     seProf ";
        for (int i = 1; i <= numAgents; ++i) {
            conv << "avgPrGain" << i << " " << " sePrGain" << i << ' ';
        }
        conv << " avgPrGain   sePrGain ";
        const int stateWidth = std::max(10, 3 + LengthFormatStatesPrint);
        for (int j = 1; j <= numStates; ++j) {
            conv << std::setw(stateWidth) << std::setfill(' ') << labelStates[j] << ' ';
        }
        conv << '\n';
    }

    write_int_field(conv, codExperiment, 10);
    conv << ' ';
    for (int i = 1; i <= numAgents; ++i) {
        write_real_field(conv, alpha[i], 10, 5);
        conv << ' ';
    }
    for (int i = 1; i <= numExplorationParameters; ++i) {
        write_real_field(conv, MExpl[i], 10, 5);
        conv << ' ';
    }
    write_real_field(conv, delta, 10, 5);
    conv << ' ';

    for (int i = 1; i <= numAgents; ++i) {
        write_char_field(conv, typeQInitialization[i], 9);
        conv << ' ';
        for (int j = 1; j <= numAgents; ++j) {
            write_real_field(conv, parQInitialization[i][j], 9, 2);
            conv << ' ';
        }
    }

    for (int i = 1; i <= numDemandParameters; ++i) {
        write_real_field(conv, DemandParameters[i], 10, 5);
        conv << ' ';
    }

    for (int i = 1; i <= numAgents; ++i) {
        write_real_field(conv, NashPrices[i], 10, 5);
        conv << ' ';
    }
    for (int i = 1; i <= numAgents; ++i) {
        write_real_field(conv, CoopPrices[i], 10, 5);
        conv << ' ';
    }
    for (int i = 1; i <= numAgents; ++i) {
        write_real_field(conv, NashProfits[i], 10, 5);
        conv << ' ';
    }
    for (int i = 1; i <= numAgents; ++i) {
        write_real_field(conv, CoopProfits[i], 10, 5);
        conv << ' ';
    }
    for (int i = 1; i <= numAgents; ++i) {
        write_real_field(conv, NashMarketShares[i], 10, 5);
        conv << ' ';
    }
    for (int i = 1; i <= numAgents; ++i) {
        write_real_field(conv, CoopMarketShares[i], 10, 5);
        conv << ' ';
    }
    for (int i = 1; i <= numAgents; ++i) {
        for (int j = 1; j <= numPrices; ++j) {
            write_real_field(conv, PricesGrids[j][i], 10, 5);
            conv << ' ';
        }
    }

    for (int i = 1; i <= numAgents; ++i) {
        write_real_field(conv, meanProfit[i], 10, 5);
        conv << ' ';
        write_real_field(conv, seProfit[i], 10, 5);
        conv << ' ';
    }
    write_real_field(conv, meanAvgProfit, 10, 5);
    conv << ' ';
    write_real_field(conv, seAvgProfit, 10, 5);
    conv << ' ';

    for (int i = 1; i <= numAgents; ++i) {
        write_real_field(conv, meanProfitGain[i], 10, 5);
        conv << ' ';
        write_real_field(conv, seProfitGain[i], 10, 5);
        conv << ' ';
    }
    write_real_field(conv, meanAvgProfitGain, 10, 5);
    conv << ' ';
    write_real_field(conv, seAvgProfitGain, 10, 5);
    conv << ' ';

    const int stateWidth = std::max(10, 3 + LengthFormatStatesPrint);
    for (int i = 1; i <= numStates; ++i) {
        write_real_field(conv, meanFreqStates[i], stateWidth, 6);
        if (i < numStates) {
            conv << ' ';
        }
    }
    conv << '\n';
}

}  // namespace ConvergenceResults
