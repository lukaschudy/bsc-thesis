#include "QL_routines.hpp"

#include "generic_routines.hpp"

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace QL_routines {

namespace {

int first_repeat_index(const std::vector<int>& v, int upto, int value) {
    for (int i = 1; i <= upto; ++i) {
        if (v[i] == value) {
            return i;
        }
    }
    return 1;
}

std::string trim_copy(const std::string& s) {
    std::size_t b = 0;
    while (b < s.size() && std::isspace(static_cast<unsigned char>(s[b])) != 0) {
        ++b;
    }
    std::size_t e = s.size();
    while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1])) != 0) {
        --e;
    }
    return s.substr(b, e - b);
}

std::string format_int_fortran_i0w(int value, int width) {
    std::ostringstream oss;
    oss << std::setw(width) << std::setfill('0') << value;
    return oss.str();
}

void read_nonempty_line(std::istream& in, std::string& line) {
    while (std::getline(in, line)) {
        if (!line.empty()) {
            return;
        }
    }
    throw std::runtime_error("Unexpected EOF while reading info file");
}

}  // namespace

void initQMatrices(
    int iSession,
    int& idumQ,
    std::vector<int>& ivQ,
    int& iyQ,
    int& idum2Q,
    const std::vector<std::vector<double>>& PI,
    double delta,
    std::vector<std::vector<std::vector<double>>>& Q,
    std::vector<std::vector<double>>& maxValQ,
    std::vector<std::vector<int>>& maxLocQ) {
    using namespace globals;

    std::vector<std::vector<int>> Strategy = runtime::make2<int>(numStates, numAgents, 0);
    std::vector<int> VisitedStates = runtime::make1<int>(numPeriods, 0);
    int PreCycleLength = 0;
    int CycleLengthLocal = 0;

    for (int iAgent = 1; iAgent <= numAgents; ++iAgent) {
        const char initType = typeQInitialization[iAgent];

        if (initType == 'F') {
            for (int jAgent = 1; jAgent <= numAgents; ++jAgent) {
                const int v = static_cast<int>(std::llround(parQInitialization[iAgent][jAgent]));
                for (int iState = 1; iState <= numStates; ++iState) {
                    Strategy[iState][jAgent] = v;
                }
            }

            for (int iState = 1; iState <= numStates; ++iState) {
                for (int iPrice = 1; iPrice <= numPrices; ++iPrice) {
                    double qcell = 0.0;
                    computeQCell(
                        Strategy,
                        iState,
                        iPrice,
                        iAgent,
                        delta,
                        qcell,
                        VisitedStates,
                        PreCycleLength,
                        CycleLengthLocal);
                    Q[iState][iPrice][iAgent] = qcell;
                }
            }
        } else if (initType == 'G') {
            for (int jAgent = 1; jAgent <= numAgents; ++jAgent) {
                const int v = static_cast<int>(std::llround(parQInitialization[iAgent][2]));
                for (int iState = 1; iState <= numStates; ++iState) {
                    Strategy[iState][jAgent] = v;
                }
            }

            auto p = runtime::make2<int>(DepthState, numAgents, 0);
            const int grim = static_cast<int>(std::llround(parQInitialization[iAgent][1]));
            for (int d = 1; d <= DepthState; ++d) {
                for (int a = 1; a <= numAgents; ++a) {
                    p[d][a] = grim;
                }
            }
            const int s = computeStateNumber(p);
            for (int a = 1; a <= numAgents; ++a) {
                Strategy[s][a] = grim;
            }

            for (int iState = 1; iState <= numStates; ++iState) {
                for (int iPrice = 1; iPrice <= numPrices; ++iPrice) {
                    double qcell = 0.0;
                    computeQCell(
                        Strategy,
                        iState,
                        iPrice,
                        iAgent,
                        delta,
                        qcell,
                        VisitedStates,
                        PreCycleLength,
                        CycleLengthLocal);
                    Q[iState][iPrice][iAgent] = qcell;
                }
            }
        } else if (initType == 'O') {
            for (int iPrice = 1; iPrice <= numPrices; ++iPrice) {
                int cnt = 0;
                double sumPI = 0.0;
                for (int a = 1; a <= numActions; ++a) {
                    if (indexActions[a][iAgent] == iPrice) {
                        ++cnt;
                        sumPI += PI[a][iAgent];
                    }
                }
                const double den = static_cast<double>(cnt) * (1.0 - delta);
                const double v = sumPI / den;
                for (int iState = 1; iState <= numStates; ++iState) {
                    Q[iState][iPrice][iAgent] = v;
                }
            }
        } else if (initType == 'T') {
            const int sourceExperiment = static_cast<int>(std::llround(parQInitialization[iAgent][1]));
            const std::string codExperimentChar = format_int_fortran_i0w(sourceExperiment, LengthFormatTotExperimentsPrint);
            const int sampledSession = 1 + static_cast<int>(static_cast<double>(numSessions) * generic_routines::ran2(idumQ, ivQ, iyQ, idum2Q));
            const std::string iChar = format_int_fortran_i0w(sampledSession, LengthFormatNumSessionsPrint);
            std::string qFileName = "Q_" + codExperimentChar + "_" + iChar + ".txt";
            qFileName = trim_copy(QFileFolderName[iAgent]) + trim_copy(qFileName);

            std::ifstream in(qFileName);
            if (!in.is_open()) {
                throw std::runtime_error("Cannot open trained Q file: " + qFileName);
            }

            std::string line;
            for (int skip = 1; skip <= (iAgent - 1) * numStates; ++skip) {
                std::getline(in, line);
            }

            for (int iState = 1; iState <= numStates; ++iState) {
                read_nonempty_line(in, line);
                std::istringstream iss(line);
                for (int iPrice = 1; iPrice <= numPrices; ++iPrice) {
                    iss >> Q[iState][iPrice][iAgent];
                }
            }
        } else if (initType == 'R') {
            for (int iState = 1; iState <= numStates; ++iState) {
                for (int iPrice = 1; iPrice <= numPrices; ++iPrice) {
                    Q[iState][iPrice][iAgent] = generic_routines::ran2(idumQ, ivQ, iyQ, idum2Q);
                }
            }

            for (int iState = 1; iState <= numStates; ++iState) {
                for (int iPrice = 1; iPrice <= numPrices; ++iPrice) {
                    Q[iState][iPrice][iAgent] =
                        parQInitialization[iAgent][1] +
                        (parQInitialization[iAgent][2] - parQInitialization[iAgent][1]) * Q[iState][iPrice][iAgent];
                }
            }
        } else if (initType == 'U') {
            for (int iState = 1; iState <= numStates; ++iState) {
                for (int iPrice = 1; iPrice <= numPrices; ++iPrice) {
                    Q[iState][iPrice][iAgent] = parQInitialization[iAgent][1];
                }
            }
        }
    }

    for (int iAgent = 1; iAgent <= numAgents; ++iAgent) {
        std::vector<double> row = runtime::make1<double>(numPrices, 0.0);
        for (int iState = 1; iState <= numStates; ++iState) {
            for (int iPrice = 1; iPrice <= numPrices; ++iPrice) {
                row[iPrice] = Q[iState][iPrice][iAgent];
            }
            generic_routines::MaxLocBreakTies(
                numPrices,
                row,
                idumQ,
                ivQ,
                iyQ,
                idum2Q,
                maxValQ[iState][iAgent],
                maxLocQ[iState][iAgent]);
        }
    }
}

void initState(
    const std::vector<std::vector<double>>& u,
    std::vector<std::vector<int>>& p,
    int& stateNumber,
    int& actionNumber) {
    using namespace globals;

    for (int d = 1; d <= DepthState; ++d) {
        for (int a = 1; a <= numAgents; ++a) {
            p[d][a] = 1 + static_cast<int>(static_cast<double>(numPrices) * u[d][a]);
        }
    }

    stateNumber = computeStateNumber(p);
    std::vector<int> p1 = runtime::make1<int>(numAgents, 0);
    for (int a = 1; a <= numAgents; ++a) {
        p1[a] = p[1][a];
    }
    actionNumber = computeActionNumber(p1);
}

void generate_uIniPrice(
    std::vector<std::vector<std::vector<double>>>& uIniPrice,
    int& idum,
    std::vector<int>& iv,
    int& iy,
    int& idum2) {
    using namespace globals;

    for (int iSession = 1; iSession <= numSessions; ++iSession) {
        for (int iDepth = 1; iDepth <= DepthState; ++iDepth) {
            for (int iAgent = 1; iAgent <= numAgents; ++iAgent) {
                uIniPrice[iDepth][iAgent][iSession] = generic_routines::ran2(idum, iv, iy, idum2);
            }
        }
    }
}

void generateUExploration(
    std::vector<std::vector<double>>& uExploration,
    int& idum,
    std::vector<int>& iv,
    int& iy,
    int& idum2) {
    using namespace globals;

    for (int iDecision = 1; iDecision <= 2; ++iDecision) {
        for (int iAgent = 1; iAgent <= numAgents; ++iAgent) {
            uExploration[iDecision][iAgent] = generic_routines::ran2(idum, iv, iy, idum2);
        }
    }
}

int computeStateNumber(const std::vector<std::vector<int>>& p) {
    using namespace globals;

    if (DepthState0 > 0) {
        const auto stateVector = runtime::flatten_transpose_depth_agent(p, DepthState, numAgents);
        int state = 1;
        for (int i = 1; i <= LengthStates; ++i) {
            state += cStates[i] * (stateVector[i] - 1);
        }
        return state;
    }
    return 1;
}

int computeActionNumber(const std::vector<int>& p) {
    using namespace globals;

    int action = 1;
    for (int i = 1; i <= numAgents; ++i) {
        action += cActions[i] * (p[i] - 1);
    }
    return action;
}

std::vector<std::string> computeStatesCodePrint() {
    using namespace globals;

    auto out = runtime::make1<std::string>(numStates, std::string());

    for (int i = 1; i <= numStates; ++i) {
        const auto indexState = generic_routines::convertNumberBase(i - 1, numPrices, LengthStates);
        std::string label;

        for (int j = 1; j <= LengthStates; ++j) {
            std::ostringstream oss;
            oss << std::setw(LengthFormatActionPrint) << std::setfill('0') << indexState[j];
            const std::string tmp = oss.str();

            if (j == 1) {
                label = tmp;
            } else if ((j % numAgents) != 1) {
                label += "." + tmp;
            } else {
                label += "-" + tmp;
            }
        }

        out[i] = label;
    }

    return out;
}

std::vector<int> computeStrategyNumber(const std::vector<std::vector<int>>& maxLocQ) {
    using namespace globals;

    auto out = runtime::make1<int>(lengthStrategies, 0);
    int iu = 0;

    for (int i = 1; i <= numAgents; ++i) {
        const int il = iu + 1;
        iu += numStates;
        for (int s = 1; s <= numStates; ++s) {
            out[il + s - 1] = maxLocQ[s][i];
        }
    }

    return out;
}

void computeQCell(
    const std::vector<std::vector<int>>& OptimalStrategy,
    int iState,
    int iPrice,
    int iAgent,
    double delta,
    double& QCell,
    std::vector<int>& VisitedStates,
    int& PreCycleLength,
    int& CycleLengthLocal) {
    using namespace globals;

    auto p = runtime::reshape_to_depth_agent(
        generic_routines::convertNumberBase(iState - 1, numPrices, numAgents * DepthState),
        DepthState,
        numAgents);

    auto pPrime = runtime::make1<int>(numAgents, 0);
    for (int a = 1; a <= numAgents; ++a) {
        pPrime[a] = OptimalStrategy[iState][a];
    }
    pPrime[iAgent] = iPrice;

    VisitedStates = runtime::make1<int>(numPeriods, 0);
    std::vector<double> VisitedProfits = runtime::make1<double>(numPeriods, 0.0);

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

        VisitedStates[iPeriod] = computeStateNumber(p);
        VisitedProfits[iPeriod] = PI[computeActionNumber(pPrime)][iAgent];

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
            PreCycleLength = first_repeat_index(VisitedStates, iPeriod - 1, VisitedStates[iPeriod]);
            CycleLengthLocal = iPeriod - PreCycleLength;
            break;
        }

        for (int a = 1; a <= numAgents; ++a) {
            pPrime[a] = OptimalStrategy[VisitedStates[iPeriod]][a];
        }
    }

    double PreCycleProfit = 0.0;
    for (int t = 0; t <= PreCycleLength - 1; ++t) {
        PreCycleProfit += DiscountFactors[t] * VisitedProfits[t + 1];
    }

    double CycleProfit = 0.0;
    for (int t = 0; t <= CycleLengthLocal - 1; ++t) {
        CycleProfit += DiscountFactors[t] * VisitedProfits[PreCycleLength + 1 + t];
    }

    QCell = PreCycleProfit +
            std::pow(delta, static_cast<double>(PreCycleLength)) * CycleProfit /
                (1.0 - std::pow(delta, static_cast<double>(CycleLengthLocal)));
}

void ReadInfoExperiment() {
    using namespace globals;

    std::ifstream in(FileNameInfoExperiment);
    if (!in.is_open()) {
        throw std::runtime_error("Cannot open info experiment file: " + FileNameInfoExperiment);
    }

    std::string line;
    for (int iSession = 1; iSession <= numSessions; ++iSession) {
        if ((iSession % 100) == 0) {
            std::cout << "Read " << iSession << " strategies\n";
        }

        int rSession = 0;
        read_nonempty_line(in, line);
        {
            std::istringstream iss(line);
            iss >> rSession;
        }

        read_nonempty_line(in, line);
        {
            std::istringstream iss(line);
            iss >> converged[iSession];
        }

        read_nonempty_line(in, line);
        {
            std::istringstream iss(line);
            iss >> timeToConvergence[iSession];
        }

        read_nonempty_line(in, line);
        {
            std::istringstream iss(line);
            iss >> CycleLength[iSession];
        }

        read_nonempty_line(in, line);
        {
            std::istringstream iss(line);
            for (int iCycle = 1; iCycle <= CycleLength[iSession]; ++iCycle) {
                iss >> CycleStates[iCycle][iSession];
            }
        }

        for (int iAgent = 1; iAgent <= numAgents; ++iAgent) {
            read_nonempty_line(in, line);
            std::istringstream iss(line);
            for (int iCycle = 1; iCycle <= CycleLength[iSession]; ++iCycle) {
                iss >> CyclePrices[iAgent][iCycle][iSession];
            }
        }

        for (int iAgent = 1; iAgent <= numAgents; ++iAgent) {
            read_nonempty_line(in, line);
            std::istringstream iss(line);
            for (int iCycle = 1; iCycle <= CycleLength[iSession]; ++iCycle) {
                iss >> CycleProfits[iAgent][iCycle][iSession];
            }
        }

        for (int iState = 1; iState <= numStates; ++iState) {
            read_nonempty_line(in, line);
            std::istringstream iss(line);
            for (int iAgent = 1; iAgent <= numAgents; ++iAgent) {
                iss >> indexStrategies[(iAgent - 1) * numStates + iState][iSession];
            }
        }
    }

    std::cout << "Finished reading InfoExperiment\n";
}

}  // namespace QL_routines
