// LearningSimulationNoise.cpp
//
// Friction: noisy monitoring.
//
// Each period, each agent observes the opponent's previous price through
// an i.i.d. Gaussian measurement error applied to the price INDEX. The
// noisy observation is rounded to the nearest grid index and clamped to
// [1, numPrices]. The agent uses this corrupted observation as its state
// and cannot "denoise". At sigma = 0 the behaviour is identical to the
// baseline LearningSimulation.cpp.
//
// Design notes
// ------------
// * Noise is drawn independently per agent and per iteration.
// * The state is PER-AGENT because each agent sees its own noisy draw.
// * Profits are computed from the TRUE joint action so rewards are
//   unaffected by the information friction.
// * A per-session std::mt19937_64 is used for Gaussian draws (seeded from
//   the session index). This is a separate stream from the Calvano
//   Ran2-style RNG used for exploration and Q initialisation, so
//   sigma = 0 reproduces the baseline bit-for-bit (no draws are taken).

#include "LearningSimulationNoise.hpp"

#include "QL_routines.hpp"
#include "generic_routines.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>

// ---------------------------------------------------------------------------
// Local helpers
// ---------------------------------------------------------------------------
namespace {

void write_int_field(std::ostream& os, int value, int width) {
    os << std::setw(width) << std::setfill(' ') << value;
}

void write_char_field(std::ostream& os, char value, int width) {
    os << std::setw(width) << std::setfill(' ') << value;
}

void write_real_field(std::ostream& os, double value, int width, int precision) {
    os << std::setw(width) << std::setfill(' ') << std::fixed
       << std::setprecision(precision) << value;
}

// Compute the state number that agent `iAgent` perceives given its own
// true lagged price and a (possibly noisy) observation of the opponent.
// The encoding matches the original computeStateNumber for n=2, k=1:
//   state = 1 + (price_agent1 - 1)*numPrices + (price_agent2 - 1)
int computeStateNoise(int ownPrice, int observedOppPrice, int iAgent) {
    using namespace globals;
    assert(numAgents == 2 && DepthState == 1);
    int p1, p2;
    if (iAgent == 1) { p1 = ownPrice; p2 = observedOppPrice; }
    else             { p1 = observedOppPrice; p2 = ownPrice; }
    return 1 + (p1 - 1) * numPrices + (p2 - 1);
}

// Draw a noisy observation of `trueIdx` (a 1-based price index).
// At sigma == 0 this short-circuits to `trueIdx` and DOES NOT touch the
// RNG, so sigma = 0 reproduces the baseline deterministically.
int observeWithNoise(
    int trueIdx, double sigma,
    std::mt19937_64& rng, std::normal_distribution<double>& nd)
{
    using namespace globals;
    if (sigma <= 0.0) { return trueIdx; }
    const double eps   = nd(rng);                 // N(0, 1)
    const double noisy = static_cast<double>(trueIdx) + sigma * eps;
    int obs = static_cast<int>(std::lround(noisy));
    if (obs < 1)         { obs = 1; }
    if (obs > numPrices) { obs = numPrices; }
    return obs;
}

// Epsilon-greedy / Boltzmann action selection using per-agent states.
// Identical logic to computePPrimeLatency in LearningSimulationLatency.cpp.
void computePPrimeNoise(
    const std::vector<double>& ExplorationParameters,
    const std::vector<std::vector<double>>& uExploration,
    const std::vector<std::vector<int>>& strategyPrime,
    const std::vector<int>& statePerAgent,
    std::vector<int>& pPrime,
    const std::vector<std::vector<std::vector<double>>>& Q,
    std::vector<double>& eps)
{
    using namespace globals;

    if (typeExplorationMechanism == 1) {
        for (int iAgent = 1; iAgent <= numAgents; ++iAgent) {
            const int s = statePerAgent[iAgent];
            if (MExpl[iAgent] < 0.0) {
                pPrime[iAgent] = strategyPrime[s][iAgent];
            } else {
                const double u1 = uExploration[1][iAgent];
                const double u2 = uExploration[2][iAgent];
                if (u1 <= eps[iAgent]) {
                    pPrime[iAgent] = 1 + static_cast<int>(
                        static_cast<double>(numPrices) * u2);
                } else {
                    pPrime[iAgent] = strategyPrime[s][iAgent];
                }
                eps[iAgent] *= ExplorationParameters[iAgent];
            }
        }
    }

    if (typeExplorationMechanism == 2) {
        for (int iAgent = 1; iAgent <= numAgents; ++iAgent) {
            const int s = statePerAgent[iAgent];
            double maxQ = Q[s][1][iAgent];
            for (int iPrice = 2; iPrice <= numPrices; ++iPrice) {
                maxQ = std::max(maxQ, Q[s][iPrice][iAgent]);
            }

            std::vector<double> probs = runtime::make1<double>(numPrices, 0.0);
            double sumProbs = 0.0;
            for (int iPrice = 1; iPrice <= numPrices; ++iPrice) {
                probs[iPrice] = std::exp(
                    (Q[s][iPrice][iAgent] - maxQ) / eps[iAgent]);
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
            eps[iAgent] *= ExplorationParameters[iAgent];
        }
    }
}

}  // anonymous namespace

// ---------------------------------------------------------------------------
// Public implementation
// ---------------------------------------------------------------------------
namespace LearningSimulationNoise {

void computeExperimentNoise(
    int iExperiment,
    int codExperiment,
    const std::vector<double>& alpha,
    const std::vector<double>& ExplorationParameters,
    double delta,
    double sigma)
{
    using namespace globals;

    // -----------------------------------------------------------------------
    // Initialise session-level accumulators (identical to baseline)
    // -----------------------------------------------------------------------
    for (int i = 1; i <= numSessions; ++i) {
        converged[i]          = 0;
        timeToConvergence[i]  = 0.0;
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

    const std::string codExperimentChar = [](int value, int width) {
        std::ostringstream oss;
        oss << std::setw(width) << std::setfill('0') << value;
        return oss.str();
    }(codExperiment, LengthFormatTotExperimentsPrint);

    // Shared random state for initial-price generation (identical to baseline)
    int idumIP = -1;
    int idum2IP = 123456789;
    std::vector<int> ivIP(33, 0);
    int iyIP = 0;
    auto uIniPrice =
        runtime::make3<double>(DepthState, numAgents, numSessions, 0.0);
    QL_routines::generate_uIniPrice(uIniPrice, idumIP, ivIP, iyIP, idum2IP);

    // -----------------------------------------------------------------------
    // Session loop
    // -----------------------------------------------------------------------
    for (int iSession = 1; iSession <= numSessions; ++iSession) {
        std::cout << "Session = " << iSession << " started\n";

        // Random state for exploration draws (Calvano Ran2)
        int idum  = -iSession;
        int idum2 = 123456789;
        std::vector<int> iv(33, 0);
        int iy = 0;

        // Random state for Q-initialisation tie-breaking (Calvano Ran2)
        int idumQ  = -iSession;
        int idum2Q = 123456789;
        std::vector<int> ivQ(33, 0);
        int iyQ = 0;

        // Independent RNG stream for Gaussian noise draws.
        // Seeded deterministically from the session index. When sigma == 0
        // no draws are taken, so the baseline is reproduced bit-for-bit.
        std::mt19937_64 noiseRng(
            static_cast<std::uint64_t>(iSession) * 1000003ULL + 42ULL);
        std::normal_distribution<double> normal01(0.0, 1.0);

        // Q-tables and derived structures
        auto Q            = runtime::make3<double>(numStates, numPrices, numAgents, 0.0);
        auto maxValQLocal = runtime::make2<double>(numStates, numAgents, 0.0);
        auto strategyPrime = runtime::make2<int>(numStates, numAgents, 0);

        QL_routines::initQMatrices(
            iSession, idumQ, ivQ, iyQ, idum2Q, PI, delta,
            Q, maxValQLocal, strategyPrime);

        auto strategy = strategyPrime;

        // Initial prices (identical to baseline)
        auto p = runtime::make2<int>(DepthState, numAgents, 0);
        int statePrime  = 0;
        int actionPrime = 0;

        auto uIniSlice = runtime::make2<double>(DepthState, numAgents, 0.0);
        for (int d = 1; d <= DepthState; ++d) {
            for (int a = 1; a <= numAgents; ++a) {
                uIniSlice[d][a] = uIniPrice[d][a][iSession];
            }
        }
        QL_routines::initState(uIniSlice, p, statePrime, actionPrime);
        // After initState: p[1][a] = initial price for agent a.

        // -----------------------------------------------------------------------
        // Per-agent state initialisation
        //   statePerAgent[i] = (own_true_price, noisy_obs_of_opp_true_price)
        // One Gaussian draw per agent at session start.
        // -----------------------------------------------------------------------
        auto statePerAgent      = runtime::make1<int>(numAgents, 1);
        auto statePrimePerAgent = runtime::make1<int>(numAgents, 1);

        for (int iAgent = 1; iAgent <= numAgents; ++iAgent) {
            const int jAgent = 3 - iAgent;   // opponent (works for n=2)
            const int ownTrue = p[1][iAgent];
            const int oppTrue = p[1][jAgent];
            const int oppObs  = observeWithNoise(oppTrue, sigma, noiseRng, normal01);
            statePerAgent[iAgent] =
                computeStateNoise(ownTrue, oppObs, iAgent);
        }

        // Track the current "true" price of each agent (the last action each
        // agent actually played). Needed so we can feed a noisy observation
        // of the opponent's TRUE price into the next state computation.
        auto currentPrice = runtime::make1<int>(numAgents, 0);
        for (int a = 1; a <= numAgents; ++a) { currentPrice[a] = p[1][a]; }

        // Loop counters (identical to baseline)
        int iIters           = 0;
        int iItersFix        = 0;
        int iItersInStrategy = 0;
        int convergedSession = -1;

        auto strategyFix = runtime::make2<int>(numStates, numAgents, 0);
        int  stateFix    = statePerAgent[1];   // agent-1 state at convergence

        // Exploration parameter initialisation (identical to baseline)
        auto eps = runtime::make1<double>(numAgents, 0.0);
        if (typeExplorationMechanism == 1) {
            for (int iAgent = 1; iAgent <= numAgents; ++iAgent) { eps[iAgent] = 1.0; }
        }
        if (typeExplorationMechanism == 2) {
            for (int iAgent = 1; iAgent <= numAgents; ++iAgent) { eps[iAgent] = 1.0e3; }
        }

        auto uExploration = runtime::make2<double>(2, numAgents, 0.0);
        auto pPrime       = runtime::make1<int>(numAgents, 0);

        // -----------------------------------------------------------------------
        // Main learning loop
        // -----------------------------------------------------------------------
        while (true) {
            ++iIters;

            // 1. Draw exploration random numbers
            QL_routines::generateUExploration(uExploration, idum, iv, iy, idum2);

            // 2. Each agent picks an action based on its own perceived state
            computePPrimeNoise(
                ExplorationParameters, uExploration,
                strategyPrime, statePerAgent,
                pPrime, Q, eps);

            // 3. Update current true prices
            for (int a = 1; a <= numAgents; ++a) {
                currentPrice[a] = pPrime[a];
            }

            // 4. Compute next perceived states with a FRESH noise draw per agent
            for (int iAgent = 1; iAgent <= numAgents; ++iAgent) {
                const int jAgent  = 3 - iAgent;
                const int ownTrue = currentPrice[iAgent];
                const int oppTrue = currentPrice[jAgent];
                const int oppObs  = observeWithNoise(oppTrue, sigma, noiseRng, normal01);
                statePrimePerAgent[iAgent] =
                    computeStateNoise(ownTrue, oppObs, iAgent);
            }

            // 5. Actual joint action and profit (uses TRUE prices, not noisy)
            actionPrime = QL_routines::computeActionNumber(pPrime);

            // 6. Q-update: each agent uses its own current/next state
            for (int iAgent = 1; iAgent <= numAgents; ++iAgent) {
                const int s      = statePerAgent[iAgent];
                const int sPrime = statePrimePerAgent[iAgent];

                const double oldq = Q[s][pPrime[iAgent]][iAgent];
                const double newq = oldq + alpha[iAgent] * (
                    PI[actionPrime][iAgent]
                    + delta * maxValQLocal[sPrime][iAgent]
                    - oldq);

                Q[s][pPrime[iAgent]][iAgent] = newq;

                // Update max-value cache and greedy strategy
                if (newq > maxValQLocal[s][iAgent]) {
                    maxValQLocal[s][iAgent]  = newq;
                    strategyPrime[s][iAgent] = pPrime[iAgent];
                }

                if ((newq < maxValQLocal[s][iAgent]) &&
                    (strategyPrime[s][iAgent] == pPrime[iAgent])) {
                    std::vector<double> row =
                        runtime::make1<double>(numPrices, 0.0);
                    for (int iPrice = 1; iPrice <= numPrices; ++iPrice) {
                        row[iPrice] = Q[s][iPrice][iAgent];
                    }
                    generic_routines::MaxLocBreakTies(
                        numPrices, row,
                        idumQ, ivQ, iyQ, idum2Q,
                        maxValQLocal[s][iAgent],
                        strategyPrime[s][iAgent]);
                }
            }

            // 7. Convergence check: each agent's policy stable at its own state
            bool equalOnState = true;
            for (int a = 1; a <= numAgents; ++a) {
                if (strategyPrime[statePerAgent[a]][a] !=
                    strategy[statePerAgent[a]][a]) {
                    equalOnState = false;
                    break;
                }
            }
            if (equalOnState) { ++iItersInStrategy; }
            else              { iItersInStrategy = 1; }

            if (convergedSession == -1) {
                if (iIters > maxIters) {
                    convergedSession = 0;
                    strategyFix  = strategy;
                    stateFix     = statePerAgent[1];
                    iItersFix    = iIters;
                }
                if (iItersInStrategy == itersInPerfMeasPeriod) {
                    convergedSession = 1;
                    strategyFix  = strategy;
                    stateFix     = statePerAgent[1];
                    iItersFix    = iIters;
                }
            }

            if (convergedSession != -1) { break; }

            // 8. Advance strategy snapshot and per-agent states
            for (int a = 1; a <= numAgents; ++a) {
                strategy[statePerAgent[a]][a] = strategyPrime[statePerAgent[a]][a];
            }
            statePerAgent = statePrimePerAgent;

        }  // end while(true)

        // -----------------------------------------------------------------------
        // Q print-out (identical to baseline)
        // -----------------------------------------------------------------------
        if (printQ == 1) {
            std::ostringstream sss;
            sss << std::setw(LengthFormatNumSessionsPrint)
                << std::setfill('0') << iSession;
            const std::string QFileName =
                "Q_" + codExperimentChar + "_" + sss.str() + ".txt";

            std::ofstream qout(QFileName);
            for (int iAgent = 1; iAgent <= numAgents; ++iAgent) {
                for (int iState = 1; iState <= numStates; ++iState) {
                    for (int iPrice = 1; iPrice <= numPrices; ++iPrice) {
                        qout << Q[iState][iPrice][iAgent]
                             << (iPrice == numPrices ? '\n' : ' ');
                    }
                }
            }
        }

        // -----------------------------------------------------------------------
        // Record session results (identical to baseline)
        // -----------------------------------------------------------------------
        converged[iSession] = convergedSession;
        timeToConvergence[iSession] =
            static_cast<double>(iItersFix - itersInPerfMeasPeriod) /
            static_cast<double>(itersPerEpisode);

        const auto lastState =
            generic_routines::convertNumberBase(stateFix - 1, numPrices, LengthStates);
        for (int k = 1; k <= LengthStates; ++k) {
            indexLastState[k][iSession] = lastState[k];
        }

        const auto strategyNumber =
            QL_routines::computeStrategyNumber(strategyFix);
        for (int k = 1; k <= lengthStrategies; ++k) {
            indexStrategies[k][iSession] = strategyNumber[k];
        }

        if (convergedSession == 1) {
            std::cout << "Session = " << iSession << " converged\n";
        }
        if (convergedSession == 0) {
            std::cout << "Session = " << iSession << " did not converge\n";
        }
    }  // end session loop

    // -----------------------------------------------------------------------
    // Write InfoExperiment file (identical to baseline)
    // -----------------------------------------------------------------------
    {
        std::ofstream out(FileNameInfoExperiment, std::ios::trunc);
        for (int iSession = 1; iSession <= numSessions; ++iSession) {
            out << iSession << '\n';
            out << converged[iSession] << '\n';
            out << timeToConvergence[iSession] << '\n';

            for (int i = 1; i <= LengthStates; ++i) {
                out << indexLastState[i][iSession]
                    << (i == LengthStates ? '\n' : ' ');
            }

            for (int iState = 1; iState <= numStates; ++iState) {
                for (int iAgent = 1; iAgent <= numAgents; ++iAgent) {
                    out << indexStrategies[(iAgent - 1) * numStates + iState][iSession]
                        << (iAgent == numAgents ? '\n' : ' ');
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Summary statistics and A_res.txt output (identical to baseline)
    // -----------------------------------------------------------------------
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

    double meanTimeToConvergence   = 0.0;
    double seTimeToConvergence     = 0.0;
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
        seTimeToConvergence = std::sqrt(std::max(
            0.0,
            sq / static_cast<double>(numSessionsConverged) -
                meanTimeToConvergence * meanTimeToConvergence));
    }

    std::vector<double> ttc;
    ttc.reserve(numSessions);
    for (int i = 1; i <= numSessions; ++i) {
        ttc.push_back(timeToConvergence[i]);
    }
    std::sort(ttc.begin(), ttc.end());
    int med_idx = static_cast<int>(
        std::floor(0.5 * static_cast<double>(numSessions) + 0.5));
    med_idx = std::max(1, std::min(numSessions, med_idx));
    medianTimeToConvergence = ttc[med_idx - 1];

    std::fstream& res = runtime::io::unit(10002);
    if (iExperiment == 1) {
        res << "Experiment ";
        for (int i = 1; i <= numAgents; ++i) { res << "    alpha" << i << ' '; }
        for (int i = 1; i <= numExplorationParameters; ++i) { res << "     beta" << i << ' '; }
        res << "     delta ";
        for (int i = 1; i <= numAgents; ++i) {
            res << "typeQini" << i << ' ';
            for (int j = 1; j <= numAgents; ++j) { res << "par" << j << "Qini" << i << ' '; }
        }
        for (int i = 1; i <= numDemandParameters; ++i) {
            res << "  DemPar" << std::setw(2) << std::setfill('0') << i
                << std::setfill(' ') << ' ';
        }
        for (int i = 1; i <= numAgents; ++i) { res << "NashPrice" << i << ' '; }
        for (int i = 1; i <= numAgents; ++i) { res << "CoopPrice" << i << ' '; }
        for (int i = 1; i <= numAgents; ++i) { res << "NashProft" << i << ' '; }
        for (int i = 1; i <= numAgents; ++i) { res << "CoopProft" << i << ' '; }
        for (int i = 1; i <= numAgents; ++i) { res << "NashMktSh" << i << ' '; }
        for (int i = 1; i <= numAgents; ++i) { res << "CoopMktSh" << i << ' '; }
        for (int i = 1; i <= numAgents; ++i) {
            for (int j = 1; j <= numPrices; ++j) {
                res << "Ag" << i << "Price"
                    << std::setw(2) << std::setfill('0') << j
                    << std::setfill(' ') << ' ';
            }
        }
        res << "   numConv     avgTTC      seTTC     medTTC \n";
    }

    write_int_field(res, codExperiment, 10); res << ' ';
    for (int i = 1; i <= numAgents; ++i) {
        write_real_field(res, alpha[i], 10, 5); res << ' ';
    }
    for (int i = 1; i <= numExplorationParameters; ++i) {
        write_real_field(res, MExpl[i], 10, 5); res << ' ';
    }
    write_real_field(res, delta, 10, 5); res << ' ';

    for (int i = 1; i <= numAgents; ++i) {
        write_char_field(res, typeQInitialization[i], 9); res << ' ';
        for (int j = 1; j <= numAgents; ++j) {
            write_real_field(res, parQInitialization[i][j], 9, 2); res << ' ';
        }
    }
    for (int i = 1; i <= numDemandParameters; ++i) {
        write_real_field(res, DemandParameters[i], 10, 5); res << ' ';
    }
    for (int i = 1; i <= numAgents; ++i) {
        write_real_field(res, NashPrices[i], 10, 5); res << ' ';
    }
    for (int i = 1; i <= numAgents; ++i) {
        write_real_field(res, CoopPrices[i], 10, 5); res << ' ';
    }
    for (int i = 1; i <= numAgents; ++i) {
        write_real_field(res, NashProfits[i], 10, 5); res << ' ';
    }
    for (int i = 1; i <= numAgents; ++i) {
        write_real_field(res, CoopProfits[i], 10, 5); res << ' ';
    }
    for (int i = 1; i <= numAgents; ++i) {
        write_real_field(res, NashMarketShares[i], 10, 5); res << ' ';
    }
    for (int i = 1; i <= numAgents; ++i) {
        write_real_field(res, CoopMarketShares[i], 10, 5); res << ' ';
    }
    for (int i = 1; i <= numAgents; ++i) {
        for (int j = 1; j <= numPrices; ++j) {
            write_real_field(res, PricesGrids[j][i], 10, 7); res << ' ';
        }
    }

    write_int_field(res, numSessionsConverged, 10); res << ' ';
    write_real_field(res, meanTimeToConvergence, 10, 2); res << ' ';
    write_real_field(res, seTimeToConvergence, 10, 2); res << ' ';
    write_real_field(res, medianTimeToConvergence, 10, 2);
    res << '\n';
}

}  // namespace LearningSimulationNoise
