// main_latency.cpp
//
// Entry point for the observation-latency friction experiment.
//
// In addition to the standard A_InputParameters.txt (unchanged format),
// this executable reads a single file:
//
//   A_Latency.txt
//       One line containing a non-negative integer:  the latency L.
//       L = 0  -> identical to the baseline (exact same output).
//       L = k  -> each agent observes the opponent's price k periods late.
//
// Usage
// -----
//   cd <experiment_folder>
//   ogcode_latency.exe
//
// Each experiment folder should contain:
//   A_InputParameters.txt   (same format as the baseline)
//   A_Latency.txt           (one integer)

#include "ConvergenceResults.hpp"
#include "LearningSimulationLatency.hpp"
#include "PI_routines.hpp"
#include "QL_routines.hpp"
#include "globals.hpp"

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace {

std::string format_experiment_number(int value, int width) {
    std::ostringstream oss;
    oss << std::setw(width) << std::setfill('0') << value << ".txt";
    return oss.str();
}

int read_latency() {
    std::ifstream f("A_Latency.txt");
    if (!f.is_open()) {
        throw std::runtime_error(
            "Cannot open A_Latency.txt. "
            "Create this file with a single integer (the observation lag).");
    }
    int lat = 0;
    f >> lat;
    if (f.fail() || lat < 0) {
        throw std::runtime_error(
            "A_Latency.txt must contain a single non-negative integer.");
    }
    std::cout << "Observation latency: L = " << lat << "\n";
    return lat;
}

}  // namespace

int main() {
    using namespace globals;

    const int latency = read_latency();

    runtime::io::open_unit(10001, "A_InputParameters.txt", std::ios::in);
    readBatchVariables(10001);

    runtime::io::open_unit(10002, "A_res.txt",         std::ios::out | std::ios::trunc);
    runtime::io::open_unit(100022, "A_convResults.txt", std::ios::out | std::ios::trunc);

    // Impulse-response and other optional analyses are not wired up for the
    // friction exe (they can be added later if needed).

    labelStates = QL_routines::computeStatesCodePrint();

    for (int iExperiment = 1; iExperiment <= numExperiments; ++iExperiment) {
        readExperimentVariables(10001);

        for (int iAgent = 1; iAgent <= numAgents; ++iAgent) {
            if (typeQInitialization[iAgent] == 'T') {
                QFileFolderName[iAgent] = "trained_Q/";
            }
        }

        if (typePayoffInput == 1) {
            PI_routines::computePIMatricesSinghVives(
                DemandParameters, NashPrices, CoopPrices,
                PI, NashProfits, CoopProfits,
                indexNashPrices, indexCoopPrices,
                NashMarketShares, CoopMarketShares, PricesGrids);
        }
        if (typePayoffInput == 2) {
            PI_routines::computePIMatricesLogit(
                DemandParameters, NashPrices, CoopPrices,
                PI, NashProfits, CoopProfits,
                indexNashPrices, indexCoopPrices,
                NashMarketShares, CoopMarketShares, PricesGrids);
        }
        if (typePayoffInput == 3) {
            PI_routines::computePIMatricesLogitMu0(
                DemandParameters, NashPrices, CoopPrices,
                PI, NashProfits, CoopProfits,
                indexNashPrices, indexCoopPrices,
                NashMarketShares, CoopMarketShares, PricesGrids);
        }

        for (int iAction = 1; iAction <= numActions; ++iAction) {
            avgPI[iAction] = 0.0;
            for (int iAgent = 1; iAgent <= numAgents; ++iAgent) {
                PIQ[iAction][iAgent] = PI[iAction][iAgent] * PI[iAction][iAgent];
                avgPI[iAction] += PI[iAction][iAgent];
            }
            avgPI[iAction] /= static_cast<double>(numAgents);
            avgPIQ[iAction] = avgPI[iAction] * avgPI[iAction];
        }

        for (int iAgent = 1; iAgent <= numAgents; ++iAgent) {
            for (int iAction = 1; iAction <= numActions; ++iAction) {
                PG[iAction][iAgent] =
                    (PI[iAction][iAgent] - NashProfits[iAgent]) /
                    (CoopProfits[iAgent] - NashProfits[iAgent]);
            }
        }

        for (int iAction = 1; iAction <= numActions; ++iAction) {
            avgPG[iAction] = 0.0;
            for (int iAgent = 1; iAgent <= numAgents; ++iAgent) {
                PGQ[iAction][iAgent] = PG[iAction][iAgent] * PG[iAction][iAgent];
                avgPG[iAction] += PG[iAction][iAgent];
            }
            avgPG[iAction] /= static_cast<double>(numAgents);
            avgPGQ[iAction] = avgPG[iAction] * avgPG[iAction];
        }

        ExperimentNumber      = format_experiment_number(codExperiment, LengthFormatTotExperimentsPrint);
        FileNameInfoExperiment = "InfoExperiment_" + ExperimentNumber;

        std::cout << "model = " << iExperiment
                  << " / numExperiments = " << numExperiments
                  << " / numCores = " << numCores << "\n";

        // ---- Core learning simulation with latency friction ----
        LearningSimulationLatency::computeExperimentLatency(
            iExperiment, codExperiment, alpha, ExplorationParameters, delta,
            latency);

        // Convergence-results analysis (unchanged — reads the InfoExperiment
        // file written above and produces A_convResults.txt)
        ConvergenceResults::ComputeConvResults(iExperiment);
    }

    closeBatch();

    runtime::io::close_unit(10001);
    runtime::io::close_unit(10002);
    runtime::io::close_unit(100022);

    return 0;
}
