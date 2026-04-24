// main_noise.cpp
//
// Entry point for the noisy-monitoring friction experiment.
//
// In addition to the standard A_InputParameters.txt (unchanged format),
// this executable reads a single file:
//
//   A_NoiseStd.txt
//       One line containing a non-negative floating-point number: the
//       standard deviation sigma of the Gaussian noise added to the
//       opponent's observed price INDEX each period.
//       sigma = 0   -> identical to the baseline (exact same output).
//       sigma = s   -> each agent observes
//                      clamp(round(p_opp + N(0, s^2)), 1, numPrices)
//                      with an independent draw each period per agent.
//
// Usage
// -----
//   cd <experiment_folder>
//   ogcode_noise.exe
//
// Each experiment folder should contain:
//   A_InputParameters.txt   (same format as the baseline)
//   A_NoiseStd.txt          (one floating-point number)

#include "ConvergenceResults.hpp"
#include "LearningSimulationNoise.hpp"
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

double read_sigma() {
    std::ifstream f("A_NoiseStd.txt");
    if (!f.is_open()) {
        throw std::runtime_error(
            "Cannot open A_NoiseStd.txt. "
            "Create this file with a single non-negative floating-point "
            "number (noise standard deviation in price-index units).");
    }
    double s = 0.0;
    f >> s;
    if (f.fail() || s < 0.0) {
        throw std::runtime_error(
            "A_NoiseStd.txt must contain a single non-negative real number.");
    }
    std::cout << "Noise std (price-index units): sigma = " << s << "\n";
    return s;
}

}  // namespace

int main() {
    using namespace globals;

    const double sigma = read_sigma();

    runtime::io::open_unit(10001, "A_InputParameters.txt", std::ios::in);
    readBatchVariables(10001);

    runtime::io::open_unit(10002, "A_res.txt",         std::ios::out | std::ios::trunc);
    runtime::io::open_unit(100022, "A_convResults.txt", std::ios::out | std::ios::trunc);

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

        // ---- Core learning simulation with noise friction ----
        LearningSimulationNoise::computeExperimentNoise(
            iExperiment, codExperiment, alpha, ExplorationParameters, delta,
            sigma);

        ConvergenceResults::ComputeConvResults(iExperiment);
    }

    closeBatch();

    runtime::io::close_unit(10001);
    runtime::io::close_unit(10002);
    runtime::io::close_unit(100022);

    return 0;
}
