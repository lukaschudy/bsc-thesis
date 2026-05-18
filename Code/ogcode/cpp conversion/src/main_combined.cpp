// main_combined.cpp
//
// Entry point for the combined-frictions (2x2x2 factorial) experiment.
//
// Reads the standard A_InputParameters.txt plus three single-parameter files
// (one per friction). Missing files default to the "off" level.
//
//   A_Latency.txt    : one non-negative integer L  (default 0)
//   A_NoiseStd.txt   : one non-negative real sigma (default 0.0)
//   A_LockIn.txt     : one non-negative integer T  (default 0)
//
// Usage
// -----
//   cd <experiment_folder>
//   ogcode_combined.exe
//
// At (L=0, sigma=0, T=0) this reproduces the baseline exactly.

#include "ConvergenceResults.hpp"
#include "LearningSimulationCombined.hpp"
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

int read_int_or_default(const std::string& filename, int dflt, const char* label) {
    std::ifstream f(filename);
    if (!f.is_open()) {
        std::cout << label << ": " << filename
                  << " not found, defaulting to " << dflt << "\n";
        return dflt;
    }
    int v = 0;
    f >> v;
    if (f.fail() || v < 0) {
        throw std::runtime_error(
            filename + " must contain a single non-negative integer.");
    }
    std::cout << label << " = " << v << "\n";
    return v;
}

double read_double_or_default(const std::string& filename, double dflt, const char* label) {
    std::ifstream f(filename);
    if (!f.is_open()) {
        std::cout << label << ": " << filename
                  << " not found, defaulting to " << dflt << "\n";
        return dflt;
    }
    double v = 0.0;
    f >> v;
    if (f.fail() || v < 0.0) {
        throw std::runtime_error(
            filename + " must contain a single non-negative real number.");
    }
    std::cout << label << " = " << v << "\n";
    return v;
}

}  // namespace

int main() {
    using namespace globals;

    const int    latency        = read_int_or_default("A_Latency.txt",  0,  "Latency L");
    const double sigma          = read_double_or_default("A_NoiseStd.txt", 0.0, "Noise sigma");
    const int    lockIn         = read_int_or_default("A_LockIn.txt",   0,  "Lock-in T");
    const int    playbackPeriods = read_int_or_default("A_PlaybackPeriods.txt", 30, "Playback periods");

    std::cout << "Combined frictions: L=" << latency
              << "  sigma=" << sigma
              << "  T=" << lockIn
              << "  playback=" << playbackPeriods << " periods\n";

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

        LearningSimulationCombined::computeExperimentCombined(
            iExperiment, codExperiment, alpha, ExplorationParameters, delta,
            latency, sigma, lockIn, playbackPeriods);

        ConvergenceResults::ComputeConvResults(iExperiment);
    }

    closeBatch();

    runtime::io::close_unit(10001);
    runtime::io::close_unit(10002);
    runtime::io::close_unit(100022);

    return 0;
}
