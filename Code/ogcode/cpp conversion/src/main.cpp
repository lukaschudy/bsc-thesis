#include "ConvergenceResults.hpp"
#include "DetailedAnalysis.hpp"
#include "EquilibriumCheck.hpp"
#include "ImpulseResponse.hpp"
#include "LearningSimulation.hpp"
#include "LearningTrajectory.hpp"
#include "PI_routines.hpp"
#include "QL_routines.hpp"
#include "QGapToMaximum.hpp"
#include "globals.hpp"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>

namespace {

std::string format_experiment_number(int value, int width) {
    std::ostringstream oss;
    oss << std::setw(width) << std::setfill('0') << value << ".txt";
    return oss.str();
}

}  // namespace

int main() {
    using namespace globals;

    runtime::io::open_unit(10001, "A_InputParameters.txt", std::ios::in);
    readBatchVariables(10001);

    runtime::io::open_unit(10002, "A_res.txt", std::ios::out | std::ios::trunc);
    runtime::io::open_unit(100022, "A_convResults.txt", std::ios::out | std::ios::trunc);
    if (SwitchImpulseResponseToBR == 1) {
        runtime::io::open_unit(10003, "A_irToBR.txt", std::ios::out | std::ios::trunc);
    }
    if (SwitchImpulseResponseToNash >= 1) {
        runtime::io::open_unit(100031, "A_irToNash.txt", std::ios::out | std::ios::trunc);
    }
    if (SwitchImpulseResponseToAll == 1) {
        runtime::io::open_unit(100032, "A_irToAll.txt", std::ios::out | std::ios::trunc);
    }
    if (SwitchEquilibriumCheck == 1) {
        runtime::io::open_unit(10004, "A_ec.txt", std::ios::out | std::ios::trunc);
    }
    if (SwitchQGapToMaximum == 1) {
        runtime::io::open_unit(10006, "A_qg.txt", std::ios::out | std::ios::trunc);
    }

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
                DemandParameters,
                NashPrices,
                CoopPrices,
                PI,
                NashProfits,
                CoopProfits,
                indexNashPrices,
                indexCoopPrices,
                NashMarketShares,
                CoopMarketShares,
                PricesGrids);
        }
        if (typePayoffInput == 2) {
            PI_routines::computePIMatricesLogit(
                DemandParameters,
                NashPrices,
                CoopPrices,
                PI,
                NashProfits,
                CoopProfits,
                indexNashPrices,
                indexCoopPrices,
                NashMarketShares,
                CoopMarketShares,
                PricesGrids);
        }
        if (typePayoffInput == 3) {
            PI_routines::computePIMatricesLogitMu0(
                DemandParameters,
                NashPrices,
                CoopPrices,
                PI,
                NashProfits,
                CoopProfits,
                indexNashPrices,
                indexCoopPrices,
                NashMarketShares,
                CoopMarketShares,
                PricesGrids);
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
                    (PI[iAction][iAgent] - NashProfits[iAgent]) / (CoopProfits[iAgent] - NashProfits[iAgent]);
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

        ExperimentNumber = format_experiment_number(codExperiment, LengthFormatTotExperimentsPrint);
        FileNameInfoExperiment = "InfoExperiment_" + ExperimentNumber;

        std::cout << "model = " << iExperiment << " / numExperiments = " << numExperiments
                  << " / numCores = " << numCores << "\n";

        LearningSimulation::computeExperiment(iExperiment, codExperiment, alpha, ExplorationParameters, delta);

        ConvergenceResults::ComputeConvResults(iExperiment);

        if (SwitchImpulseResponseToBR == 1) {
            ImpulseResponse::computeIRAnalysis(iExperiment, 10003, 0);
        }
        if (SwitchImpulseResponseToNash >= 1) {
            ImpulseResponse::computeIRAnalysis(iExperiment, 100031, SwitchImpulseResponseToNash);
        }
        if (SwitchImpulseResponseToAll == 1) {
            for (int i = 1; i <= numPrices; ++i) {
                ImpulseResponse::computeIRAnalysis(iExperiment, 100032, -i);
            }
        }

        if (SwitchEquilibriumCheck == 1) {
            EquilibriumCheck::computeEqCheck(iExperiment);
        }

        if (SwitchQGapToMaximum == 1) {
            QGapToMaximum::computeQGapToMax(iExperiment);
        }

        if (ParamsLearningTrajectory[1] > 0) {
            LearningTrajectory::computeLearningTrajectory(
                iExperiment,
                codExperiment,
                alpha,
                ExplorationParameters,
                delta);
        }

        if (SwitchDetailedAnalysis == 1) {
            DetailedAnalysis::ComputeDetailedAnalysis(iExperiment);
        }
    }

    closeBatch();

    runtime::io::close_unit(10001);
    runtime::io::close_unit(10002);
    runtime::io::close_unit(100022);
    if (SwitchImpulseResponseToBR == 1) {
        runtime::io::close_unit(10003);
    }
    if (SwitchImpulseResponseToNash >= 1) {
        runtime::io::close_unit(100031);
    }
    if (SwitchImpulseResponseToAll == 1) {
        runtime::io::close_unit(100032);
    }
    if (SwitchEquilibriumCheck == 1) {
        runtime::io::close_unit(10004);
    }
    if (SwitchQGapToMaximum == 1) {
        runtime::io::close_unit(10006);
    }

    return 0;
}
