#include "globals.hpp"

#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace globals {

int numExperiments = 0;
int totExperiments = 0;
int numCores = 1;
int numSessions = 0;
int itersPerEpisode = 0;
int maxNumEpisodes = 0;
int maxIters = 0;
int itersInPerfMeasPeriod = 0;
int printQ = 0;
int codExperiment = 0;
int numPrices = 0;
int typeExplorationMechanism = 0;
int DepthState0 = 0;
int DepthState = 1;
int LengthStates = 1;
int numStates = 1;
int lengthStrategies = 0;
int LengthFormatStatesPrint = 0;
int LengthFormatActionPrint = 0;
int LengthFormatTotExperimentsPrint = 0;
int LengthFormatNumSessionsPrint = 0;
int typePayoffInput = 0;
int numAgents = 0;
int numActions = 0;
int numDemandParameters = 0;
int numPeriods = 0;
int numExplorationParameters = 0;
int SwitchImpulseResponseToBR = 0;
int SwitchImpulseResponseToNash = 0;
int SwitchImpulseResponseToAll = 0;
int SwitchEquilibriumCheck = 0;
int SwitchQGapToMaximum = 0;
std::vector<int> ParamsLearningTrajectory(3, 0);
int SwitchDetailedAnalysis = 0;

double delta = 0.0;
double PerfMeasPeriodLength = 0.0;
double meanNashProfit = 0.0;
double meanCoopProfit = 0.0;
double gammaSinghVives = 0.0;

std::string ExperimentNumber;
std::string FileNameInfoExperiment;

std::vector<int> converged;
std::vector<std::vector<int>> indexStrategies;
std::vector<std::vector<int>> indexLastState;
std::vector<int> CycleLength;
std::vector<std::vector<int>> CycleStates;
std::vector<std::vector<std::vector<int>>> CyclePrices;
std::vector<std::vector<int>> indexActions;
std::vector<int> cStates;
std::vector<int> cActions;
std::vector<int> indexNashPrices;
std::vector<int> indexCoopPrices;

std::vector<double> timeToConvergence;
std::vector<std::vector<std::vector<double>>> CycleProfits;
std::vector<double> NashProfits;
std::vector<double> CoopProfits;
std::vector<std::vector<double>> maxValQ;
std::vector<double> NashPrices;
std::vector<double> CoopPrices;
std::vector<std::vector<double>> PI;
std::vector<std::vector<double>> PIQ;
std::vector<double> avgPI;
std::vector<double> avgPIQ;
std::vector<std::vector<double>> PG;
std::vector<std::vector<double>> PGQ;
std::vector<double> avgPG;
std::vector<double> avgPGQ;
std::vector<double> alpha;
std::vector<double> DiscountFactors;
std::vector<double> DemandParameters;
std::vector<double> MExpl;
std::vector<double> ExplorationParameters;
std::vector<double> NashMarketShares;
std::vector<double> CoopMarketShares;
std::vector<std::vector<double>> PricesGrids;
std::vector<std::vector<double>> parQInitialization;

std::vector<std::string> labelStates;
std::vector<std::string> QFileFolderName;
std::vector<char> typeQInitialization;

namespace {

int pow_int(int base, int exp) {
    int out = 1;
    for (int i = 0; i < exp; ++i) {
        out *= base;
    }
    return out;
}

void read_nonempty_line(std::fstream& f, std::string& line) {
    while (std::getline(f, line)) {
        if (!line.empty()) {
            return;
        }
    }
    throw std::runtime_error("Unexpected EOF while reading input file");
}

}  // namespace

void readBatchVariables(int unitNumber) {
    std::fstream& f = runtime::io::unit(unitNumber);
    std::string line;

    auto skip_line = [&]() {
        if (!std::getline(f, line)) {
            throw std::runtime_error("Unexpected EOF while skipping line");
        }
    };

    auto read_line_as = [&](auto&& loader) {
        read_nonempty_line(f, line);
        std::istringstream iss(line);
        loader(iss);
    };

    skip_line();
    read_line_as([&](std::istringstream& iss) { iss >> numExperiments >> totExperiments; });
    skip_line();
    read_line_as([&](std::istringstream& iss) { iss >> numCores; });
    skip_line();
    read_line_as([&](std::istringstream& iss) { iss >> numSessions; });
    skip_line();
    read_line_as([&](std::istringstream& iss) { iss >> itersPerEpisode; });
    skip_line();
    read_line_as([&](std::istringstream& iss) { iss >> maxNumEpisodes; });
    skip_line();
    read_line_as([&](std::istringstream& iss) { iss >> PerfMeasPeriodLength; });
    skip_line();
    read_line_as([&](std::istringstream& iss) { iss >> numAgents; });
    skip_line();
    read_line_as([&](std::istringstream& iss) { iss >> DepthState0; });
    DepthState = std::max(1, DepthState0);
    skip_line();
    read_line_as([&](std::istringstream& iss) { iss >> numPrices; });

    LengthFormatTotExperimentsPrint = 1 + static_cast<int>(std::log10(static_cast<double>(totExperiments)));
    LengthFormatNumSessionsPrint = 1 + static_cast<int>(std::log10(static_cast<double>(numSessions)));
    maxIters = maxNumEpisodes * itersPerEpisode;
    itersInPerfMeasPeriod = static_cast<int>(PerfMeasPeriodLength * static_cast<double>(itersPerEpisode));
    LengthStates = std::max(1, numAgents * DepthState0);
    LengthFormatStatesPrint =
        LengthStates * (1 + static_cast<int>(std::floor(std::log10(static_cast<double>(numPrices))))) + LengthStates - 1;
    numStates = pow_int(numPrices, numAgents * DepthState0);
    numPeriods = numStates + 1;
    numActions = pow_int(numPrices, numAgents);
    lengthStrategies = numAgents * numStates;
    LengthFormatActionPrint = static_cast<int>(std::floor(std::log10(static_cast<double>(numPrices)))) + 1;

    skip_line();
    read_line_as([&](std::istringstream& iss) { iss >> typeExplorationMechanism; });
    numExplorationParameters = numAgents;

    skip_line();
    read_line_as([&](std::istringstream& iss) { iss >> typePayoffInput; });
    if (typePayoffInput == 1) {
        numDemandParameters = 3;
    }
    if (typePayoffInput == 2 || typePayoffInput == 3) {
        numDemandParameters = 2 * numAgents + 4;
    }

    skip_line();
    read_line_as([&](std::istringstream& iss) { iss >> SwitchImpulseResponseToBR; });
    skip_line();
    read_line_as([&](std::istringstream& iss) { iss >> SwitchImpulseResponseToNash; });
    skip_line();
    read_line_as([&](std::istringstream& iss) { iss >> SwitchImpulseResponseToAll; });
    skip_line();
    read_line_as([&](std::istringstream& iss) { iss >> SwitchEquilibriumCheck; });
    skip_line();
    read_line_as([&](std::istringstream& iss) { iss >> SwitchQGapToMaximum; });
    skip_line();
    read_line_as([&](std::istringstream& iss) { iss >> ParamsLearningTrajectory[1] >> ParamsLearningTrajectory[2]; });
    skip_line();
    read_line_as([&](std::istringstream& iss) { iss >> SwitchDetailedAnalysis; });
    skip_line();

    converged = runtime::make1<int>(numSessions, 0);
    timeToConvergence = runtime::make1<double>(numSessions, 0.0);
    indexStrategies = runtime::make2<int>(lengthStrategies, numSessions, 0);
    indexLastState = runtime::make2<int>(LengthStates, numSessions, 0);
    CycleLength = runtime::make1<int>(numSessions, 0);
    CycleStates = runtime::make2<int>(numPeriods, numSessions, 0);
    CyclePrices = runtime::make3<int>(numAgents, numPeriods, numSessions, 0);
    CycleProfits = runtime::make3<double>(numAgents, numPeriods, numSessions, 0.0);
    indexActions = runtime::make2<int>(numActions, numAgents, 0);
    cStates = runtime::make1<int>(LengthStates, 0);
    cActions = runtime::make1<int>(numAgents, 0);
    DiscountFactors = runtime::make1<double>(numStates, 0.0);
    maxValQ = runtime::make2<double>(numStates, numAgents, 0.0);
    DemandParameters = runtime::make1<double>(numDemandParameters, 0.0);
    ExplorationParameters = runtime::make1<double>(numExplorationParameters, 0.0);
    MExpl = runtime::make1<double>(numExplorationParameters, 0.0);
    alpha = runtime::make1<double>(numAgents, 0.0);
    NashProfits = runtime::make1<double>(numAgents, 0.0);
    CoopProfits = runtime::make1<double>(numAgents, 0.0);
    PI = runtime::make2<double>(numActions, numAgents, 0.0);
    PIQ = runtime::make2<double>(numActions, numAgents, 0.0);
    avgPI = runtime::make1<double>(numActions, 0.0);
    avgPIQ = runtime::make1<double>(numActions, 0.0);
    PG = runtime::make2<double>(numActions, numAgents, 0.0);
    PGQ = runtime::make2<double>(numActions, numAgents, 0.0);
    avgPG = runtime::make1<double>(numActions, 0.0);
    avgPGQ = runtime::make1<double>(numActions, 0.0);
    indexNashPrices = runtime::make1<int>(numAgents, 0);
    indexCoopPrices = runtime::make1<int>(numAgents, 0);
    NashPrices = runtime::make1<double>(numAgents, 0.0);
    CoopPrices = runtime::make1<double>(numAgents, 0.0);
    typeQInitialization = runtime::make1<char>(numAgents, ' ');
    parQInitialization = runtime::make2<double>(numAgents, numAgents, 0.0);
    NashMarketShares = runtime::make1<double>(numAgents, 0.0);
    CoopMarketShares = runtime::make1<double>(numAgents, 0.0);
    PricesGrids = runtime::make2<double>(numPrices, numAgents, 0.0);

    labelStates = runtime::make1<std::string>(numStates, std::string());
    QFileFolderName = runtime::make1<std::string>(numAgents, std::string(200, ' '));

    for (int i = 1; i <= LengthStates; ++i) {
        cStates[i] = pow_int(numPrices, LengthStates - i);
    }
    for (int i = 1; i <= numAgents; ++i) {
        cActions[i] = pow_int(numPrices, numAgents - i);
    }

    for (int iAction = 1; iAction <= numActions; ++iAction) {
        const auto code = runtime::convert_number_base(iAction - 1, numPrices, numAgents);
        for (int iAgent = 1; iAgent <= numAgents; ++iAgent) {
            indexActions[iAction][iAgent] = code[iAgent];
        }
    }
}

void closeBatch() {
    converged.clear();
    timeToConvergence.clear();
    indexStrategies.clear();
    indexLastState.clear();
    CycleLength.clear();
    CycleStates.clear();
    CyclePrices.clear();
    CycleProfits.clear();
    indexActions.clear();
    cStates.clear();
    cActions.clear();
    maxValQ.clear();
    labelStates.clear();
    NashProfits.clear();
    CoopProfits.clear();
    QFileFolderName.clear();
    DiscountFactors.clear();
    alpha.clear();
    MExpl.clear();
    ExplorationParameters.clear();
    NashPrices.clear();
    CoopPrices.clear();
    DemandParameters.clear();
    PI.clear();
    PIQ.clear();
    avgPI.clear();
    avgPIQ.clear();
    PG.clear();
    PGQ.clear();
    avgPG.clear();
    avgPGQ.clear();
    indexNashPrices.clear();
    indexCoopPrices.clear();
    NashMarketShares.clear();
    CoopMarketShares.clear();
    PricesGrids.clear();
    typeQInitialization.clear();
    parQInitialization.clear();
}

void readExperimentVariables(int unitNumber) {
    std::fstream& f = runtime::io::unit(unitNumber);

    f >> codExperiment >> printQ;
    for (int i = 1; i <= numAgents; ++i) {
        f >> alpha[i];
    }
    for (int i = 1; i <= numExplorationParameters; ++i) {
        f >> MExpl[i];
    }
    f >> delta;
    for (int i = 1; i <= numDemandParameters; ++i) {
        f >> DemandParameters[i];
    }
    for (int i = 1; i <= numAgents; ++i) {
        f >> NashPrices[i];
    }
    for (int i = 1; i <= numAgents; ++i) {
        f >> CoopPrices[i];
    }

    for (int i = 1; i <= numAgents; ++i) {
        std::string qtype;
        f >> qtype;
        typeQInitialization[i] = qtype.empty() ? ' ' : qtype[0];
        for (int j = 1; j <= numAgents; ++j) {
            f >> parQInitialization[i][j];
        }
    }

    if (typeExplorationMechanism == 1) {
        for (int i = 1; i <= numExplorationParameters; ++i) {
            ExplorationParameters[i] = std::exp(-MExpl[i] / static_cast<double>(itersPerEpisode));
        }
    } else if (typeExplorationMechanism == 2) {
        for (int i = 1; i <= numExplorationParameters; ++i) {
            ExplorationParameters[i] = 1.0 - std::pow(10.0, MExpl[i]);
        }
    }

    if (DepthState0 == 0) {
        delta = 0.0;
    }

    for (int i = 0; i <= numPeriods - 1; ++i) {
        DiscountFactors[i] = std::pow(delta, static_cast<double>(i));
    }
}

}  // namespace globals
