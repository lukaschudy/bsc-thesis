#pragma once

#include "fortran_runtime.hpp"

namespace globals {

constexpr int numShockPeriodsPrint = 10;
constexpr int numThresCycleLength = 10;

extern int numExperiments;
extern int totExperiments;
extern int numCores;
extern int numSessions;
extern int itersPerEpisode;
extern int maxNumEpisodes;
extern int maxIters;
extern int itersInPerfMeasPeriod;
extern int printQ;
extern int codExperiment;
extern int numPrices;
extern int typeExplorationMechanism;
extern int DepthState0;
extern int DepthState;
extern int LengthStates;
extern int numStates;
extern int lengthStrategies;
extern int LengthFormatStatesPrint;
extern int LengthFormatActionPrint;
extern int LengthFormatTotExperimentsPrint;
extern int LengthFormatNumSessionsPrint;
extern int typePayoffInput;
extern int numAgents;
extern int numActions;
extern int numDemandParameters;
extern int numPeriods;
extern int numExplorationParameters;
extern int SwitchImpulseResponseToBR;
extern int SwitchImpulseResponseToNash;
extern int SwitchImpulseResponseToAll;
extern int SwitchEquilibriumCheck;
extern int SwitchQGapToMaximum;
extern std::vector<int> ParamsLearningTrajectory;
extern int SwitchDetailedAnalysis;

extern double delta;
extern double PerfMeasPeriodLength;
extern double meanNashProfit;
extern double meanCoopProfit;
extern double gammaSinghVives;

extern std::string ExperimentNumber;
extern std::string FileNameInfoExperiment;

extern std::vector<int> converged;
extern std::vector<std::vector<int>> indexStrategies;
extern std::vector<std::vector<int>> indexLastState;
extern std::vector<int> CycleLength;
extern std::vector<std::vector<int>> CycleStates;
extern std::vector<std::vector<std::vector<int>>> CyclePrices;
extern std::vector<std::vector<int>> indexActions;
extern std::vector<int> cStates;
extern std::vector<int> cActions;
extern std::vector<int> indexNashPrices;
extern std::vector<int> indexCoopPrices;

extern std::vector<double> timeToConvergence;
extern std::vector<std::vector<std::vector<double>>> CycleProfits;
extern std::vector<double> NashProfits;
extern std::vector<double> CoopProfits;
extern std::vector<std::vector<double>> maxValQ;
extern std::vector<double> NashPrices;
extern std::vector<double> CoopPrices;
extern std::vector<std::vector<double>> PI;
extern std::vector<std::vector<double>> PIQ;
extern std::vector<double> avgPI;
extern std::vector<double> avgPIQ;
extern std::vector<std::vector<double>> PG;
extern std::vector<std::vector<double>> PGQ;
extern std::vector<double> avgPG;
extern std::vector<double> avgPGQ;
extern std::vector<double> alpha;
extern std::vector<double> DiscountFactors;
extern std::vector<double> DemandParameters;
extern std::vector<double> MExpl;
extern std::vector<double> ExplorationParameters;
extern std::vector<double> NashMarketShares;
extern std::vector<double> CoopMarketShares;
extern std::vector<std::vector<double>> PricesGrids;
extern std::vector<std::vector<double>> parQInitialization;

extern std::vector<std::string> labelStates;
extern std::vector<std::string> QFileFolderName;
extern std::vector<char> typeQInitialization;

void readBatchVariables(int unitNumber);
void closeBatch();
void readExperimentVariables(int unitNumber);

}  // namespace globals
