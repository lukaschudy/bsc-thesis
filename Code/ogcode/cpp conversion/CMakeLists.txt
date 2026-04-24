cmake_minimum_required(VERSION 3.16)
project(ogcode_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(ogcode_cpp
    src/main.cpp
    src/fortran_runtime.cpp
    src/generic_routines.cpp
    src/globals.cpp
    src/QL_routines.cpp
    src/PI_routines.cpp
    src/LearningSimulation.cpp
    src/ConvergenceResults.cpp
    src/EquilibriumCheck.cpp
    src/QGapToMaximum.cpp
    src/ImpulseResponse.cpp
    src/LearningTrajectory.cpp
    src/DetailedAnalysis.cpp
)

target_include_directories(ogcode_cpp PRIVATE include)

# --- Observation-latency friction -------------------------------------------
# Shares all source files with the baseline except main.cpp and
# LearningSimulation.cpp, which are replaced by their latency equivalents.
add_executable(ogcode_latency
    src/main_latency.cpp
    src/fortran_runtime.cpp
    src/generic_routines.cpp
    src/globals.cpp
    src/QL_routines.cpp
    src/PI_routines.cpp
    src/LearningSimulation.cpp
    src/LearningSimulationLatency.cpp
    src/ConvergenceResults.cpp
    src/EquilibriumCheck.cpp
    src/QGapToMaximum.cpp
    src/ImpulseResponse.cpp
    src/LearningTrajectory.cpp
    src/DetailedAnalysis.cpp
)

target_include_directories(ogcode_latency PRIVATE include)
# ----------------------------------------------------------------------------

# --- Noisy-monitoring friction ----------------------------------------------
# Shares all source files with the baseline except main.cpp, which is
# replaced by main_noise.cpp. LearningSimulation.cpp is still linked in
# because LearningTrajectory.cpp references LearningSimulation::computePPrime.
add_executable(ogcode_noise
    src/main_noise.cpp
    src/fortran_runtime.cpp
    src/generic_routines.cpp
    src/globals.cpp
    src/QL_routines.cpp
    src/PI_routines.cpp
    src/LearningSimulation.cpp
    src/LearningSimulationNoise.cpp
    src/ConvergenceResults.cpp
    src/EquilibriumCheck.cpp
    src/QGapToMaximum.cpp
    src/ImpulseResponse.cpp
    src/LearningTrajectory.cpp
    src/DetailedAnalysis.cpp
)

target_include_directories(ogcode_noise PRIVATE include)
# ----------------------------------------------------------------------------

# --- Asynchronous-actions friction ------------------------------------------
# Shares all source files with the baseline except main.cpp, which is
# replaced by main_async.cpp. LearningSimulation.cpp is still linked
# because LearningTrajectory.cpp and LearningSimulationAsync.cpp both
# call LearningSimulation::computePPrime.
add_executable(ogcode_async
    src/main_async.cpp
    src/fortran_runtime.cpp
    src/generic_routines.cpp
    src/globals.cpp
    src/QL_routines.cpp
    src/PI_routines.cpp
    src/LearningSimulation.cpp
    src/LearningSimulationAsync.cpp
    src/ConvergenceResults.cpp
    src/EquilibriumCheck.cpp
    src/QGapToMaximum.cpp
    src/ImpulseResponse.cpp
    src/LearningTrajectory.cpp
    src/DetailedAnalysis.cpp
)

target_include_directories(ogcode_async PRIVATE include)
# ----------------------------------------------------------------------------

# --- Combined-frictions 2x2x2 factorial -------------------------------------
# Unified learner that applies latency + noise + async together.
add_executable(ogcode_combined
    src/main_combined.cpp
    src/fortran_runtime.cpp
    src/generic_routines.cpp
    src/globals.cpp
    src/QL_routines.cpp
    src/PI_routines.cpp
    src/LearningSimulation.cpp
    src/LearningSimulationCombined.cpp
    src/ConvergenceResults.cpp
    src/EquilibriumCheck.cpp
    src/QGapToMaximum.cpp
    src/ImpulseResponse.cpp
    src/LearningTrajectory.cpp
    src/DetailedAnalysis.cpp
)

target_include_directories(ogcode_combined PRIVATE include)
# ----------------------------------------------------------------------------

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(ogcode_cpp      PRIVATE OpenMP::OpenMP_CXX)
    target_link_libraries(ogcode_latency  PRIVATE OpenMP::OpenMP_CXX)
    target_link_libraries(ogcode_noise    PRIVATE OpenMP::OpenMP_CXX)
    target_link_libraries(ogcode_async    PRIVATE OpenMP::OpenMP_CXX)
    target_link_libraries(ogcode_combined PRIVATE OpenMP::OpenMP_CXX)
endif()
