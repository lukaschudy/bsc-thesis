@echo off
REM Direct g++ build for the friction executables (latency and noise).
REM CMake is not installed on this machine; this script replaces
REM `cmake --build .` for the ogcode_latency and ogcode_noise targets.
REM
REM Usage (from "Code\ogcode\cpp conversion"):
REM     build_frictions.bat
REM
REM Outputs:
REM     ogcode_latency.exe
REM     ogcode_noise.exe

setlocal
set GXX=C:\msys64\ucrt64\bin\g++.exe
set FLAGS=-std=c++17 -O2 -fopenmp
set INC=-Iinclude

set COMMON_SRC=^
  src\fortran_runtime.cpp ^
  src\generic_routines.cpp ^
  src\globals.cpp ^
  src\QL_routines.cpp ^
  src\PI_routines.cpp ^
  src\LearningSimulation.cpp ^
  src\ConvergenceResults.cpp ^
  src\EquilibriumCheck.cpp ^
  src\QGapToMaximum.cpp ^
  src\ImpulseResponse.cpp ^
  src\LearningTrajectory.cpp ^
  src\DetailedAnalysis.cpp

echo ============================================================
echo Building ogcode_latency.exe ...
echo ============================================================
"%GXX%" %FLAGS% %INC% ^
  src\main_latency.cpp ^
  src\LearningSimulationLatency.cpp ^
  %COMMON_SRC% ^
  -o ogcode_latency.exe
if errorlevel 1 (
    echo BUILD FAILED: ogcode_latency
    exit /b 1
)

echo.
echo ============================================================
echo Building ogcode_noise.exe ...
echo ============================================================
"%GXX%" %FLAGS% %INC% ^
  src\main_noise.cpp ^
  src\LearningSimulationNoise.cpp ^
  %COMMON_SRC% ^
  -o ogcode_noise.exe
if errorlevel 1 (
    echo BUILD FAILED: ogcode_noise
    exit /b 1
)

echo.
echo ============================================================
echo Building ogcode_async.exe ...
echo ============================================================
"%GXX%" %FLAGS% %INC% ^
  src\main_async.cpp ^
  src\LearningSimulationAsync.cpp ^
  %COMMON_SRC% ^
  -o ogcode_async.exe
if errorlevel 1 (
    echo BUILD FAILED: ogcode_async
    exit /b 1
)

echo.
echo ============================================================
echo Building ogcode_combined.exe ...
echo ============================================================
"%GXX%" %FLAGS% %INC% ^
  src\main_combined.cpp ^
  src\LearningSimulationCombined.cpp ^
  %COMMON_SRC% ^
  -o ogcode_combined.exe
if errorlevel 1 (
    echo BUILD FAILED: ogcode_combined
    exit /b 1
)

echo.
echo ============================================================
echo Build succeeded. Binaries written to:
echo     ogcode_latency.exe
echo     ogcode_noise.exe
echo     ogcode_async.exe
echo     ogcode_combined.exe
echo ============================================================
endlocal
