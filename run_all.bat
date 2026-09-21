@echo off
REM Every stage skips work it already finished, so this file is safe to re-run after any stop.
setlocal
REM Double-click and typed runs must resolve the relative paths below against the repo root.
cd /d "%~dp0"

set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set TOKENIZERS_PARALLELISM=false
set HF_HUB_DISABLE_SYMLINKS_WARNING=1
REM Deterministic cuBLAS reductions; cuBLAS reads this only before the CUDA context exists.
set CUBLAS_WORKSPACE_CONFIG=:4096:8

set PYTHON_BIN=seal_env\Scripts\python.exe

if not exist config.py (
    echo config.py not found. Run setup_win.bat first.
    goto :fail
)

REM The key lives only in the environment; stop now rather than hours into the run.
if "%OPENAI_API_KEY%"=="" (
    echo OPENAI_API_KEY is not set. Run: setx OPENAI_API_KEY "sk-..." then open a new window.
    goto :fail
)

echo ============================================================
echo Stage 1/5: dev pilot - measures TTT seconds, projects runtime
echo ============================================================
%PYTHON_BIN% -m general-knowledge.src.qagen.run_pilot
if errorlevel 1 (
    echo FAILED at stage 1: run_pilot.
    goto :fail
)
echo.
echo The projected total runtime is printed above. The full run takes days.
echo Closing this window is safe - running run_all.bat again resumes where it stopped.
echo.

echo ============================================================
echo Stage 2/5: judge agreement check
echo ============================================================
%PYTHON_BIN% -m general-knowledge.src.qagen.validate_judge
if errorlevel 1 (
    echo FAILED at stage 2: validate_judge.
    goto :fail
)

echo ============================================================
echo Stage 3/5: outer loop - trains the question generator
echo ============================================================
%PYTHON_BIN% -m general-knowledge.src.qagen.run_outer
if errorlevel 1 (
    echo FAILED at stage 3: run_outer.
    goto :fail
)

echo ============================================================
echo Stage 4/5: SE-RL and held-out evaluation for every condition
echo ============================================================
%PYTHON_BIN% -m general-knowledge.src.qagen.run_eval
if errorlevel 1 (
    echo FAILED at stage 4: run_eval.
    goto :fail
)

echo ============================================================
echo Stage 5/5: report
echo ============================================================
%PYTHON_BIN% -m general-knowledge.src.qagen.report
if errorlevel 1 (
    echo FAILED at stage 5: report.
    goto :fail
)

echo.
echo All five stages finished. The summary is in summary.md
endlocal
exit /b 0

:fail
echo Run stopped. Fix the problem above and run run_all.bat again to resume.
endlocal
exit /b 1
