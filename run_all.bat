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

REM Every stage also appends its output and any traceback here, so errors stay readable.
if not exist logs mkdir logs
set QAGEN_LOG=%~dp0logs\run_all.log
echo Log file: logs\run_all.log

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
echo Stage 1/5: run directory - judge cache carried over, code snapshot
echo ============================================================
%PYTHON_BIN% -m general-knowledge.src.qagen.prepare_run
if %errorlevel% neq 0 (
    echo FAILED at stage 1: prepare_run, exit code %errorlevel%.
    goto :fail
)

echo ============================================================
echo Stage 2/5: question bank, SE-RL and held-out evaluation for every condition
echo ============================================================
echo The full run takes about two days. Closing this window is safe - running
echo run_all.bat again resumes where it stopped.
%PYTHON_BIN% -m general-knowledge.src.qagen.run_eval
if %errorlevel% neq 0 (
    echo FAILED at stage 2: run_eval, exit code %errorlevel%.
    goto :fail
)

echo ============================================================
echo Stage 3/5: summary
echo ============================================================
%PYTHON_BIN% -m general-knowledge.src.qagen.report
if %errorlevel% neq 0 (
    echo FAILED at stage 3: report, exit code %errorlevel%.
    goto :fail
)

echo ============================================================
echo Stage 4/5: figures
echo ============================================================
%PYTHON_BIN% -m general-knowledge.src.qagen.figures
if %errorlevel% neq 0 (
    echo FAILED at stage 4: figures, exit code %errorlevel%.
    goto :fail
)

echo ============================================================
echo Stage 5/5: Korean report
echo ============================================================
%PYTHON_BIN% -m general-knowledge.src.qagen.report_ko
if %errorlevel% neq 0 (
    echo FAILED at stage 5: report_ko, exit code %errorlevel%.
    goto :fail
)

echo.
echo All five stages finished. Results: summary.md, report_ko.md, figures\ in the run directory.
echo Log file: logs\run_all.log
pause
endlocal
exit /b 0

:fail
echo Run stopped. Fix the problem above and run run_all.bat again to resume.
echo The full output, including the error, is in logs\run_all.log
pause
endlocal
exit /b 1
