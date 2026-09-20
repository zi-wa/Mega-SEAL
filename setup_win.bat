@echo off
REM Everything lands in the repo-local venv seal_env, never in a global site-packages.
setlocal
REM Double-click and typed runs must resolve the relative paths below against the repo root.
cd /d "%~dp0"

echo [1/5] Checking virtual environment seal_env ...
if not exist seal_env (
    echo Creating seal_env ...
    python -m venv seal_env
    if errorlevel 1 (
        echo FAILED: could not create seal_env. Is Python on PATH?
        goto :fail
    )
) else (
    echo seal_env already exists, skipping.
)

echo [2/5] Upgrading pip ...
seal_env\Scripts\python.exe -m pip install --upgrade pip
if errorlevel 1 (
    echo FAILED: could not upgrade pip.
    goto :fail
)

echo [3/5] Installing torch from the official CUDA 13.0 index ...
seal_env\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu130
if errorlevel 1 (
    echo FAILED: could not install torch. Check the network and the NVIDIA driver version.
    goto :fail
)

echo [4/5] Installing requirements-win.txt ...
seal_env\Scripts\python.exe -m pip install -r requirements-win.txt
if errorlevel 1 (
    echo FAILED: could not install requirements-win.txt.
    goto :fail
)

echo [5/5] Checking config.py ...
if not exist config.py (
    copy config.example.py config.py
    if errorlevel 1 (
        echo FAILED: could not copy config.example.py to config.py.
        goto :fail
    )
    echo Created config.py. Open it and put your OpenAI API key in it before running anything.
) else (
    echo config.py already exists, skipping.
)

echo.
echo Setup finished. Next step: run run_all.bat
endlocal
exit /b 0

:fail
echo Setup stopped. Fix the problem above and run setup_win.bat again.
endlocal
exit /b 1
