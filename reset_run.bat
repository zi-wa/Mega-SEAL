@echo off
REM Deletes one run's results, adapters and log so run_all.bat starts from zero.
setlocal
cd /d "%~dp0"

if not exist config.py (
    echo config.py not found. Nothing to reset.
    goto :done
)
for /f "delims=" %%r in ('seal_env\Scripts\python.exe -c "import config; print(config.RUN_NAME)"') do set RUN_NAME=%%r

echo This deletes everything run %RUN_NAME% produced:
echo     general-knowledge\results\qagen\%RUN_NAME%
echo     models\qagen\%RUN_NAME%
echo     logs\run_all.log
echo Close this window now to cancel, or
pause

if exist "general-knowledge\results\qagen\%RUN_NAME%" rmdir /s /q "general-knowledge\results\qagen\%RUN_NAME%"
if exist "models\qagen\%RUN_NAME%" rmdir /s /q "models\qagen\%RUN_NAME%"
if exist "logs\run_all.log" del /q "logs\run_all.log"
echo Reset done. Run run_all.bat to start from the beginning.

:done
pause
endlocal
