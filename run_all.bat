@echo off
set PYTHONPATH=.
echo Running student-por...
py scripts/run_pipeline.py --dataset student-por --n-trials 50
if errorlevel 1 exit /b %errorlevel%
echo.
echo Running student-mat...
py scripts/run_pipeline.py --dataset student-mat --n-trials 50
if errorlevel 1 exit /b %errorlevel%
echo.
echo Running xapi...
py scripts/run_pipeline.py --dataset xapi --n-trials 150
if errorlevel 1 exit /b %errorlevel%
echo.
echo ALL DONE!
