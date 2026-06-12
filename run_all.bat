@echo off
set PYTHONPATH=.
echo Running student-por...
py scripts/run_pipeline.py --dataset student-por --n-trials 50
echo.
echo Running student-mat...
py scripts/run_pipeline.py --dataset student-mat --n-trials 30
echo.
echo Running xapi...
py scripts/run_pipeline.py --dataset xapi --n-trials 30
echo.
echo ALL DONE!
