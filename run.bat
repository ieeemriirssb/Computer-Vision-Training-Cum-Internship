@echo off
pushd "%~dp0"
where py >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    py main.py
) else (
    python main.py
)
popd
