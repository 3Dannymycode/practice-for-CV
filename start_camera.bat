@echo off
cd /d %~dp0
echo Closing old BlazeFace processes...
powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { $_.CommandLine -match 'blazeface_detect' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }"
timeout /t 1 /nobreak >nul
echo Starting BlazeFace camera...
python -u blazeface_detect.py --camera 0
echo.
echo Exit code %ERRORLEVEL%
pause
