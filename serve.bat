@echo off
rem xraypoint webapp local server - run this, then open http://localhost:8180
cd /d %~dp0
echo Starting xraypoint webapp at http://localhost:8180 (Ctrl+C to stop)
start http://localhost:8180
"%USERPROFILE%\.conda\envs\xraypoint\python.exe" -m http.server 8180 2>nul || python -m http.server 8180 2>nul || py -m http.server 8180
