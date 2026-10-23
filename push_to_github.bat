@echo off
cd /d "%~dp0"
git push -u origin main
git push origin v1.16.0
pause
