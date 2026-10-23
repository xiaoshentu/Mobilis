@echo off
rem PyInstaller 快速打包 (保护较弱，仅用于自用 / 测试；正式发布请用 build_windows.bat)
cd /d %~dp0..
python -m pip install pyinstaller numpy scipy matplotlib || (pause & exit /b 1)
pyinstaller packaging\Mobilis.spec --noconfirm || (pause & exit /b 1)
echo 完成: dist\Mobilis\Mobilis.exe
pause
