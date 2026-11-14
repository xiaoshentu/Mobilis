@echo off
setlocal
rem Quick build with PyInstaller (for testing; for distribution use build_windows.bat)
cd /d "%~dp0.."
where python >nul 2>nul
if errorlevel 1 goto nopython
python -m pip install pyinstaller numpy scipy matplotlib scikit-image
if errorlevel 1 goto failed
pyinstaller packaging\Mobilis.spec --noconfirm
if errorlevel 1 goto failed
echo Done: dist\Mobilis\Mobilis.exe
goto end
:nopython
echo [error] python was not found. Install Python 3.12 64-bit and tick "Add python.exe to PATH".
goto end
:failed
echo [error] see the messages above.
:end
pause
