@echo off
setlocal
rem ==========================================================================
rem  Mobilis: build the Windows program with Nuitka (compiled to machine code)
rem  and an installer with Inno Setup 6.
rem  Use: double-click this file, or run  packaging\build_windows.bat
rem  Needs: Python 3.10-3.12 (64-bit) on PATH. The first build downloads a
rem  C compiler and takes 10-30 minutes.
rem ==========================================================================
cd /d "%~dp0.."
echo Mobilis build folder: %CD%
echo.

where python >nul 2>nul
if errorlevel 1 goto nopython
python --version
python -c "import sys; sys.exit(0 if (3,10)<=sys.version_info[:2]<=(3,12) else 1)"
if errorlevel 1 echo [warning] Python 3.10-3.12 is recommended; with 3.13 install Visual Studio Build Tools first.

set VER=
for /f %%v in ('python -c "import mobility_tool as m; print(m.__version__)"') do set VER=%%v
if "%VER%"=="" set VER=1.27.0
echo Version: %VER%
echo.

echo [1/4] Build environment .build_venv ...
if not exist .build_venv\Scripts\activate.bat python -m venv .build_venv
if not exist .build_venv\Scripts\activate.bat goto novenv
call .build_venv\Scripts\activate.bat
python -m pip install -U pip
python -m pip install numpy scipy matplotlib scikit-image nuitka ordered-set zstandard
if errorlevel 1 goto nopip

echo.
echo [2/4] Nuitka compilation, this takes a while ...
if exist build_out rmdir /s /q build_out
python -m nuitka --standalone --assume-yes-for-downloads ^
  --enable-plugin=tk-inter ^
  --windows-console-mode=disable ^
  --windows-icon-from-ico=mobility_tool\assets\mobilis.ico ^
  --include-package=mobility_tool ^
  --include-module=skimage.measure._marching_cubes_lewiner ^
  --include-data-dir=mobility_tool\assets=mobility_tool\assets ^
  --nofollow-import-to=tests ^
  --nofollow-import-to=pytest ^
  --nofollow-import-to=IPython ^
  --company-name="Yarui Shi, Anyang Normal University" --product-name="Mobilis" ^
  --file-version=%VER%.0 --product-version=%VER%.0 ^
  --file-description="Mobilis - charge transport in organic semiconductors" ^
  --copyright="Yarui Shi, Anyang Normal University" ^
  --output-dir=build_out --output-filename=Mobilis.exe ^
  run_gui.py
if errorlevel 1 goto nonuitka

echo.
echo [3/4] Output folder dist\Mobilis ...
if exist dist\Mobilis rmdir /s /q dist\Mobilis
if not exist dist mkdir dist
move build_out\run_gui.dist dist\Mobilis >nul
if not exist dist\Mobilis\Mobilis.exe goto noexe
rem the release folder must not contain source code (Nuitka compiled it)
if exist dist\Mobilis\mobility_tool del /q /s dist\Mobilis\mobility_tool\*.py dist\Mobilis\mobility_tool\*.pyc >nul 2>nul

echo.
echo [4/4] Installer with Inno Setup 6 ...
set "ISCC="
if exist "%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe" set "ISCC=%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe"
if exist "%ProgramFiles%\Inno Setup 6\ISCC.exe" set "ISCC=%ProgramFiles%\Inno Setup 6\ISCC.exe"
if exist "%LOCALAPPDATA%\Programs\Inno Setup 6\ISCC.exe" set "ISCC=%LOCALAPPDATA%\Programs\Inno Setup 6\ISCC.exe"
if not defined ISCC goto noinno
"%ISCC%" /DMyAppVersion=%VER% packaging\Mobilis.iss
if errorlevel 1 goto noiscc
echo.
echo Done.
echo   Installer:          installer\Mobilis_Setup_%VER%.exe
echo   Portable program:   dist\Mobilis\Mobilis.exe
goto end

:nopython
echo [error] python was not found. Install Python 3.12 64-bit and tick "Add python.exe to PATH".
goto end
:novenv
echo [error] could not create the virtual environment .build_venv
goto end
:nopip
echo [error] installing the packages failed; check the network or the pip messages above.
goto end
:nonuitka
echo [error] Nuitka failed; see the messages above.
goto end
:noexe
echo [error] dist\Mobilis\Mobilis.exe was not produced.
goto end
:noinno
echo [note] Inno Setup 6 is not installed. The portable program is dist\Mobilis\Mobilis.exe.
echo        For an installer, install Inno Setup 6 from https://jrsoftware.org/isdl.php and run this file again.
goto end
:noiscc
echo [error] Inno Setup failed; see the messages above.
goto end

:end
echo.
pause
