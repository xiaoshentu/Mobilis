@echo off
setlocal enabledelayedexpansion
rem ==========================================================================
rem  Mobilis 迁跃  一键构建: Nuitka 编译 (代码编译为机器码) + Inno Setup 安装包
rem  用法: 双击本文件，或在命令行运行 packaging\build_windows.bat
rem  需要: Python 3.10-3.12 (64 位)；首次构建会自动下载 C 编译器 (约 10-30 分钟)
rem ==========================================================================
cd /d %~dp0..
set ROOT=%CD%
set VER=1.16.0

where python >nul 2>nul || (echo [错误] 找不到 python，请先安装 Python 3.11 或 3.12 (64 位) 并勾选 Add to PATH & pause & exit /b 1)
python -c "import sys; sys.exit(0 if (3,10)<=sys.version_info[:2]<=(3,12) else 1)" || (
  echo [警告] 推荐 Python 3.10-3.12：Nuitka 自动下载的 MinGW 编译器不支持更新的版本；
  echo        如使用 3.13，请先安装 Visual Studio Build Tools ^(C++^)。
)

echo [1/4] 准备独立的构建环境 .build_venv ...
if not exist .build_venv python -m venv .build_venv
call .build_venv\Scripts\activate.bat || (echo [错误] 无法创建虚拟环境 & pause & exit /b 1)
python -m pip install -U pip >nul
python -m pip install numpy scipy matplotlib scikit-image nuitka ordered-set zstandard || (echo [错误] 安装依赖失败，请检查网络 & pause & exit /b 1)

echo [2/4] Nuitka 编译 (时间较长，请耐心等待) ...
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
  --file-description="Mobilis Qianyue - organic semiconductor charge transport (Yarui Shi, Anyang Normal University)" --copyright="Yarui Shi, Anyang Normal University" ^
  --output-dir=build_out --output-filename=Mobilis.exe ^
  run_gui.py
if errorlevel 1 (echo [错误] Nuitka 编译失败，见上方信息 & pause & exit /b 1)

echo [3/4] 整理输出到 dist\Mobilis ...
if exist dist\Mobilis rmdir /s /q dist\Mobilis
if not exist dist mkdir dist
move build_out\run_gui.dist dist\Mobilis >nul
if not exist dist\Mobilis\Mobilis.exe (echo [错误] 没有找到 dist\Mobilis\Mobilis.exe & pause & exit /b 1)
rem 发布目录中不应包含本软件的源代码 (Nuitka 已编译为机器码)
if exist dist\Mobilis\mobility_tool del /q /s dist\Mobilis\mobility_tool\*.py dist\Mobilis\mobility_tool\*.pyc >nul 2>nul
dir /s /b dist\Mobilis\mobility_tool\*.py 2>nul && echo [警告] 发布目录中仍有 .py 文件

echo [4/4] 生成安装包 (Inno Setup) ...
set ISCC=
for %%p in ("%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe" "%ProgramFiles%\Inno Setup 6\ISCC.exe" "%LOCALAPPDATA%\Programs\Inno Setup 6\ISCC.exe") do (
  if exist %%p set ISCC=%%p
)
if not defined ISCC (
  where iscc >nul 2>nul && set ISCC=iscc
)
if not defined ISCC (
  echo [提示] 未安装 Inno Setup 6：免安装版程序已在 dist\Mobilis\Mobilis.exe，可直接运行或压缩分发。
  echo        如需安装包，请安装 Inno Setup 6 ^(https://jrsoftware.org/isdl.php^) 后重新运行本脚本。
  pause & exit /b 0
)
%ISCC% /DMyAppVersion=%VER% packaging\Mobilis.iss || (echo [错误] Inno Setup 失败 & pause & exit /b 1)
echo.
echo 完成！安装包: installer\Mobilis_Setup_%VER%.exe
echo 免安装版:   dist\Mobilis\Mobilis.exe
pause
