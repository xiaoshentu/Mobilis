@echo off
setlocal
rem ===== 如果高斯不在下面的目录，请修改 G16DIR =====
if not defined G16DIR set G16DIR=C:\G16W
cd /d %~dp0
set GAUSS_EXEDIR=%G16DIR%
set GAUSS_SCRDIR=%~dp0scratch
if not exist "%GAUSS_SCRDIR%" mkdir "%GAUSS_SCRDIR%"
set PATH=%G16DIR%;%PATH%
where g16 >nul 2>nul || (echo [错误] 找不到 g16.exe，请修改本文件开头的 G16DIR & pause & exit /b 1)
echo 当前目录: %CD%
echo 注意: 路径中如含中文或空格，Windows 版高斯可能无法运行，请把文件夹放到如 D:\calc\ 这样的路径。
echo ---- 运行 G5_A.gjf ----
g16 G5_A.gjf G5_A.out
findstr /c:"Error termination" G5_A.out >nul && echo [警告] G5_A 出错，请查看 G5_A.out
if exist G5_A.chk formchk G5_A.chk G5_A.fchk
echo ---- 运行 G5_B.gjf ----
g16 G5_B.gjf G5_B.out
findstr /c:"Error termination" G5_B.out >nul && echo [警告] G5_B 出错，请查看 G5_B.out
if exist G5_B.chk formchk G5_B.chk G5_B.fchk
echo ---- 运行 G5_AB.gjf ----
g16 G5_AB.gjf G5_AB.out
findstr /c:"Error termination" G5_AB.out >nul && echo [警告] G5_AB 出错，请查看 G5_AB.out
if exist G5_AB.chk formchk G5_AB.chk G5_AB.fchk
echo 全部任务结束
pause
