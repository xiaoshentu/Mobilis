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
echo ---- 运行 G4_A.gjf ----
g16 G4_A.gjf G4_A.out
findstr /c:"Error termination" G4_A.out >nul && echo [警告] G4_A 出错，请查看 G4_A.out
if exist G4_A.chk formchk G4_A.chk G4_A.fchk
echo ---- 运行 G4_B.gjf ----
g16 G4_B.gjf G4_B.out
findstr /c:"Error termination" G4_B.out >nul && echo [警告] G4_B 出错，请查看 G4_B.out
if exist G4_B.chk formchk G4_B.chk G4_B.fchk
echo ---- 运行 G4_AB.gjf ----
g16 G4_AB.gjf G4_AB.out
findstr /c:"Error termination" G4_AB.out >nul && echo [警告] G4_AB 出错，请查看 G4_AB.out
if exist G4_AB.chk formchk G4_AB.chk G4_AB.fchk
echo 全部任务结束
pause
