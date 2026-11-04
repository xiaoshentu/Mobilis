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
echo ---- 运行 G1_A.gjf ----
g16 G1_A.gjf G1_A.out
findstr /c:"Error termination" G1_A.out >nul && echo [警告] G1_A 出错，请查看 G1_A.out
if exist G1_A.chk formchk G1_A.chk G1_A.fchk
echo ---- 运行 G1_B.gjf ----
g16 G1_B.gjf G1_B.out
findstr /c:"Error termination" G1_B.out >nul && echo [警告] G1_B 出错，请查看 G1_B.out
if exist G1_B.chk formchk G1_B.chk G1_B.fchk
echo ---- 运行 G1_AB.gjf ----
g16 G1_AB.gjf G1_AB.out
findstr /c:"Error termination" G1_AB.out >nul && echo [警告] G1_AB 出错，请查看 G1_AB.out
if exist G1_AB.chk formchk G1_AB.chk G1_AB.fchk
echo ---- 运行 G2_A.gjf ----
g16 G2_A.gjf G2_A.out
findstr /c:"Error termination" G2_A.out >nul && echo [警告] G2_A 出错，请查看 G2_A.out
if exist G2_A.chk formchk G2_A.chk G2_A.fchk
echo ---- 运行 G2_B.gjf ----
g16 G2_B.gjf G2_B.out
findstr /c:"Error termination" G2_B.out >nul && echo [警告] G2_B 出错，请查看 G2_B.out
if exist G2_B.chk formchk G2_B.chk G2_B.fchk
echo ---- 运行 G2_AB.gjf ----
g16 G2_AB.gjf G2_AB.out
findstr /c:"Error termination" G2_AB.out >nul && echo [警告] G2_AB 出错，请查看 G2_AB.out
if exist G2_AB.chk formchk G2_AB.chk G2_AB.fchk
echo ---- 运行 G3_A.gjf ----
g16 G3_A.gjf G3_A.out
findstr /c:"Error termination" G3_A.out >nul && echo [警告] G3_A 出错，请查看 G3_A.out
if exist G3_A.chk formchk G3_A.chk G3_A.fchk
echo ---- 运行 G3_B.gjf ----
g16 G3_B.gjf G3_B.out
findstr /c:"Error termination" G3_B.out >nul && echo [警告] G3_B 出错，请查看 G3_B.out
if exist G3_B.chk formchk G3_B.chk G3_B.fchk
echo ---- 运行 G3_AB.gjf ----
g16 G3_AB.gjf G3_AB.out
findstr /c:"Error termination" G3_AB.out >nul && echo [警告] G3_AB 出错，请查看 G3_AB.out
if exist G3_AB.chk formchk G3_AB.chk G3_AB.fchk
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
