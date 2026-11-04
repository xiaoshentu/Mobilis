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
echo ---- 运行 anthracene_neutral.gjf ----
g16 anthracene_neutral.gjf anthracene_neutral.out
findstr /c:"Error termination" anthracene_neutral.out >nul && echo [警告] anthracene_neutral 出错，请查看 anthracene_neutral.out
if exist anthracene_neutral.chk formchk anthracene_neutral.chk anthracene_neutral.fchk
echo ---- 运行 anthracene_cation.gjf ----
g16 anthracene_cation.gjf anthracene_cation.out
findstr /c:"Error termination" anthracene_cation.out >nul && echo [警告] anthracene_cation 出错，请查看 anthracene_cation.out
if exist anthracene_cation.chk formchk anthracene_cation.chk anthracene_cation.fchk
echo ---- 运行 anthracene_anion.gjf ----
g16 anthracene_anion.gjf anthracene_anion.out
findstr /c:"Error termination" anthracene_anion.out >nul && echo [警告] anthracene_anion 出错，请查看 anthracene_anion.out
if exist anthracene_anion.chk formchk anthracene_anion.chk anthracene_anion.fchk
echo 全部任务结束
pause
