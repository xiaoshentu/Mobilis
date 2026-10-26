; Mobilis 迁跃 安装包 (Inno Setup 6)。由 build_windows.bat 调用，也可在 Inno Setup 中直接打开编译。
#ifndef MyAppVersion
  #define MyAppVersion "1.17.0"
#endif
#define MyAppName "Mobilis"
#define MyAppExe "Mobilis.exe"

[Setup]
AppId={{6E1A9C52-7B3D-4C8E-9A41-3F2B6D8E5C17}
AppName={#MyAppName}
AppVerName={#MyAppName} {#MyAppVersion}
AppVersion={#MyAppVersion}
AppPublisher=Yarui Shi, 安阳师范学院 (Anyang Normal University)
AppCopyright=Yarui Shi
DefaultDirName={autopf}\Mobilis
DefaultGroupName=Mobilis
DisableProgramGroupPage=yes
OutputDir=..\installer
OutputBaseFilename=Mobilis_Setup_{#MyAppVersion}
SetupIconFile=..\mobility_tool\assets\mobilis.ico
UninstallDisplayIcon={app}\{#MyAppExe}
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
PrivilegesRequiredOverridesAllowed=dialog

[Languages]
#ifexist AddBackslash(CompilerPath) + "Languages\ChineseSimplified.isl"
Name: "chinesesimp"; MessagesFile: "compiler:Languages\ChineseSimplified.isl"
#endif
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[Files]
Source: "..\dist\Mobilis\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\Mobilis 迁跃"; Filename: "{app}\{#MyAppExe}"
Name: "{group}\{cm:UninstallProgram,Mobilis}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\Mobilis 迁跃"; Filename: "{app}\{#MyAppExe}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExe}"; Description: "{cm:LaunchProgram,Mobilis}"; Flags: nowait postinstall skipifsilent
