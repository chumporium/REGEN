; Inno Setup script for REGEN-Setup.exe
; Run build.ps1 first (it creates dist\REGEN).
; The AppId is the same as in the earlier DOE Studio versions, so this installer replaces them.

#define AppName "REGEN"
#define AppVersion "7.0.0"
#define AppExe "REGEN.exe"

[Setup]
AppId={{6B1F3C2A-8D4E-4F7A-9C21-5E0A7B3D9F11}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher=Erdiyanto Munandar and Nasruddin, Universitas Indonesia
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
OutputDir=installer_output
OutputBaseFilename=REGEN-Setup-{#AppVersion}
SetupIconFile=assets\icon.ico
UninstallDisplayIcon={app}\{#AppExe}
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
PrivilegesRequiredOverridesAllowed=dialog
ChangesAssociations=yes

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"
Name: "assoc"; Description: "Open .doe files with {#AppName}"; GroupDescription: "File association:"

[Files]
Source: "dist\REGEN\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[InstallDelete]
; leftovers of the earlier versions named DOE Studio
Type: files; Name: "{app}\DOE Studio.exe"
Type: filesandordirs; Name: "{autoprograms}\DOE Studio"
Type: files; Name: "{autoprograms}\DOE Studio.lnk"
Type: files; Name: "{autodesktop}\DOE Studio.lnk"

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\{#AppExe}"
Name: "{group}\Uninstall {#AppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExe}"; Tasks: desktopicon

[Registry]
Root: HKA; Subkey: "Software\Classes\.doe"; ValueType: string; ValueName: ""; ValueData: "DOEStudio.Project"; Flags: uninsdeletevalue; Tasks: assoc
Root: HKA; Subkey: "Software\Classes\DOEStudio.Project"; ValueType: string; ValueName: ""; ValueData: "{#AppName} Project"; Flags: uninsdeletekey; Tasks: assoc
Root: HKA; Subkey: "Software\Classes\DOEStudio.Project\DefaultIcon"; ValueType: string; ValueName: ""; ValueData: "{app}\{#AppExe},0"; Tasks: assoc
Root: HKA; Subkey: "Software\Classes\DOEStudio.Project\shell\open\command"; ValueType: string; ValueName: ""; ValueData: """{app}\{#AppExe}"" ""%1"""; Tasks: assoc

[Run]
Filename: "{app}\{#AppExe}"; Description: "{cm:LaunchProgram,{#AppName}}"; Flags: nowait postinstall skipifsilent
