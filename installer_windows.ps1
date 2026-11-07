# Installation de Gabarit sous Windows (PowerShell).
# Usage, depuis le dossier du projet :
#   powershell -ExecutionPolicy Bypass -File installer_windows.ps1
$ErrorActionPreference = "Stop"

Write-Host "1/4 Verification de Python..."
$py = Get-Command py -ErrorAction SilentlyContinue
if (-not $py) { $py = Get-Command python -ErrorAction SilentlyContinue }
if (-not $py) {
    Write-Host "Python introuvable. Installez Python 3.11 ou 3.12 depuis python.org (cocher 'Add to PATH'), puis relancez." -ForegroundColor Red
    exit 1
}

Write-Host "2/4 Creation de l'environnement virtuel .venv ..."
& $py.Source -m venv .venv
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt

Write-Host "3/4 Recherche de CalculiX (ccx.exe)..."
$ccx = Get-Command ccx -ErrorAction SilentlyContinue
if ($ccx) { $ccxPath = $ccx.Source }
else {
    # FreeCAD embarque ccx.exe dans son dossier bin
    $ccxPath = Get-ChildItem -Path "C:\Program Files\FreeCAD*", "$env:LOCALAPPDATA\Programs\FreeCAD*" -Recurse -Filter "ccx.exe" -ErrorAction SilentlyContinue |
               Select-Object -First 1 -ExpandProperty FullName
}
if (-not $ccxPath) {
    Write-Host "CalculiX introuvable." -ForegroundColor Yellow
    Write-Host "Le plus simple : installer FreeCAD (gratuit), qui contient ccx.exe :"
    Write-Host "    winget install FreeCAD.FreeCAD"
    Write-Host "ou depuis https://www.freecad.org/downloads.php, puis relancer ce script."
    exit 1
}
[Environment]::SetEnvironmentVariable("GABARIT_CCX", $ccxPath, "User")
$env:GABARIT_CCX = $ccxPath
Write-Host "CalculiX trouve : $ccxPath"

Write-Host "4/4 Validation du solveur..."
& .\.venv\Scripts\python.exe tests\valider_solveur.py
