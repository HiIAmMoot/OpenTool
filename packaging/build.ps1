param(
    [string]$Python = 'python.exe',
    [string]$VCRedist,
    [switch]$CheckRuntimeOnly
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$Root = Split-Path $PSScriptRoot -Parent
$Bundle = Join-Path $Root 'build\dist\OpenTool'

function Assert-Runtime {
    $Names = @('vcruntime140.dll', 'vcruntime140_1.dll', 'msvcp140.dll',
        'msvcp140_1.dll', 'msvcp140_2.dll', 'msvcp140_atomic_wait.dll',
        'msvcp140_codecvt_ids.dll', 'concrt140.dll')
    if (-not (Test-Path $Bundle)) { throw "Bundle not found: $Bundle" }
    foreach ($File in Get-ChildItem $Bundle -Recurse -File -Filter '*.dll') {
        if ($Names -contains $File.Name) {
            $Info = $File.VersionInfo
            Write-Output "$($File.FullName): $($Info.FileVersion)"
            # OIDN needs 14.4x; python.org and wheel copies can be older.
            if ($Info.FileMajorPart -lt 14 -or
                ($Info.FileMajorPart -eq 14 -and $Info.FileMinorPart -lt 40)) {
                throw "C++ runtime must be >= 14.40: $($File.FullName) ($($Info.FileVersion))"
            }
        } elseif ($File.Name -match '^(vcruntime140|msvcp140|concrt140).*\.dll$') {
            # Hashed wheel-private names do not replace the named runtime.
            Write-Output "Private runtime (informational): $($File.FullName): $($File.VersionInfo.FileVersion)"
        }
    }
}

if ($CheckRuntimeOnly) {
    Assert-Runtime
    exit 0
}

Push-Location $Root
try {
    Write-Output '1. Check base Python'
    & $Python -c "import sys, pathlib; print(sys.version); print('base:', sys.base_prefix); assert sys.version_info[:2] == (3, 11), 'Use Python 3.11'; assert not any(s in sys.version.lower() for s in ('conda', 'anaconda')) and not (pathlib.Path(sys.base_prefix) / 'conda-meta').exists(), 'Anaconda is not supported'; assert sys.maxsize > 2**32, 'Use 64-bit Python'"
    if ($LASTEXITCODE -ne 0) { throw 'Base Python check failed' }

    Write-Output '2. Recreate clean build venv'
    if (Test-Path 'build\venv') { Remove-Item 'build\venv' -Recurse -Force }
    & $Python -m venv 'build\venv'
    if ($LASTEXITCODE -ne 0) { throw 'venv failed' }
    $VenvPython = Join-Path $Root 'build\venv\Scripts\python.exe'
    & $VenvPython -m pip install --only-binary=:all: -r requirements.txt -r packaging\requirements-build.txt
    if ($LASTEXITCODE -ne 0) { throw 'pip install failed' }

    Write-Output '3. Rebuild xatlas'
    if (Test-Path 'lightmapper\vendor\xatlas\build') {
        Remove-Item 'lightmapper\vendor\xatlas\build' -Recurse -Force
    }
    & $VenvPython -c 'from lightmapper import xatlas_c; print(xatlas_c.build())'
    if ($LASTEXITCODE -ne 0) { throw 'xatlas build failed' }

    Write-Output '4. Freeze both entry points'
    & $VenvPython -m PyInstaller --noconfirm --clean --distpath build\dist --workpath build\pyi packaging\opentool.spec
    if ($LASTEXITCODE -ne 0) { throw 'PyInstaller failed' }

    Write-Output '5. Install and check app-local C++ runtime'
    if (-not $VCRedist) {
        $Candidates = @(Get-Item 'C:\Program Files\Microsoft Visual Studio\2022\*\VC\Redist\MSVC\14.4*\x64\Microsoft.VC143.CRT' |
            Sort-Object { [version]$_.Parent.Parent.Name } -Descending)
        if ($Candidates.Count -eq 0) { throw 'No Visual Studio 14.4x x64 CRT found; pass -VCRedist' }
        $VCRedist = $Candidates[0].FullName
    }
    $Dlls = @(Get-ChildItem $VCRedist -File -Filter '*.dll' | Where-Object Name -ne 'vccorlib140.dll')
    if ($Dlls.Count -eq 0) { throw "No runtime DLLs in $VCRedist" }
    $Dlls | Copy-Item -Destination (Join-Path $Bundle '_internal') -Force
    Assert-Runtime

    Write-Output '6. Collect licenses'
    Copy-Item LICENSE, THIRD_PARTY_NOTICES.md -Destination $Bundle -Force
    & $VenvPython packaging\collect_licenses.py (Join-Path $Bundle 'licenses')
    if ($LASTEXITCODE -ne 0) { throw 'License collection failed' }

    Write-Output '7. Frozen self-test'
    & (Join-Path $Bundle 'OpenTool-cli.exe') check
    if ($LASTEXITCODE -ne 0) { throw 'Frozen self-test failed' }

    Write-Output '8. Zip app folder'
    $Describe = & git.exe describe --tags --always
    if ($LASTEXITCODE -ne 0) { throw 'git describe failed' }
    New-Item dist -ItemType Directory -Force | Out-Null
    $Zip = Join-Path $Root "dist\OpenTool-$Describe-windows-x64.zip"
    if (Test-Path $Zip) { Remove-Item $Zip -Force }
    # Full path selects Windows bsdtar, not Git's tar (and avoids PS 5.1 zip backslashes).
    & "$env:SystemRoot\System32\tar.exe" -a -c -f $Zip -C build\dist OpenTool
    if ($LASTEXITCODE -ne 0) { throw 'Zip failed' }
    $FolderBytes = (Get-ChildItem $Bundle -Recurse -File | Measure-Object Length -Sum).Sum
    Write-Output "Folder: $Bundle ($FolderBytes bytes)"
    Write-Output "Zip: $Zip ($((Get-Item $Zip).Length) bytes)"
} finally {
    Pop-Location
}
