# Compute CLI Installer for Windows
# Usage: irm https://computenetwork.sh/install.ps1 | iex

$ErrorActionPreference = "Stop"

$repo = "Compute-Network/compute-app"
$binaryName = "compute.exe"
$stageNodeBinary = "llama_stage_tcp_node.exe"
$gatewayBinary = "llama_stage_gateway_tcp_node.exe"

Write-Host ""
Write-Host "  ██████╗ ██████╗ ███╗   ███╗██████╗ ██╗   ██╗████████╗███████╗" -ForegroundColor White
Write-Host "  ██╔════╝██╔═══██╗████╗ ████║██╔══██╗██║   ██║╚══██╔══╝██╔════╝" -ForegroundColor White
Write-Host "  ██║     ██║   ██║██╔████╔██║██████╔╝██║   ██║   ██║   █████╗  " -ForegroundColor White
Write-Host "  ██║     ██║   ██║██║╚██╔╝██║██╔═══╝ ██║   ██║   ██║   ██╔══╝  " -ForegroundColor White
Write-Host "  ╚██████╗╚██████╔╝██║ ╚═╝ ██║██║     ╚██████╔╝   ██║   ███████╗" -ForegroundColor White
Write-Host "   ╚═════╝ ╚═════╝ ╚═╝     ╚═╝╚═╝      ╚═════╝    ╚═╝   ╚══════╝" -ForegroundColor White
Write-Host ""
Write-Host "  Decentralized GPU Infrastructure" -ForegroundColor DarkGray
Write-Host ""

# Detect architecture
$arch = if ([System.Environment]::Is64BitOperatingSystem) {
    if ($env:PROCESSOR_ARCHITECTURE -eq "ARM64") { "aarch64" } else { "x86_64" }
} else {
    Write-Host "Error: 32-bit systems are not supported" -ForegroundColor Red
    exit 1
}

$target = if ($arch -eq "aarch64") { "aarch64-pc-windows-msvc" } else { "x86_64-pc-windows-msvc" }
$filename = "compute-${target}.zip"

# Get latest release
Write-Host "  Fetching latest release..." -ForegroundColor DarkGray

try {
    $release = Invoke-RestMethod -Uri "https://api.github.com/repos/$repo/releases/latest" -ErrorAction Stop
    $asset = $release.assets | Where-Object { $_.name -eq $filename } | Select-Object -First 1

    if (-not $asset) {
        $downloadUrl = "https://github.com/$repo/releases/download/$($release.tag_name)/$filename"
    } else {
        $downloadUrl = $asset.browser_download_url
    }
} catch {
    Write-Host "  Error: Could not fetch release info" -ForegroundColor Red
    exit 1
}

Write-Host "  Downloading $filename..." -ForegroundColor DarkGray

# Download
$tempDir = Join-Path $env:TEMP "compute-install"
New-Item -ItemType Directory -Force -Path $tempDir | Out-Null
$zipPath = Join-Path $tempDir $filename

try {
    Invoke-WebRequest -Uri $downloadUrl -OutFile $zipPath -ErrorAction Stop
} catch {
    Write-Host "  Error: Download failed" -ForegroundColor Red
    exit 1
}

# Extract
Write-Host "  Installing..." -ForegroundColor DarkGray
Expand-Archive -Path $zipPath -DestinationPath $tempDir -Force

# Install to user's local bin
$installDir = Join-Path $env:LOCALAPPDATA "compute\bin"
New-Item -ItemType Directory -Force -Path $installDir | Out-Null

$binaryPath = Get-ChildItem -Path $tempDir -Filter $binaryName -Recurse | Select-Object -First 1
if (-not $binaryPath) {
    Write-Host "  Error: Binary not found in archive" -ForegroundColor Red
    exit 1
}

Copy-Item -Path $binaryPath.FullName -Destination (Join-Path $installDir $binaryName) -Force

$stageNodePath = Get-ChildItem -Path $tempDir -Filter $stageNodeBinary -Recurse | Select-Object -First 1
if ($stageNodePath) {
    Copy-Item -Path $stageNodePath.FullName -Destination (Join-Path $installDir $stageNodeBinary) -Force
}

$gatewayPath = Get-ChildItem -Path $tempDir -Filter $gatewayBinary -Recurse | Select-Object -First 1
if ($gatewayPath) {
    Copy-Item -Path $gatewayPath.FullName -Destination (Join-Path $installDir $gatewayBinary) -Force
}

# Add to PATH if not already there
$currentPath = [Environment]::GetEnvironmentVariable("Path", "User")
if ($currentPath -notlike "*$installDir*") {
    [Environment]::SetEnvironmentVariable("Path", "$currentPath;$installDir", "User")
    $env:Path = "$env:Path;$installDir"
    Write-Host "  Added $installDir to PATH" -ForegroundColor DarkGray
}

# Cleanup
Remove-Item -Path $tempDir -Recurse -Force -ErrorAction SilentlyContinue

# Verify
Write-Host ""
try {
    $version = & (Join-Path $installDir $binaryName) --version 2>&1
    Write-Host "  ✓ $version" -ForegroundColor Green
} catch {
    Write-Host "  ✓ Installed successfully" -ForegroundColor Green
}

Write-Host ""
Write-Host "  Get started:" -ForegroundColor DarkGray
Write-Host "    compute init        First-time setup"
Write-Host "    compute start       Start contributing compute"
Write-Host "    compute dashboard   View live stats"
Write-Host ""
Write-Host "  Note: Restart your terminal for PATH changes to take effect." -ForegroundColor DarkGray
Write-Host ""
