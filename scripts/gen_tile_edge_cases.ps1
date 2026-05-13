# =============================================================================
# gen_tile_edge_cases.ps1 - Generate P2 tile edge case data only
#
# This script does not compile or run RTL. P2.1.3 will add the simulation
# wrapper. Generated cases are CPU/testbench im2col + tile-packed A/W streams.
# =============================================================================

param(
    [ValidateSet("4x4", "8x8", "16x16", "8x32", "all")]
    [string]$Shape = "4x4",
    [ValidateSet("OS", "WS", "all")]
    [string]$Mode = "OS",
    [ValidateSet(0, 1, 2, 4)]
    [int]$Lanes = 1,
    [string]$Name = "",
    [switch]$Matrix,
    [int]$Batch = 1,
    [int]$Cin = 1,
    [int]$Cout = 0,
    [int]$IH = 8,
    [int]$IW = 8,
    [int]$MBase = 9,
    [int]$NBase = 0
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Gen = Join-Path $ProjectRoot "tb\edge\gen_tile_edge_data.py"

$args = @(
    "--shape", $Shape.ToLower(),
    "--dataflow", $Mode.ToLower(),
    "--lanes", $Lanes,
    "--batch", $Batch,
    "--cin", $Cin,
    "--ih", $IH,
    "--iw", $IW,
    "--m-base", $MBase,
    "--n-base", $NBase
)

if ($Name) {
    $args += @("--name", $Name)
}
if ($Cout -gt 0) {
    $args += @("--cout", $Cout)
}
if ($Matrix) {
    $args += "--matrix"
}

Write-Host "Generating P2 tile edge case data" -ForegroundColor Cyan
& python $Gen @args
if ($LASTEXITCODE -ne 0) {
    throw "tile edge case generation failed."
}
