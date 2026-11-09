<#
.SYNOPSIS
  GOD PROMPT "CKKS PHASE 25": resource-aware GO/NO-GO preflight for the
  n=8192 CKKS bootstrap (tests/test_boot.cpp and its multi-input
  sibling, tests/test_boot_multi_input.cpp).

.DESCRIPTION
  This script NEVER launches the bootstrap itself and NEVER takes any
  corrective action (no process is killed, no service is restarted, no
  .wslconfig/system setting is touched). It does exactly one thing:
  measure, decide, and report. On a GO decision it prints the exact
  command an operator must run manually next -- launching remains a
  deliberate, separate human (or agent) action, never an automatic
  consequence of running this script.

  Phase 23/24 established, the hard way, that WSL `free -h` and
  `nvidia-smi`'s reported headroom are BOTH systematically misleading
  for this project's real constraint -- Windows host physical RAM,
  under WSL2's Hyper-V passthrough, is what actually gates whether a
  CUDA-heavy process can run without destabilizing the whole machine
  (a `HCS_E_CONNECTION_TIMEOUT` was reproduced in Phase 24 at ~0.49GB
  free). This script therefore measures Windows free RAM via
  `Get-CimInstance Win32_OperatingSystem` ONLY -- never WSL `free -h`
  -- and refuses to produce a GO decision if that measurement cannot be
  obtained or looks untrustworthy (zero/negative/implausible values).

  The historical basis for the default 10GB threshold: tests/
  test_boot.cpp's own source comment records a measured peak host RSS
  of 8.12-8.32GB (post-KSC-cap-fix) for the n=8192/sizeQ=30/sizeP=2
  configuration. 10GB is a provisional floor with a margin over that
  figure, not a number derived from first-principles allocation
  accounting -- see -MinFreeRamGB to raise it if a more conservative
  analysis is ever done, and research/EXPERIMENT_LEDGER.md entry #58
  for the full reasoning.

  The decision logic itself lives in bootstrap_preflight_lib.ps1 (dot-
  sourced below) as a pure function taking plain values -- this script
  is only the live-measurement + reporting shell around it. See
  bootstrap_preflight.tests.ps1 for unit tests against that pure
  function using mocked inputs, never real system memory.

.PARAMETER N
  CKKS ring dimension of the workload being gated. Used ONLY to look up
  the documented free-RAM floor for that configuration (see
  Get-BootstrapWorkloadRequirement in bootstrap_preflight_lib.ps1) --
  this script never writes, and never feeds back into, any
  cryptographic parameter. Default 8192 (tests/test_boot.cpp and
  tests/test_boot_multi_input.cpp). Pass -N 131072 for the n=131072
  work, which has its own lower 6 GiB floor. An unrecognized value
  fails CLOSED: the maximum known floor is applied and a warning is
  emitted.

.PARAMETER SizeQ
.PARAMETER SizeP
  Modulus-chain sizes of the workload being gated. Used ONLY for
  threshold lookup, never modified. They matter because the documented
  8.12-8.32GB peak RSS was measured at sizeQ=30,sizeP=2 specifically;
  passing a different chain at n=8192 is treated as unrecognized and
  falls back to the maximum known floor rather than assuming the
  reference figure transfers. Defaults 30 / 2.

.PARAMETER MinFreeRamGB
  Explicit override for the required Windows free physical RAM, in GiB.
  Default 0, meaning "derive the floor from -N/-SizeQ/-SizeP" -- which
  for the n=8192 default derives 10 GiB, a provisional floor ABOVE the
  documented 8.12-8.32GB historical peak, not below it. Any value > 0
  overrides the derivation, and the log record marks that entry's
  threshold_source as an override so it can never be mistaken for a
  derived decision. Configurable without touching any cryptographic
  parameter (n, sizeQ, sizeP, h, etc. are never written by this script).

  UNITS: GiB, not GB. Win32_OperatingSystem reports free memory in
  kilobytes and this harness divides by 1048576, so every "GB"-suffixed
  value here is really a GiB value. See
  Get-BootstrapWorkloadRequirement's .NOTES.

.PARAMETER MinFreeVramMB
  Required free GPU VRAM, in MiB, before a GO decision. Default 4096 --
  informational/secondary; Windows host RAM, not VRAM, is this
  project's established real constraint.

.PARAMETER BootstrapConfig
  Free-text label identifying which configuration this preflight check
  is for. Purely descriptive; never parsed or used to change the
  thresholds above.

.PARAMETER DryRun
  When set, the decision is computed and printed exactly as normal, but
  NO record is appended to the persistent log file
  (research/benchmarks/bootstrap_preflight_log.jsonl). Dry-run or not,
  this script NEVER launches anything.

.PARAMETER LogPath
  Override the persistent JSONL log file path. Defaults to
  research/benchmarks/bootstrap_preflight_log.jsonl next to this
  script. Ignored when -DryRun is set.

.EXAMPLE
  .\research\benchmarks\bootstrap_preflight.ps1

.EXAMPLE
  powershell.exe -File research/benchmarks/bootstrap_preflight.ps1 -DryRun

.EXAMPLE
  .\research\benchmarks\bootstrap_preflight.ps1 -MinFreeRamGB 12
#>
[CmdletBinding()]
param(
    [int]$N = 8192,
    [int]$SizeQ = 30,
    [int]$SizeP = 2,
    [double]$MinFreeRamGB = 0.0,
    [double]$MinFreeVramMB = 4096.0,
    [string]$BootstrapConfig = $null,
    [switch]$DryRun,
    [string]$LogPath = $null
)

$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot "bootstrap_preflight_lib.ps1")

# Phase 25: the RAM floor is now DERIVED from the requested workload
# rather than hardcoded, so that n=8192 and n=131072 get their own
# (different) documented thresholds instead of sharing one. An explicit
# -MinFreeRamGB still wins -- but only upward or downward by deliberate
# operator choice, and the record below always states which of the two
# it was, so a log entry can never be mistaken for a derived decision
# when it was in fact an override.
$workload = Get-BootstrapWorkloadRequirement -N $N -SizeQ $SizeQ -SizeP $SizeP
$thresholdSource = "derived"
if ($MinFreeRamGB -gt 0.0) {
    $thresholdSource = "explicit -MinFreeRamGB override"
} else {
    $MinFreeRamGB = $workload.MinFreeRamGB
}
if (-not $BootstrapConfig) { $BootstrapConfig = $workload.Label }

$timestamp = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
$ram = Get-WindowsFreeRam
$wslOk = Test-WslAvailable
$vramMB = if ($wslOk) { Get-GpuFreeVramMB } else { $null }

$decision = Get-BootstrapGoNoGoDecision `
    -FreeRamGB $ram.FreeRamGB -TotalRamGB $ram.TotalRamGB -MinFreeRamGB $MinFreeRamGB `
    -FreeVramMB $vramMB -MinFreeVramMB $MinFreeVramMB `
    -WslAvailable $wslOk -BootstrapConfig $BootstrapConfig

$record = [PSCustomObject]@{
    timestamp         = $timestamp
    bootstrap_config  = $BootstrapConfig
    n                 = $N
    size_q            = $SizeQ
    size_p            = $SizeP
    free_ram_gib      = $ram.FreeRamGB
    total_ram_gib     = $ram.TotalRamGB
    min_free_ram_gib  = $MinFreeRamGB
    threshold_source  = $thresholdSource
    threshold_basis   = $workload.Basis
    config_recognized = $workload.Recognized
    wsl_available     = $wslOk
    free_vram_mb      = $vramMB
    min_free_vram_mb  = $MinFreeVramMB
    go                = $decision.Go
    reasons           = $decision.Reasons
    dry_run           = [bool]$DryRun
}

Write-Output "=== Bootstrap preflight: $timestamp ==="
Write-Output ("Config: {0}" -f $BootstrapConfig)
Write-Output ("RAM floor: {0:N2} GiB ({1})" -f $MinFreeRamGB, $thresholdSource)
Write-Output ("  basis: {0}" -f $workload.Basis)
if (-not $workload.Recognized) {
    Write-Warning "Workload n=$N was NOT recognized -- the floor above is the fail-closed maximum known value, not a measured figure for this configuration."
}
Write-Output ("Windows free RAM: {0} GiB (of {1:N2} GiB total)" -f `
    $(if ($null -ne $ram.FreeRamGB) { "{0:N2}" -f $ram.FreeRamGB } else { "UNAVAILABLE" }), $ram.TotalRamGB)
Write-Output ("WSL responsive: {0}" -f $wslOk)
Write-Output ("GPU free VRAM: {0}" -f $(if ($null -ne $vramMB) { "$vramMB MiB" } else { "UNAVAILABLE" }))
Write-Output ""
foreach ($r in $decision.Reasons) { Write-Output "  - $r" }
Write-Output ""
if ($decision.Go) {
    Write-Output "DECISION: GO"
    Write-Output ""
    Write-Output "This script does NOT launch anything itself. To proceed, an operator"
    Write-Output "must now explicitly run the bootstrap build/test manually, e.g.:"
    Write-Output "  wsl.exe -d Ubuntu -- bash -lc 'cd ~/GPU-Resident-Library && <build+run command>'"
} else {
    Write-Output "DECISION: NO-GO -- the bootstrap must NOT be launched."
}

if (-not $DryRun) {
    if (-not $LogPath) {
        $LogPath = Join-Path $PSScriptRoot "bootstrap_preflight_log.jsonl"
    }
    ($record | ConvertTo-Json -Compress) | Add-Content -Path $LogPath -Encoding utf8
    Write-Output ""
    Write-Output "Recorded to $LogPath"
} else {
    Write-Output ""
    Write-Output "(dry run -- no log record written)"
}

if ($null -eq $ram.FreeRamGB) { exit 2 }
exit $(if ($decision.Go) { 0 } else { 1 })
