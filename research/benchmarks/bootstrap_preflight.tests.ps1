<#
.SYNOPSIS
  GOD PROMPT "CKKS PHASE 25": unit tests for Get-BootstrapGoNoGoDecision
  (bootstrap_preflight_lib.ps1), using ONLY mocked/synthetic inputs --
  never real system memory, per the governing directive's explicit
  instruction. Dependency-free (no Pester, no external module): each
  case asserts via plain if/throw so this runs on any PowerShell 5.1+
  with nothing installed beyond what the repo already requires.

.EXAMPLE
  powershell.exe -File research/benchmarks/bootstrap_preflight.tests.ps1
#>
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot "bootstrap_preflight_lib.ps1")

$script:pass = 0
$script:fail = 0

function Assert-Decision {
    param(
        [string]$Name,
        [Parameter(Mandatory)] $Decision,
        [Parameter(Mandatory)][bool]$ExpectGo,
        [string]$ExpectReasonContains = $null
    )
    $ok = $true
    if ($Decision.Go -ne $ExpectGo) {
        Write-Output "  [FAIL] $Name : expected Go=$ExpectGo, got Go=$($Decision.Go)"
        $ok = $false
    }
    if ($ExpectReasonContains) {
        $found = $Decision.Reasons | Where-Object { $_ -match [regex]::Escape($ExpectReasonContains) }
        if (-not $found) {
            Write-Output "  [FAIL] $Name : expected a reason containing '$ExpectReasonContains', got: $($Decision.Reasons -join ' | ')"
            $ok = $false
        }
    }
    if ($ok) {
        Write-Output "  [PASS] $Name"
        $script:pass++
    } else {
        $script:fail++
    }
}

Write-Output "=== bootstrap_preflight_lib.ps1 unit tests (mocked inputs only) ==="

# Case 1: all-clear -- plenty of RAM, plenty of VRAM, WSL up.
$d = Get-BootstrapGoNoGoDecision -FreeRamGB 14.0 -TotalRamGB 16.0 -MinFreeRamGB 10.0 `
    -FreeVramMB 5000.0 -MinFreeVramMB 4096.0 -WslAvailable $true -BootstrapConfig "test"
Assert-Decision -Name "all-clear -> GO" -Decision $d -ExpectGo $true

# Case 2: insufficient Windows RAM (the Phase 24/25 real-world case: ~2-3GB free, need 10GB).
$d = Get-BootstrapGoNoGoDecision -FreeRamGB 2.2 -TotalRamGB 15.42 -MinFreeRamGB 10.0 `
    -FreeVramMB 5900.0 -MinFreeVramMB 4096.0 -WslAvailable $true -BootstrapConfig "test"
Assert-Decision -Name "insufficient RAM -> NO-GO" -Decision $d -ExpectGo $false -ExpectReasonContains "is below the required"

# Case 3: missing Windows RAM measurement entirely (Get-CimInstance failed).
$d = Get-BootstrapGoNoGoDecision -FreeRamGB $null -TotalRamGB 0.0 -MinFreeRamGB 10.0 `
    -FreeVramMB 5900.0 -MinFreeVramMB 4096.0 -WslAvailable $true -BootstrapConfig "test"
Assert-Decision -Name "missing RAM measurement -> NO-GO (never an implicit pass)" -Decision $d -ExpectGo $false -ExpectReasonContains "unavailable or untrustworthy"

# Case 4: implausible RAM measurement (free > total -- a corrupted/untrustworthy reading).
$d = Get-BootstrapGoNoGoDecision -FreeRamGB 20.0 -TotalRamGB 16.0 -MinFreeRamGB 10.0 `
    -FreeVramMB 5900.0 -MinFreeVramMB 4096.0 -WslAvailable $true -BootstrapConfig "test"
Assert-Decision -Name "implausible RAM (free > total) -> NO-GO" -Decision $d -ExpectGo $false -ExpectReasonContains "unavailable or untrustworthy"

# Case 5: sufficient RAM but WSL unresponsive.
$d = Get-BootstrapGoNoGoDecision -FreeRamGB 14.0 -TotalRamGB 16.0 -MinFreeRamGB 10.0 `
    -FreeVramMB 5000.0 -MinFreeVramMB 4096.0 -WslAvailable $false -BootstrapConfig "test"
Assert-Decision -Name "WSL down -> NO-GO even with enough RAM" -Decision $d -ExpectGo $false -ExpectReasonContains "WSL did not respond"

# Case 6: sufficient RAM, WSL up, but insufficient GPU VRAM.
$d = Get-BootstrapGoNoGoDecision -FreeRamGB 14.0 -TotalRamGB 16.0 -MinFreeRamGB 10.0 `
    -FreeVramMB 1000.0 -MinFreeVramMB 4096.0 -WslAvailable $true -BootstrapConfig "test"
Assert-Decision -Name "insufficient VRAM -> NO-GO" -Decision $d -ExpectGo $false -ExpectReasonContains "GPU free VRAM"

# Case 7: VRAM measurement missing (nvidia-smi unreachable) -- recorded as a gap, NOT independently fatal,
# since Windows host RAM is this project's established real constraint.
$d = Get-BootstrapGoNoGoDecision -FreeRamGB 14.0 -TotalRamGB 16.0 -MinFreeRamGB 10.0 `
    -FreeVramMB $null -MinFreeVramMB 4096.0 -WslAvailable $true -BootstrapConfig "test"
Assert-Decision -Name "missing VRAM measurement alone -> still GO (RAM/WSL both fine)" -Decision $d -ExpectGo $true -ExpectReasonContains "VRAM measurement unavailable"

# Case 8: boundary -- exactly at the RAM floor must pass (>=, not >).
$d = Get-BootstrapGoNoGoDecision -FreeRamGB 10.0 -TotalRamGB 16.0 -MinFreeRamGB 10.0 `
    -FreeVramMB 5000.0 -MinFreeVramMB 4096.0 -WslAvailable $true -BootstrapConfig "test"
Assert-Decision -Name "exactly at RAM floor -> GO" -Decision $d -ExpectGo $true

# Case 9: just under the RAM floor must fail.
$d = Get-BootstrapGoNoGoDecision -FreeRamGB 9.99 -TotalRamGB 16.0 -MinFreeRamGB 10.0 `
    -FreeVramMB 5000.0 -MinFreeVramMB 4096.0 -WslAvailable $true -BootstrapConfig "test"
Assert-Decision -Name "just under RAM floor -> NO-GO" -Decision $d -ExpectGo $false

# Case 10: multiple simultaneous failures must all be reported, not just the first.
$d = Get-BootstrapGoNoGoDecision -FreeRamGB 2.0 -TotalRamGB 16.0 -MinFreeRamGB 10.0 `
    -FreeVramMB 500.0 -MinFreeVramMB 4096.0 -WslAvailable $false -BootstrapConfig "test"
Assert-Decision -Name "multiple failures (RAM+WSL+VRAM) -> NO-GO, RAM reason present" -Decision $d -ExpectGo $false -ExpectReasonContains "is below the required"
Assert-Decision -Name "multiple failures (RAM+WSL+VRAM) -> NO-GO, WSL reason present" -Decision $d -ExpectGo $false -ExpectReasonContains "WSL did not respond"
Assert-Decision -Name "multiple failures (RAM+WSL+VRAM) -> NO-GO, VRAM reason present" -Decision $d -ExpectGo $false -ExpectReasonContains "GPU free VRAM"

# ---------------------------------------------------------------------
# Get-BootstrapWorkloadRequirement -- the Phase 25 addition. These are
# pure integer-in / object-out cases: no system memory is read, and the
# function has no live dependency at all.
# ---------------------------------------------------------------------

function Assert-Requirement {
    param(
        [string]$Name,
        [Parameter(Mandatory)] $Req,
        [Parameter(Mandatory)][double]$ExpectGB,
        [Parameter(Mandatory)][bool]$ExpectRecognized
    )
    $ok = $true
    if ([math]::Abs($Req.MinFreeRamGB - $ExpectGB) -gt 1e-9) {
        Write-Output "  [FAIL] $Name : expected MinFreeRamGB=$ExpectGB, got $($Req.MinFreeRamGB)"
        $ok = $false
    }
    if ($Req.Recognized -ne $ExpectRecognized) {
        Write-Output "  [FAIL] $Name : expected Recognized=$ExpectRecognized, got $($Req.Recognized)"
        $ok = $false
    }
    if (-not $Req.Basis) {
        Write-Output "  [FAIL] $Name : Basis string is empty -- every threshold must carry its justification"
        $ok = $false
    }
    if ($ok) { Write-Output "  [PASS] $Name"; $script:pass++ } else { $script:fail++ }
}

Write-Output ""
Write-Output "=== Get-BootstrapWorkloadRequirement unit tests (pure, no live measurement) ==="

# Case 11: the reference n=8192 configuration the 8.12-8.32GB peak RSS was measured at.
$r = Get-BootstrapWorkloadRequirement -N 8192 -SizeQ 30 -SizeP 2
Assert-Requirement -Name "n=8192,sizeQ=30,sizeP=2 -> 10 GiB, recognized" -Req $r -ExpectGB 10.0 -ExpectRecognized $true

# Case 12: n=131072 gets its OWN, lower floor -- proves the table is actually
# consulted rather than a constant being returned.
$r = Get-BootstrapWorkloadRequirement -N 131072
Assert-Requirement -Name "n=131072 -> 6 GiB (its own floor, not n=8192's), recognized" -Req $r -ExpectGB 6.0 -ExpectRecognized $true

# Case 13: n=8192 with a NON-reference modulus chain must NOT inherit the
# reference figure -- the historical RSS was measured at sizeQ=30,sizeP=2 only.
$r = Get-BootstrapWorkloadRequirement -N 8192 -SizeQ 20 -SizeP 2
Assert-Requirement -Name "n=8192 with sizeQ=20 -> fails closed to max known floor, NOT recognized" -Req $r -ExpectGB 10.0 -ExpectRecognized $false

# Case 14: an entirely unknown ring dimension must fail CLOSED to the MAXIMUM
# known floor -- never to the minimum, and never to a permissive default.
$r = Get-BootstrapWorkloadRequirement -N 16384
Assert-Requirement -Name "n=16384 (unknown) -> fails closed to max known floor, NOT recognized" -Req $r -ExpectGB 10.0 -ExpectRecognized $false

# Case 15: the fail-closed value must be the MAXIMUM of the known floors, not
# any individual one. Asserted structurally so that adding a higher floor to
# the table later cannot silently make the fallback too permissive.
$unknown = Get-BootstrapWorkloadRequirement -N 999999
$knownFloors = @(
    (Get-BootstrapWorkloadRequirement -N 8192 -SizeQ 30 -SizeP 2).MinFreeRamGB,
    (Get-BootstrapWorkloadRequirement -N 131072).MinFreeRamGB
)
$maxFloor = ($knownFloors | Measure-Object -Maximum).Maximum
if ([math]::Abs($unknown.MinFreeRamGB - $maxFloor) -le 1e-9) {
    Write-Output "  [PASS] unknown config fallback equals the MAXIMUM known floor ($maxFloor GiB)"
    $script:pass++
} else {
    Write-Output "  [FAIL] unknown config fallback is $($unknown.MinFreeRamGB) GiB but max known floor is $maxFloor GiB -- fallback must never be more permissive than the strictest known case"
    $script:fail++
}

# Case 16: the estimator must be side-effect free and stable -- calling it
# twice with the same input must give the same answer (it is consulted inside
# a gate whose whole value depends on being deterministic).
$a = Get-BootstrapWorkloadRequirement -N 8192 -SizeQ 30 -SizeP 2
$b = Get-BootstrapWorkloadRequirement -N 8192 -SizeQ 30 -SizeP 2
if ($a.MinFreeRamGB -eq $b.MinFreeRamGB -and $a.Recognized -eq $b.Recognized -and $a.Basis -eq $b.Basis) {
    Write-Output "  [PASS] estimator is deterministic across repeated calls"
    $script:pass++
} else {
    Write-Output "  [FAIL] estimator returned different results for identical input"
    $script:fail++
}

# Case 17: end-to-end wiring -- a derived n=131072 floor of 6 GiB must turn the
# SAME measured free-RAM figure that is a NO-GO at n=8192 into a GO. This is the
# whole point of deriving the threshold, so it is asserted directly.
$d = Get-BootstrapGoNoGoDecision -FreeRamGB 7.0 -TotalRamGB 15.42 `
    -MinFreeRamGB (Get-BootstrapWorkloadRequirement -N 8192 -SizeQ 30 -SizeP 2).MinFreeRamGB `
    -FreeVramMB 5900.0 -MinFreeVramMB 4096.0 -WslAvailable $true -BootstrapConfig "n=8192"
Assert-Decision -Name "7 GiB free at n=8192 (10 GiB floor) -> NO-GO" -Decision $d -ExpectGo $false
$d = Get-BootstrapGoNoGoDecision -FreeRamGB 7.0 -TotalRamGB 15.42 `
    -MinFreeRamGB (Get-BootstrapWorkloadRequirement -N 131072).MinFreeRamGB `
    -FreeVramMB 5900.0 -MinFreeVramMB 4096.0 -WslAvailable $true -BootstrapConfig "n=131072"
Assert-Decision -Name "same 7 GiB free at n=131072 (6 GiB floor) -> GO" -Decision $d -ExpectGo $true

Write-Output ""
Write-Output "=== $script:pass passed, $script:fail failed ==="
if ($script:fail -gt 0) { exit 1 }
exit 0
