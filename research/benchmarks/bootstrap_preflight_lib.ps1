<#
  GOD PROMPT "CKKS PHASE 25": shared library for bootstrap_preflight.ps1
  and its unit tests (bootstrap_preflight.tests.ps1). Dot-source this
  file to get the pure decision function and the live-measurement
  functions without triggering any live system call or launch logic --
  this file defines functions only, it runs nothing on its own.

  Kept in its own file specifically so the unit tests can exercise
  Get-BootstrapGoNoGoDecision with synthetic, mocked inputs (missing
  measurement, insufficient RAM, insufficient VRAM, WSL down, all-clear)
  WITHOUT calling Get-WindowsFreeRam/Test-WslAvailable/Get-GpuFreeVramMB
  at all -- i.e. without touching real system memory, per the governing
  directive's own "use mocked resource measurements for harness unit
  tests rather than manipulating real system memory" instruction.
#>

function Get-BootstrapGoNoGoDecision {
    <#
    .SYNOPSIS
      Pure GO/NO-GO decision: takes plain numeric/boolean inputs, returns
      a plain decision object. No live system call, no side effect.
    #>
    param(
        [Nullable[double]]$FreeRamGB,        # $null = measurement unavailable/untrusted
        [double]$TotalRamGB,
        [double]$MinFreeRamGB,
        [Nullable[double]]$FreeVramMB,        # $null = measurement unavailable
        [double]$MinFreeVramMB,
        [bool]$WslAvailable,
        [string]$BootstrapConfig
    )

    $reasons = New-Object System.Collections.Generic.List[string]
    $go = $true

    if ($null -eq $FreeRamGB -or $FreeRamGB -le 0 -or $TotalRamGB -le 0 -or $FreeRamGB -gt $TotalRamGB) {
        $go = $false
        $reasons.Add("Windows free-RAM measurement unavailable or untrustworthy (FreeRamGB=$FreeRamGB, TotalRamGB=$TotalRamGB) -- refusing to decide GO. Per policy, an untrustworthy measurement is treated as NO-GO, never as an implicit pass.")
    } elseif ($FreeRamGB -lt $MinFreeRamGB) {
        $go = $false
        $reasons.Add(("Windows free RAM {0:N2}GiB is below the required {1:N2}GiB floor for {2}." -f $FreeRamGB, $MinFreeRamGB, $BootstrapConfig))
    } else {
        $reasons.Add(("Windows free RAM {0:N2}GiB meets the {1:N2}GiB floor." -f $FreeRamGB, $MinFreeRamGB))
    }

    if (-not $WslAvailable) {
        $go = $false
        $reasons.Add("WSL did not respond to a basic liveness check -- cannot safely launch anything that depends on it.")
    }

    if ($null -eq $FreeVramMB) {
        $reasons.Add("GPU free-VRAM measurement unavailable (nvidia-smi query failed or WSL unreachable) -- not independently fatal, since Windows host RAM is this project's established real constraint, but recorded as a gap.")
    } elseif ($FreeVramMB -lt $MinFreeVramMB) {
        $go = $false
        $reasons.Add(("GPU free VRAM {0:N0}MiB is below the required {1:N0}MiB floor." -f $FreeVramMB, $MinFreeVramMB))
    } else {
        $reasons.Add(("GPU free VRAM {0:N0}MiB meets the {1:N0}MiB floor." -f $FreeVramMB, $MinFreeVramMB))
    }

    [PSCustomObject]@{
        Go      = $go
        Reasons = $reasons
    }
}

function Get-BootstrapWorkloadRequirement {
    <#
    .SYNOPSIS
      Pure workload -> required-free-RAM estimator. Maps a CKKS ring
      dimension (and, where it matters, the modulus-chain sizes) onto the
      host free-RAM floor that configuration must clear. No live system
      call, no side effect -- unit-testable with plain integers.

    .DESCRIPTION
      Phase 25 found that bootstrap_preflight.ps1, as recovered from the
      lost session, took -BootstrapConfig as FREE TEXT that it never
      parsed, and so could only ever apply ONE fixed threshold. The
      governing directive requires two DIFFERENT floors (n=8192 and
      n=131072), so the threshold has to be derived from the workload
      rather than hardcoded. This function is that derivation.

      The floors are EVIDENCE-BASED, not first-principles allocation
      accounting, and each carries its own basis string so a reader can
      audit where the number came from:

        n=8192  (sizeQ=30, sizeP=2) -> 10 GiB
          tests/test_boot.cpp's own source comment records a measured
          peak host RSS of 8.12-8.32GB (post-KSC-cap-fix) for exactly
          this configuration. 10 GiB is that peak plus a deliberate
          margin, since host free RAM on this machine has been measured
          to swing by several GB WITHIN a single session.

        n=131072 -> 6 GiB
          The directive's own explicit floor. Note what this gate does
          and does not cover: at n=131072 this project has never
          completed a full bootstrap -- it OOMs far earlier than the
          KROT_CAP/KSC_CAP bound predicts (see
          research/MEMORY_FEASIBILITY.md and tests/
          test_n131072_cache_churn.cpp). 6 GiB is therefore the floor
          for ATTEMPTING the n=131072 key-size / cache-churn work, not a
          claim that 6 GiB suffices to finish a bootstrap at that ring
          dimension.

      FAIL-CLOSED on an unrecognized ring dimension: returns the MAXIMUM
      known floor (not the minimum, and not a permissive default), and
      flags Recognized=$false so the caller can report that the
      threshold was assumed rather than derived.

    .NOTES
      UNITS. Get-WindowsFreeRam below computes
      $os.FreePhysicalMemory / 1MB. Win32_OperatingSystem reports that
      property in KILOBYTES, and PowerShell's 1MB literal is 1048576, so
      the quotient is KB/1048576 = GiB -- NOT GB. Every "RamGB" name in
      this harness is therefore a GiB value, and the floors here are GiB
      to match. The names are kept as-is for continuity with the
      recovered code and the project's existing documentation; this note
      is the correction. The distinction is not academic at these
      magnitudes: 10 GiB is 10.74 GB, so the gate is slightly MORE
      conservative than the prose "10GB" suggests, which is the safe
      direction for a fail-closed resource gate.
    #>
    param(
        [Parameter(Mandatory)][int]$N,
        [int]$SizeQ = 0,
        [int]$SizeP = 0
    )

    # Keep this table and the fail-closed maximum in one place so they
    # cannot drift apart.
    $known = @(
        [PSCustomObject]@{
            N = 8192; MinFreeRamGB = 10.0
            Label = "n=8192,sizeQ=30,sizeP=2 (tests/test_boot.cpp and tests/test_boot_multi_input.cpp)"
            Basis = "test_boot.cpp's own documented 8.12-8.32GB historical peak host RSS, plus margin for this machine's measured intra-session free-RAM swings."
        },
        [PSCustomObject]@{
            N = 131072; MinFreeRamGB = 6.0
            Label = "n=131072 (tests/test_boot_n131072.cpp, tests/test_n131072_cache_churn.cpp)"
            Basis = "The governing directive's explicit 6 GiB floor. Gates ATTEMPTING the n=131072 key-size/cache-churn work; a full bootstrap has never completed at this ring dimension (OOMs early -- see research/MEMORY_FEASIBILITY.md)."
        }
    )

    $maxKnown = ($known | Measure-Object -Property MinFreeRamGB -Maximum).Maximum
    $hit = $known | Where-Object { $_.N -eq $N } | Select-Object -First 1

    if ($hit) {
        $label = $hit.Label
        # Only assert the full sizeQ/sizeP-specific label when the caller
        # actually supplied them AND they match the configuration the
        # historical RSS figure was measured at.
        if ($N -eq 8192 -and $SizeQ -gt 0 -and ($SizeQ -ne 30 -or $SizeP -ne 2)) {
            return [PSCustomObject]@{
                MinFreeRamGB = $maxKnown
                Label        = "n=8192 with a NON-REFERENCE modulus chain (sizeQ=$SizeQ, sizeP=$SizeP)"
                Basis        = "The 8.12-8.32GB historical peak RSS was measured at sizeQ=30,sizeP=2 ONLY. A different chain length changes the per-tower allocation footprint, so that measurement does not transfer. Falling back to the maximum known floor ($maxKnown GiB) rather than assuming the reference figure applies."
                Recognized   = $false
            }
        }
        return [PSCustomObject]@{
            MinFreeRamGB = $hit.MinFreeRamGB
            Label        = $label
            Basis        = $hit.Basis
            Recognized   = $true
        }
    }

    [PSCustomObject]@{
        MinFreeRamGB = $maxKnown
        Label        = "n=$N (UNRECOGNIZED configuration)"
        Basis        = "No measured peak-RSS figure exists for this ring dimension. Failing closed: applying the maximum known floor ($maxKnown GiB) rather than guessing a lower one. Supply -MinFreeRamGB explicitly to override, and record the measured peak RSS so this table can be extended."
        Recognized   = $false
    }
}

function Get-WindowsFreeRam {
    <# Live measurement. MUST come from Get-CimInstance Win32_OperatingSystem
       -- never WSL `free -h` -- per the governing directive. Degrades to
       $null on any failure rather than throwing or defaulting to a value
       that could look like a pass.

       UNITS: FreePhysicalMemory/TotalVisibleMemorySize are in KILOBYTES;
       dividing by 1MB (1048576) yields GiB, not GB. See
       Get-BootstrapWorkloadRequirement's .NOTES for why this matters. #>
    try {
        $os = Get-CimInstance Win32_OperatingSystem -ErrorAction Stop
        [PSCustomObject]@{
            FreeRamGB  = [double]$os.FreePhysicalMemory / 1MB
            TotalRamGB = [double]$os.TotalVisibleMemorySize / 1MB
        }
    } catch {
        Write-Warning "Get-CimInstance Win32_OperatingSystem failed: $_"
        [PSCustomObject]@{ FreeRamGB = $null; TotalRamGB = 0.0 }
    }
}

function Test-WslAvailable {
    <# Live liveness check -- NOT a substitute for the Windows RAM
       measurement above; this only confirms WSL itself responds. #>
    try {
        $out = & wsl.exe -d Ubuntu -- bash -c "echo ok" 2>$null
        return ($LASTEXITCODE -eq 0 -and $out -match 'ok')
    } catch {
        return $false
    }
}

function Get-GpuFreeVramMB {
    <# Live measurement, informational/secondary -- GPU VRAM has never
       been the limiting resource in this project's own history; Windows
       host RAM is. #>
    try {
        $out = & wsl.exe -d Ubuntu -- nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>$null
        if ($LASTEXITCODE -ne 0 -or -not $out) { return $null }
        $val = ($out | Select-Object -First 1).Trim()
        return [double]$val
    } catch {
        return $null
    }
}
