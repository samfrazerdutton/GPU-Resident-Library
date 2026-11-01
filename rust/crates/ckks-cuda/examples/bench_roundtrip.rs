//! Four-tier benchmark for `ckks_cuda::NttContext`, at n=1024 / tower 0's
//! modulus (same chain this round's C++ work uses), structured to mirror
//! `bench/bench_ntt_roundtrip_cpp_vs_rust.cpp`'s `NttContextCpp` tier-for-
//! tier so the two languages' numbers compare the SAME workload at each
//! tier:
//!   1. cold-start: host-only (native_root FFI call + table build), no GPU.
//!   2. context init: NttContext::new, including cold-start + device
//!      allocation + table upload.
//!   3. steady-state: repeated NttContext::roundtrip calls on ONE reused
//!      context -- the real per-operation cost for a caller doing many
//!      round trips.
//!   4. end-to-end: a fresh context + its first roundtrip, combined.
//!
//! Multiple outer trials per tier, reporting min/median/max ACROSS
//! trials (not a single favorable run), matching the project's own
//! existing benchmark methodology
//! (bench/bench_evalmult_resident.cpp's median+dispersion approach).
use ckks_core::{Modulus, RingDimension};
use ckks_cuda::{cold_start_cost_probe, NttContext};
use std::time::Instant;

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    }
}

fn report(label: &str, per_trial: &mut [f64]) {
    let min = per_trial.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = per_trial.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let m = median(per_trial);
    println!(
        "{label}: median={m:.2}us min={min:.2}us max={max:.2}us (n={} trials)",
        per_trial.len()
    );
}

fn main() {
    let n = RingDimension::new(1024).unwrap();
    let q = Modulus::new(1152921504606830593).unwrap();
    let input: Vec<u64> = (0..1024u64).map(|k| (k * 1000003 + 7) % q.get()).collect();

    let reps: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(50);
    let trials: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(7);

    // --- Tier 1: cold-start, host-only (native_root FFI call + both
    // table builds -- see cold_start_cost_probe's doc comment for why
    // this must match the C++ side's cold-start tier exactly). ---
    let mut cold_start_us = Vec::with_capacity(trials);
    for _ in 0..trials {
        let t0 = Instant::now();
        let checksum = cold_start_cost_probe(n, q);
        std::hint::black_box(checksum);
        cold_start_us.push(t0.elapsed().as_secs_f64() * 1e6);
    }
    report("cold-start (host-only: native_root + table build)", &mut cold_start_us);

    // --- Tier 2: context init (NttContext::new, incl. cold-start + device alloc + table upload). ---
    let mut init_us = Vec::with_capacity(trials);
    for _ in 0..trials {
        let t0 = Instant::now();
        let ctx = NttContext::new(n, q).expect("context init failed");
        init_us.push(t0.elapsed().as_secs_f64() * 1e6);
        drop(ctx);
    }
    report("context init (NttContext::new, incl. cold-start)", &mut init_us);

    // --- Tier 3: steady-state, ONE context reused across all reps/trials. ---
    let ctx = NttContext::new(n, q).expect("context init failed");
    let mut out = vec![0u64; 1024];
    // Warm-up call: first use of a freshly allocated scratch buffer can
    // pay a one-time page-fault/driver cost that would otherwise bias
    // trial 1 unfairly against later trials.
    ctx.roundtrip(&input, &mut out).expect("warm-up roundtrip failed");

    let mut per_trial_median_us = Vec::with_capacity(trials);
    for _ in 0..trials {
        let mut per_call_us = Vec::with_capacity(reps);
        for _ in 0..reps {
            let t0 = Instant::now();
            ctx.roundtrip(&input, &mut out).expect("roundtrip failed");
            let elapsed = t0.elapsed().as_secs_f64() * 1e6;
            assert_eq!(out, input, "correctness check failed mid-benchmark");
            per_call_us.push(elapsed);
        }
        per_trial_median_us.push(median(&mut per_call_us));
    }
    report("steady-state roundtrip (reused context)", &mut per_trial_median_us);

    // --- Tier 4: end-to-end, fresh context + first roundtrip, combined. ---
    let mut e2e_us = Vec::with_capacity(trials);
    for _ in 0..trials {
        let t0 = Instant::now();
        let fresh = NttContext::new(n, q).expect("context init failed");
        let mut out2 = vec![0u64; 1024];
        fresh.roundtrip(&input, &mut out2).expect("roundtrip failed");
        e2e_us.push(t0.elapsed().as_secs_f64() * 1e6);
        assert_eq!(out2, input, "correctness check failed mid-benchmark");
    }
    report("end-to-end (fresh context + first roundtrip)", &mut e2e_us);

    println!("---");
    println!("ckks-cuda (Rust host + existing CUDA C++ NTT/INTT kernels), n=1024, {trials} trials x {reps} reps/trial for the steady-state tier.");
}
