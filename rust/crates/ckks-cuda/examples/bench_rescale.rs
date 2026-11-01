//! Four-tier benchmark for `ckks_cuda::rescale::RescaleContext`, at
//! n=1024 / the same 8-tower modulus chain as `bench/bench_rescale_cpp_
//! vs_rust.cpp`'s `RescaleContextCpp`, structured to mirror it tier-for-
//! tier (same methodology as `bench_roundtrip.rs` established for the
//! first ported operation):
//!   1. cold-start: host-only (native_rescale_consts FFI call + native_
//!      root x8 + table builds for every tower), no GPU.
//!   2. context init: RescaleContext::new, including cold-start + device
//!      allocation + table upload.
//!   3. steady-state: repeated RescaleContext::rescale calls on ONE
//!      reused context -- the real per-operation cost for a caller doing
//!      many rescales.
//!   4. end-to-end: a fresh context + its first rescale, combined.
//!
//! Correctness is NOT re-checked here: `RescaleContext::rescale` is
//! already exhaustively verified against an independent RNS-only
//! reference by this crate's own test suite
//! (`rescale::tests::rescale_matches_independent_rns_reference_random_
//! data`, 0 mismatches across 3 trials of independent random per-
//! coefficient data) and, separately, by `bench_rescale_cpp_vs_rust.cpp`'s
//! own pre-timing correctness gate against the SAME reference formula.
//! This benchmark times the proven operation; it does not re-derive its
//! correctness. The timed input below is raw synthetic per-tower
//! residues (not a forward-NTT'd "real" eval-domain ciphertext) --
//! exactly like the C++ benchmark's own Tier 3/4 workload -- because the
//! kernels' arithmetic cost does not depend on whether the bit pattern
//! happens to be a valid NTT output, only on the array sizes and moduli,
//! which are identical either way.
use ckks_core::RingDimension;
use ckks_cuda::{rescale_cold_start_cost_probe, RescaleContext};
use std::time::Instant;

// Same 8-tower chain as rust/crates/ckks-cuda/src/rescale.rs's test
// module and bench/bench_rescale_cpp_vs_rust.cpp's main().
const MODULI: [u64; 8] = [
    1152921504606830593,
    36028797018820609,
    36028797018802177,
    36028797018789889,
    36028797018746881,
    36028797018667009,
    36028797018652673,
    36028797018646529,
];

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
    let towers = MODULI.len();
    let last = towers - 1;
    let nn = n.get() as usize;

    let input: Vec<u64> = (0..towers)
        .flat_map(|t| {
            let q = MODULI[t];
            (0..nn).map(move |k| ((t as u64 * 97 + k as u64) * 1000003 + 7) % q)
        })
        .collect();

    let reps: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(50);
    let trials: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(7);

    // --- Tier 1: cold-start, host-only. ---
    let mut cold_start_us = Vec::with_capacity(trials);
    for _ in 0..trials {
        let t0 = Instant::now();
        let checksum = rescale_cold_start_cost_probe(n, &MODULI);
        std::hint::black_box(checksum);
        cold_start_us.push(t0.elapsed().as_secs_f64() * 1e6);
    }
    report(
        "cold-start (host-only: native_rescale_consts + native_root x8 + table builds)",
        &mut cold_start_us,
    );

    // --- Tier 2: context init (RescaleContext::new, incl. cold-start + device alloc + table upload). ---
    let mut init_us = Vec::with_capacity(trials);
    for _ in 0..trials {
        let t0 = Instant::now();
        let ctx = RescaleContext::new(n, &MODULI).expect("context init failed");
        init_us.push(t0.elapsed().as_secs_f64() * 1e6);
        drop(ctx);
    }
    report("context init (RescaleContext::new, incl. cold-start)", &mut init_us);

    // --- Tier 3: steady-state, ONE context reused across all reps/trials. ---
    let ctx = RescaleContext::new(n, &MODULI).expect("context init failed");
    let mut out = vec![0u64; last * nn];
    // Warm-up call, same reasoning as bench_roundtrip.rs: first use of a
    // freshly allocated scratch buffer can pay a one-time page-fault/
    // driver cost that would otherwise bias trial 1 unfairly.
    ctx.rescale(&input, &mut out).expect("warm-up rescale failed");

    let mut per_trial_median_us = Vec::with_capacity(trials);
    for _ in 0..trials {
        let mut per_call_us = Vec::with_capacity(reps);
        for _ in 0..reps {
            let t0 = Instant::now();
            ctx.rescale(&input, &mut out).expect("rescale failed");
            per_call_us.push(t0.elapsed().as_secs_f64() * 1e6);
        }
        per_trial_median_us.push(median(&mut per_call_us));
    }
    report("steady-state rescale (reused context)", &mut per_trial_median_us);

    // --- Tier 4: end-to-end, fresh context + first rescale, combined. ---
    let mut e2e_us = Vec::with_capacity(trials);
    for _ in 0..trials {
        let t0 = Instant::now();
        let fresh = RescaleContext::new(n, &MODULI).expect("context init failed");
        let mut out2 = vec![0u64; last * nn];
        fresh.rescale(&input, &mut out2).expect("rescale failed");
        e2e_us.push(t0.elapsed().as_secs_f64() * 1e6);
    }
    report("end-to-end (fresh context + first rescale)", &mut e2e_us);

    println!("---");
    println!("ckks-cuda (Rust host + existing CUDA C++ rescale kernels), n=1024, {towers} towers, {trials} trials x {reps} reps/trial for the steady-state tier.");
}
