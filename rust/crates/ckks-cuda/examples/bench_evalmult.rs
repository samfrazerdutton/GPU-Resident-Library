//! Benchmark for `ckks_cuda::evalmult::EvalMultContext`, at the same
//! n=1024/sizeQ=8/sizeP=2/numPart=2 native configuration as `bench/
//! bench_evalmult_cpp_vs_rust.cpp`'s `EvalMultFixtureCpp`, structured to
//! mirror it tier-for-tier:
//!   1. fixture creation: `EvalMultContext::new` (builds the relin eval
//!      key + a matching rescale context + 11 per-call device buffers --
//!      see the crate's `evalmult` module doc comment for why this is
//!      ONE tier, same reasoning as `bench_keyswitch.rs`: key generation
//!      itself launches GPU kernels, so there's no meaningful host-only
//!      "cold-start" sub-tier here).
//!   2. steady-state: repeated `EvalMultContext::multiply_relin_rescale`
//!      calls on ONE reused context.
//!   3. end-to-end: a fresh context + its first call, combined.
//!
//! Correctness is NOT re-checked here: `EvalMultContext::
//! multiply_relin_rescale` is already checked against a fresh independent
//! reference (tensor/combine computed from scratch, relin via the
//! established keyswitch host oracle, rescale via the SAME from-scratch
//! RNS identity `rescale`'s own test independently proved) by this
//! crate's own test suite. This benchmark times the proven operation; it
//! does not re-derive its correctness.
use ckks_core::RingDimension;
use ckks_cuda::EvalMultContext;
use std::time::Instant;

const N: u32 = 1024;
const SIZE_Q: u32 = 8;
const SIZE_P: u32 = 2;
const NUM_PART: u32 = 2;

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

fn deterministic_input(ctx: &EvalMultContext, salt: u64) -> Vec<u64> {
    let nn = ctx.ring_dimension().get() as usize;
    let mut v = vec![0u64; ctx.input_towers() as usize * nn];
    for t in 0..ctx.input_towers() {
        let q = ctx.modulus_q(t);
        for k in 0..nn {
            v[t as usize * nn + k] = ((salt.wrapping_add(t as u64 * 97 + k as u64)) * 1000003 + 7) % q;
        }
    }
    v
}

fn main() {
    let n = RingDimension::new(N).unwrap();
    let reps: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(20);
    let trials: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(7);

    // --- Tier 1: fixture creation (full setup, GPU-touching throughout). ---
    let mut init_us = Vec::with_capacity(trials);
    for trial in 0..trials {
        let t0 = Instant::now();
        let ctx = EvalMultContext::new(n, SIZE_Q, SIZE_P, NUM_PART, 1000 + trial as u64)
            .expect("context init failed");
        init_us.push(t0.elapsed().as_secs_f64() * 1e6);
        drop(ctx);
    }
    report("fixture creation (EvalMultContext::new)", &mut init_us);

    // --- Tier 2: steady-state, ONE context reused across all reps/trials. ---
    let ctx = EvalMultContext::new(n, SIZE_Q, SIZE_P, NUM_PART, 42).expect("context init failed");
    let c0a = deterministic_input(&ctx, 1);
    let c1a = deterministic_input(&ctx, 2);
    let c0b = deterministic_input(&ctx, 3);
    let c1b = deterministic_input(&ctx, 4);
    // Warm-up call, same reasoning as the ntt/rescale/keyswitch benchmarks.
    ctx.multiply_relin_rescale(&c0a, &c1a, &c0b, &c1b).expect("warm-up multiply failed");

    let mut per_trial_median_us = Vec::with_capacity(trials);
    for _ in 0..trials {
        let mut per_call_us = Vec::with_capacity(reps);
        for _ in 0..reps {
            let t0 = Instant::now();
            ctx.multiply_relin_rescale(&c0a, &c1a, &c0b, &c1b).expect("multiply failed");
            per_call_us.push(t0.elapsed().as_secs_f64() * 1e6);
        }
        per_trial_median_us.push(median(&mut per_call_us));
    }
    report("steady-state multiply_relin_rescale (reused context)", &mut per_trial_median_us);

    // --- Tier 3: end-to-end, fresh context + first call, combined. ---
    let mut e2e_us = Vec::with_capacity(trials);
    for trial in 0..trials {
        let t0 = Instant::now();
        let fresh = EvalMultContext::new(n, SIZE_Q, SIZE_P, NUM_PART, 5000 + trial as u64)
            .expect("context init failed");
        fresh.multiply_relin_rescale(&c0a, &c1a, &c0b, &c1b).expect("multiply failed");
        e2e_us.push(t0.elapsed().as_secs_f64() * 1e6);
    }
    report("end-to-end (fresh context + first multiply_relin_rescale)", &mut e2e_us);

    println!("---");
    println!("ckks-cuda (Rust host + existing CUDA C++ tensor/keyswitch/rescale kernels), n=1024, sizeQ=8, sizeP=2, numPart=2, {trials} trials x {reps} reps/trial for the steady-state tier.");
}
