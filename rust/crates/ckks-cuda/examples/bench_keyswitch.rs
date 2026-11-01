//! Benchmark for `ckks_cuda::keyswitch::KeySwitchContext`, at the same
//! n=1024/sizeQ=8/sizeP=2/numPart=2 native configuration as `bench/
//! bench_keyswitch_cpp_vs_rust.cpp`'s `KeySwitchFixtureCpp`, structured to
//! mirror it tier-for-tier:
//!   1. fixture creation: `KeySwitchContext::new` (modulus chain + keygen +
//!      evalkeygen + device context/work -- see the crate's `keyswitch`
//!      module doc comment and the C++ benchmark's header comment for why
//!      this is ONE tier here, unlike ntt/rescale's separate host-only
//!      "cold-start" tier: key generation itself launches GPU kernels, so
//!      there is no meaningful GPU-free sub-portion to split out).
//!   2. steady-state: repeated `KeySwitchContext::keyswitch` calls on ONE
//!      reused context.
//!   3. end-to-end: a fresh context + its first keyswitch, combined.
//!
//! Correctness is NOT re-checked here: `KeySwitchContext::keyswitch` is
//! already checked bit-exact against `keyswitch_core_resident` (the host
//! reference) by this crate's own test suite
//! (`keyswitch::tests::keyswitch_matches_host_reference_random_data` and
//! three other tests) and, separately, by `bench_keyswitch_cpp_vs_rust.cpp`'s
//! own pre-timing correctness gate against the SAME reference. This
//! benchmark times the proven operation; it does not re-derive its
//! correctness.
use ckks_core::RingDimension;
use ckks_cuda::KeySwitchContext;
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

fn deterministic_input(ctx: &KeySwitchContext) -> Vec<u64> {
    let nn = ctx.ring_dimension().get() as usize;
    let mut a = vec![0u64; ctx.size_ql() as usize * nn];
    for t in 0..ctx.size_ql() {
        let q = ctx.modulus_q(t);
        for k in 0..nn {
            a[t as usize * nn + k] = ((t as u64 * 97 + k as u64) * 1000003 + 7) % q;
        }
    }
    a
}

fn main() {
    let n = RingDimension::new(N).unwrap();
    let reps: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(20);
    let trials: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(7);

    // --- Tier 1: fixture creation (full setup, GPU-touching throughout). ---
    let mut init_us = Vec::with_capacity(trials);
    for trial in 0..trials {
        let t0 = Instant::now();
        let ctx = KeySwitchContext::new(n, SIZE_Q, SIZE_P, NUM_PART, 1000 + trial as u64)
            .expect("context init failed");
        init_us.push(t0.elapsed().as_secs_f64() * 1e6);
        drop(ctx);
    }
    report(
        "fixture creation (KeySwitchContext::new: modulus chain + keygen + evalkeygen + device context/work)",
        &mut init_us,
    );

    // --- Tier 2: steady-state, ONE context reused across all reps/trials. ---
    let ctx = KeySwitchContext::new(n, SIZE_Q, SIZE_P, NUM_PART, 42).expect("context init failed");
    let input = deterministic_input(&ctx);
    let expected_len = ctx.size_ql() as usize * ctx.ring_dimension().get() as usize;
    let mut ba0 = vec![0u64; expected_len];
    let mut ba1 = vec![0u64; expected_len];
    // Warm-up call, same reasoning as the ntt/rescale benchmarks.
    ctx.keyswitch(&input, &mut ba0, &mut ba1).expect("warm-up keyswitch failed");

    let mut per_trial_median_us = Vec::with_capacity(trials);
    for _ in 0..trials {
        let mut per_call_us = Vec::with_capacity(reps);
        for _ in 0..reps {
            let t0 = Instant::now();
            ctx.keyswitch(&input, &mut ba0, &mut ba1).expect("keyswitch failed");
            per_call_us.push(t0.elapsed().as_secs_f64() * 1e6);
        }
        per_trial_median_us.push(median(&mut per_call_us));
    }
    report("steady-state keyswitch (reused context)", &mut per_trial_median_us);

    // --- Tier 3: end-to-end, fresh context + first keyswitch, combined. ---
    let mut e2e_us = Vec::with_capacity(trials);
    for trial in 0..trials {
        let t0 = Instant::now();
        let fresh = KeySwitchContext::new(n, SIZE_Q, SIZE_P, NUM_PART, 5000 + trial as u64)
            .expect("context init failed");
        let mut out0 = vec![0u64; expected_len];
        let mut out1 = vec![0u64; expected_len];
        fresh.keyswitch(&input, &mut out0, &mut out1).expect("keyswitch failed");
        e2e_us.push(t0.elapsed().as_secs_f64() * 1e6);
    }
    report("end-to-end (fresh context + first keyswitch)", &mut e2e_us);

    println!("---");
    println!("ckks-cuda (Rust host + existing CUDA C++ key-switch kernels), n=1024, sizeQ=8, sizeP=2, numPart=2, {trials} trials x {reps} reps/trial for the steady-state tier.");
}
