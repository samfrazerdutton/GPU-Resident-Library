// Builds this crate's CUDA/C++ dependencies directly with nvcc/g++,
// independent of the main project's CMake build. This is deliberate, not
// an oversight: the main `gpufhe` CMake target has
// `CUDA_SEPARABLE_COMPILATION ON` (needed for OTHER files that call
// device functions across translation units), which requires an extra
// `nvcc -dlink` step to link against. Every kernel file compiled below
// (ntt_ct.cu, intt_ct.cu, switch_modulus.cu, rescale.cu) is fully
// self-contained -- every `__device__` function it calls is defined in
// the SAME file, confirmed by reading each one -- so compiling them
// WITHOUT separable compilation (the default) produces ordinary,
// directly-linkable object files and avoids that whole step. Reuses the
// EXISTING, unmodified .cu/.cpp source files from the main project --
// this build script does not duplicate or modify any cryptographic or
// NTT/rescale logic, only the build invocation.
use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn repo_root() -> PathBuf {
    // CARGO_MANIFEST_DIR = <repo>/rust/crates/ckks-cuda
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    manifest_dir
        .parent() // rust/crates
        .unwrap()
        .parent() // rust
        .unwrap()
        .parent() // <repo>
        .unwrap()
        .to_path_buf()
}

fn run_or_panic(cmd: &mut Command) {
    let status = cmd.status().unwrap_or_else(|e| {
        panic!("failed to spawn {:?}: {e}", cmd);
    });
    if !status.success() {
        panic!("command failed ({status}): {:?}", cmd);
    }
}

/// Compiles one `.cu` device-kernel file with nvcc, NO separable
/// compilation (see module doc), into `out_dir/<stem>.o`. `include_dir` is
/// the main project's `include/` (needed starting this round: the
/// key-switch kernel files are still self-contained w.r.t. `__device__`
/// calls, but passing it uniformly costs nothing and avoids re-deriving
/// this per kernel file as more operations are added).
fn compile_cuda(src: &Path, out_dir: &Path, include_dir: &Path) -> PathBuf {
    let stem = src.file_stem().unwrap().to_str().unwrap();
    let out = out_dir.join(format!("{stem}.o"));
    run_or_panic(
        Command::new("nvcc")
            .args([
                "-std=c++17",
                "-arch=sm_75", // matches CMakeLists.txt's CMAKE_CUDA_ARCHITECTURES for this RTX 2060 Max-Q
                "-Xcompiler",
                "-fPIC",
                "-c",
            ])
            .arg("-I")
            .arg(include_dir)
            .arg(src)
            .arg("-o")
            .arg(&out),
    );
    println!("cargo:rerun-if-changed={}", src.display());
    out
}

/// Compiles one host-only `.cpp` file with g++, into `out_dir/<stem>.o`.
/// `include_dir` is the main project's `include/` (needed starting this
/// round: the key-switch host files -- keygen.cpp, evalkeygen.cpp,
/// ks_constants.cpp, keyswitch.cpp, keyswitch_resident.cpp, and this
/// crate's new keyswitch_bridge.cpp -- all `#include "keyswitch.h"` etc.,
/// unlike params.cpp/the first two bridge files, which were header-free).
/// `cuda_include_dir` is needed for the same files' `#include
/// <cuda_runtime.h>` (g++ has no built-in CUDA include path; nvcc does, so
/// this is a no-op duplicate for compile_cuda but still passed there for
/// uniformity).
fn compile_cpp(src: &Path, out_dir: &Path, include_dir: &Path, cuda_include_dir: &Path) -> PathBuf {
    let stem = src.file_stem().unwrap().to_str().unwrap();
    let out = out_dir.join(format!("{stem}.o"));
    run_or_panic(
        Command::new("g++")
            .args(["-std=c++17", "-fPIC", "-O2", "-c"])
            .arg("-I")
            .arg(include_dir)
            .arg("-I")
            .arg(cuda_include_dir)
            .arg(src)
            .arg("-o")
            .arg(&out),
    );
    println!("cargo:rerun-if-changed={}", src.display());
    out
}

fn main() {
    let root = repo_root();
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    // CUDA toolkit location: trust what nvcc's own install reports via
    // CUDA_PATH/CUDA_HOME if set, else the conventional /usr/local/cuda
    // symlink this environment is confirmed to have (see this round's
    // toolchain detection: CUDA 13.2 at /usr/local/cuda).
    let cuda_root = env::var("CUDA_PATH")
        .or_else(|_| env::var("CUDA_HOME"))
        .unwrap_or_else(|_| "/usr/local/cuda".to_string());
    let cuda_lib = Path::new(&cuda_root).join("lib64");
    let cuda_include = Path::new(&cuda_root).join("include");
    let include_dir = root.join("include");

    // --- Device kernels (ntt op, rescale op, key-switch op). basis_convert.cu
    // / ks_fastcore.cu / moddown.cu confirmed self-contained the same way as
    // the first four (grepped for cross-file `__device__`/`extern` device
    // symbols: none; each only exports `extern "C"` host-callable Launch*
    // wrappers), so they compile and link the same way, no `-rdc=true` needed.
    let mut objs = vec![
        compile_cuda(&root.join("kernels/ntt_ct.cu"), &out_dir, &include_dir),
        compile_cuda(&root.join("kernels/intt_ct.cu"), &out_dir, &include_dir),
        compile_cuda(&root.join("kernels/switch_modulus.cu"), &out_dir, &include_dir),
        compile_cuda(&root.join("kernels/rescale.cu"), &out_dir, &include_dir),
        compile_cuda(&root.join("kernels/basis_convert.cu"), &out_dir, &include_dir),
        compile_cuda(&root.join("kernels/ks_fastcore.cu"), &out_dir, &include_dir),
        compile_cuda(&root.join("kernels/moddown.cu"), &out_dir, &include_dir),
        // EvalMult's two new kernels (tensor product + relin-combine).
        // Confirmed self-contained the same way (one inline `mulmod64`
        // device helper, defined and used in this one file).
        compile_cuda(&root.join("kernels/tensor.cu"), &out_dir, &include_dir),
    ];

    // --- Host-only helpers: plain g++. src/params.cpp defines
    // gpufhe::native_root/native_primes/native_rescale_consts. The five
    // files below (keygen/evalkeygen/ks_constants/keyswitch/
    // keyswitch_resident) are the EXISTING, unmodified native key-switch
    // pipeline this round's keyswitch_bridge.cpp wraps -- none of their
    // cryptographic logic is reimplemented here, only compiled directly
    // (same reasoning as every kernel file above).
    objs.push(compile_cpp(&root.join("src/params.cpp"), &out_dir, &include_dir, &cuda_include));
    objs.push(compile_cpp(&root.join("src/keygen.cpp"), &out_dir, &include_dir, &cuda_include));
    objs.push(compile_cpp(&root.join("src/evalkeygen.cpp"), &out_dir, &include_dir, &cuda_include));
    objs.push(compile_cpp(&root.join("src/ks_constants.cpp"), &out_dir, &include_dir, &cuda_include));
    objs.push(compile_cpp(&root.join("src/keyswitch.cpp"), &out_dir, &include_dir, &cuda_include));
    objs.push(compile_cpp(&root.join("src/keyswitch_resident.cpp"), &out_dir, &include_dir, &cuda_include));
    // GOD PROMPT "CKKS PHASE 10": the Hybrid key-switch P>=Q_part_max
    // validator (include/ks_params_validate.h), called from
    // keyswitch_resident.cpp's ks_context_create -- pure host C++, no
    // CUDA, but still compiled via this build script's g++ path since it
    // links into the same static object set every other file here does.
    objs.push(compile_cpp(&root.join("src/ks_params_validate.cpp"), &out_dir, &include_dir, &cuda_include));
    // src/encode.cpp / src/encdec.cpp: the EXISTING, unmodified native
    // CKKS encode/decode/encrypt/decrypt primitives (GOD PROMPT "CKKS
    // PHASE 5"), wrapped by the new cpp_bridge/encdec_bridge.cpp below.
    // Both confirmed pure-host, no device state retained across calls.
    objs.push(compile_cpp(&root.join("src/encode.cpp"), &out_dir, &include_dir, &cuda_include));
    objs.push(compile_cpp(&root.join("src/encdec.cpp"), &out_dir, &include_dir, &cuda_include));
    objs.push(compile_cpp(
        &manifest_dir.join("cpp_bridge/native_root_bridge.cpp"),
        &out_dir, &include_dir, &cuda_include,
    ));
    objs.push(compile_cpp(
        &manifest_dir.join("cpp_bridge/rescale_consts_bridge.cpp"),
        &out_dir, &include_dir, &cuda_include,
    ));
    objs.push(compile_cpp(
        &manifest_dir.join("cpp_bridge/keyswitch_bridge.cpp"),
        &out_dir, &include_dir, &cuda_include,
    ));
    objs.push(compile_cpp(
        &manifest_dir.join("cpp_bridge/encdec_bridge.cpp"),
        &out_dir, &include_dir, &cuda_include,
    ));

    for obj in &objs {
        println!("cargo:rustc-link-arg={}", obj.display());
    }
    println!("cargo:rustc-link-search=native={}", cuda_lib.display());
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=stdc++");
}
