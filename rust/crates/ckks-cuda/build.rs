// Builds this crate's CUDA/C++ dependencies directly with nvcc/g++,
// independent of the main project's CMake build. This is deliberate, not
// an oversight: the main `gpufhe` CMake target has
// `CUDA_SEPARABLE_COMPILATION ON` (needed for OTHER files that call
// device functions across translation units), which requires an extra
// `nvcc -dlink` step to link against. kernels/ntt_ct.cu and
// kernels/intt_ct.cu are each fully self-contained (every `__device__`
// function they call is defined in the SAME file -- verified by reading
// both files; see this crate's own doc comment in src/lib.rs), so
// compiling them WITHOUT separable compilation (the default) produces
// ordinary, directly-linkable object files and avoids that whole step.
// Reuses the EXISTING, unmodified .cu/.cpp source files from the main
// project -- this build script does not duplicate or modify any
// cryptographic or NTT logic, only the build invocation.
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

fn main() {
    let root = repo_root();
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    // CUDA toolkit location: trust what nvcc's own install reports via
    // CUDA_PATH/CUDA_HOME if set, else the conventional /usr/local/cuda
    // symlink this environment is confirmed to have (see this round's
    // toolchain detection: CUDA 13.2 at /usr/local/cuda).
    let cuda_root = env::var("CUDA_PATH")
        .or_else(|_| env::var("CUDA_HOME"))
        .unwrap_or_else(|_| "/usr/local/cuda".to_string());
    let cuda_lib = Path::new(&cuda_root).join("lib64");

    // --- Device kernels: nvcc, NO separable compilation (see module doc). ---
    let ntt_o = out_dir.join("ntt_ct.o");
    let intt_o = out_dir.join("intt_ct.o");
    run_or_panic(
        Command::new("nvcc")
            .args([
                "-std=c++17",
                "-arch=sm_75", // matches CMakeLists.txt's CMAKE_CUDA_ARCHITECTURES for this RTX 2060 Max-Q
                "-Xcompiler",
                "-fPIC",
                "-c",
            ])
            .arg(root.join("kernels/ntt_ct.cu"))
            .arg("-o")
            .arg(&ntt_o),
    );
    run_or_panic(
        Command::new("nvcc")
            .args(["-std=c++17", "-arch=sm_75", "-Xcompiler", "-fPIC", "-c"])
            .arg(root.join("kernels/intt_ct.cu"))
            .arg("-o")
            .arg(&intt_o),
    );

    // --- Host-only helpers: plain g++, no CUDA involved at all. ---
    // src/params.cpp is the EXISTING, unmodified file defining
    // gpufhe::native_root (and native_primes/native_rescale_consts, not
    // used by this crate yet). The bridge file forward-declares and wraps
    // it with extern "C" linkage -- see cpp_bridge/native_root_bridge.cpp.
    let params_o = out_dir.join("params.o");
    let bridge_o = out_dir.join("native_root_bridge.o");
    run_or_panic(
        Command::new("g++")
            .args(["-std=c++17", "-fPIC", "-O2", "-c"])
            .arg(root.join("src/params.cpp"))
            .arg("-o")
            .arg(&params_o),
    );
    run_or_panic(
        Command::new("g++")
            .args(["-std=c++17", "-fPIC", "-O2", "-c"])
            .arg(
                PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap())
                    .join("cpp_bridge/native_root_bridge.cpp"),
            )
            .arg("-o")
            .arg(&bridge_o),
    );

    for obj in [&ntt_o, &intt_o, &params_o, &bridge_o] {
        println!("cargo:rustc-link-arg={}", obj.display());
    }
    println!("cargo:rustc-link-search=native={}", cuda_lib.display());
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=stdc++");

    println!("cargo:rerun-if-changed={}", root.join("kernels/ntt_ct.cu").display());
    println!("cargo:rerun-if-changed={}", root.join("kernels/intt_ct.cu").display());
    println!("cargo:rerun-if-changed={}", root.join("src/params.cpp").display());
    println!(
        "cargo:rerun-if-changed={}",
        PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap())
            .join("cpp_bridge/native_root_bridge.cpp")
            .display()
    );
}
