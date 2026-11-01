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
/// compilation (see module doc), into `out_dir/<stem>.o`.
fn compile_cuda(src: &Path, out_dir: &Path) -> PathBuf {
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
            .arg(src)
            .arg("-o")
            .arg(&out),
    );
    println!("cargo:rerun-if-changed={}", src.display());
    out
}

/// Compiles one host-only `.cpp` file with g++ (no CUDA at all), into
/// `out_dir/<stem>.o`.
fn compile_cpp(src: &Path, out_dir: &Path) -> PathBuf {
    let stem = src.file_stem().unwrap().to_str().unwrap();
    let out = out_dir.join(format!("{stem}.o"));
    run_or_panic(
        Command::new("g++")
            .args(["-std=c++17", "-fPIC", "-O2", "-c"])
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

    // --- Device kernels (ntt op, then rescale op). ---
    let mut objs = vec![
        compile_cuda(&root.join("kernels/ntt_ct.cu"), &out_dir),
        compile_cuda(&root.join("kernels/intt_ct.cu"), &out_dir),
        compile_cuda(&root.join("kernels/switch_modulus.cu"), &out_dir),
        compile_cuda(&root.join("kernels/rescale.cu"), &out_dir),
    ];

    // --- Host-only helpers: plain g++, no CUDA involved at all. ---
    // src/params.cpp is the EXISTING, unmodified file defining
    // gpufhe::native_root AND gpufhe::native_rescale_consts (both used by
    // this crate now, via the two bridge files below).
    objs.push(compile_cpp(&root.join("src/params.cpp"), &out_dir));
    objs.push(compile_cpp(
        &manifest_dir.join("cpp_bridge/native_root_bridge.cpp"),
        &out_dir,
    ));
    objs.push(compile_cpp(
        &manifest_dir.join("cpp_bridge/rescale_consts_bridge.cpp"),
        &out_dir,
    ));

    for obj in &objs {
        println!("cargo:rustc-link-arg={}", obj.display());
    }
    println!("cargo:rustc-link-search=native={}", cuda_lib.display());
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=stdc++");
}
