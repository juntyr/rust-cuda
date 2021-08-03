pub fn skip_kernel_compilation() -> bool {
    let mut skip_compilation = false;

    if let Ok(rustc) = proc_macro::tracked_env::var("RUSTC_WRAPPER") {
        skip_compilation |= rustc.contains("clippy-driver");
        skip_compilation |= rustc.contains("rust-analyzer");
    }

    if let Ok(rustc) = proc_macro::tracked_env::var("RUSTC_WORKSPACE_WRAPPER") {
        skip_compilation |= rustc.contains("clippy-driver");
        skip_compilation |= rustc.contains("rust-analyzer");
    }

    skip_compilation
}
