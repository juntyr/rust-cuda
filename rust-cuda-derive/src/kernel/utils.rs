use syn::spanned::Spanned;

pub fn skip_kernel_compilation() -> bool {
    let mut skip_compilation = false;

    if let Ok(rustc) = proc_macro::tracked_env::var("RUSTC_WRAPPER") {
        skip_compilation |= rustc.contains("clippy-driver");
    }

    if let Ok(rustc) = proc_macro::tracked_env::var("RUSTC_WORKSPACE_WRAPPER") {
        skip_compilation |= rustc.contains("clippy-driver");
    }

    skip_compilation
}

pub fn r2c_move_lifetime(arg: usize, ty: &syn::Type) -> syn::Lifetime {
    syn::Lifetime::new(&format!("'__r2c_move_lt_{}", arg), ty.span())
}
