#![deny(clippy::complexity)]
#![deny(clippy::correctness)]
#![warn(clippy::nursery)]
#![warn(clippy::pedantic)]
#![deny(clippy::perf)]
#![deny(clippy::style)]
#![deny(clippy::suspicious)]
#![feature(if_let_guard)]
#![feature(let_chains)]
#![doc(html_root_url = "https://juntyr.github.io/rust-cuda/")]

extern crate proc_macro;

#[macro_use]
extern crate proc_macro_error;

use proc_macro::TokenStream;

mod rust_to_cuda;

#[proc_macro_error]
#[proc_macro_derive(LendRustToCuda, attributes(cuda))]
pub fn rust_to_cuda_derive(input: TokenStream) -> TokenStream {
    // Note: We cannot report a more precise span yet
    let ast = match syn::parse(input) {
        Ok(ast) => ast,
        Err(err) => abort!(err),
    };

    // Build the implementation of the `RustToCuda` and `CudaAsRust` traits
    rust_to_cuda::impl_rust_to_cuda(&ast)
}
