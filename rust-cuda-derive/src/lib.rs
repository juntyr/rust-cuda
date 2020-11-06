#![deny(clippy::pedantic)]

extern crate proc_macro;

#[macro_use]
extern crate proc_macro_error;

use proc_macro::TokenStream;

mod generics;
mod lend_to_cuda;
mod rust_to_cuda;

#[proc_macro_error]
#[proc_macro_derive(RustToCuda, attributes(r2cEmbed, r2cBound))]
pub fn rust_to_cuda_derive(input: TokenStream) -> TokenStream {
    let ast = match syn::parse(input) {
        Ok(ast) => ast,
        Err(err) => abort!(err),
    };

    // Build the implementation of the `RustToCuda` trait
    rust_to_cuda::impl_rust_to_cuda(&ast)
}

#[proc_macro_error]
#[proc_macro_derive(LendToCuda)]
pub fn lend_to_cuda_derive(input: TokenStream) -> TokenStream {
    let ast = match syn::parse(input) {
        Ok(ast) => ast,
        Err(err) => abort!(err),
    };

    // Build the implementation of the `LendToCuda` trait
    lend_to_cuda::impl_lend_to_cuda(&ast)
}
