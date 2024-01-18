#![deny(clippy::complexity)]
#![deny(clippy::correctness)]
#![warn(clippy::nursery)]
#![warn(clippy::pedantic)]
#![deny(clippy::perf)]
#![deny(clippy::style)]
#![deny(clippy::suspicious)]
#![deny(unsafe_code)]
// #![warn(missing_docs)] // FIXME
#![feature(box_patterns)]
#![feature(proc_macro_tracked_env)]
#![feature(proc_macro_span)]
#![feature(let_chains)]
#![feature(map_try_insert)]
#![feature(proc_macro_def_site)]
#![feature(proc_macro_c_str_literals)]
#![feature(cfg_version)]
#![cfg_attr(not(version("1.76.0")), feature(c_str_literals))]
#![doc(html_root_url = "https://juntyr.github.io/rust-cuda/")]

extern crate proc_macro;

#[macro_use]
extern crate proc_macro_error;

use proc_macro::TokenStream;

mod kernel;

#[proc_macro_error]
#[proc_macro_attribute]
pub fn kernel(attr: TokenStream, func: TokenStream) -> TokenStream {
    kernel::wrapper::kernel(attr, func)
}

#[doc(hidden)]
#[proc_macro_error]
#[proc_macro]
pub fn specialise_kernel_type(tokens: TokenStream) -> TokenStream {
    kernel::specialise::ty::specialise_kernel_type(tokens)
}

#[doc(hidden)]
#[proc_macro_error]
#[proc_macro]
pub fn specialise_kernel_entry_point(tokens: TokenStream) -> TokenStream {
    kernel::specialise::entry_point::specialise_kernel_entry_point(tokens)
}

#[doc(hidden)]
#[proc_macro_error]
#[proc_macro_attribute]
pub fn specialise_kernel_function(attr: TokenStream, func: TokenStream) -> TokenStream {
    kernel::specialise::function::specialise_kernel_function(attr, func)
}

#[doc(hidden)]
#[proc_macro_error]
#[proc_macro]
pub fn check_kernel(tokens: TokenStream) -> TokenStream {
    kernel::link::check_kernel(tokens)
}

#[doc(hidden)]
#[proc_macro_error]
#[proc_macro]
pub fn link_kernel(tokens: TokenStream) -> TokenStream {
    kernel::link::link_kernel(tokens)
}
