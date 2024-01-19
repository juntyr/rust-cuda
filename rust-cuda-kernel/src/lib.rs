//! [![CI Status]][workflow] [![MSRV]][repo] [![Rust Doc]][docs] [![License
//! Status]][fossa] [![Code Coverage]][codecov] [![Gitpod
//! Ready-to-Code]][gitpod]
//!
//! [CI Status]: https://img.shields.io/github/actions/workflow/status/juntyr/rust-cuda/ci.yml?branch=main
//! [workflow]: https://github.com/juntyr/rust-cuda/actions/workflows/ci.yml?query=branch%3Amain
//!
//! [MSRV]: https://img.shields.io/badge/MSRV-1.77.0--nightly-orange
//! [repo]: https://github.com/juntyr/rust-cuda
//!
//! [Rust Doc]: https://img.shields.io/badge/docs-main-blue
//! [docs]: https://juntyr.github.io/rust-cuda/rust_cuda_kernel/
//!
//! [License Status]: https://app.fossa.com/api/projects/custom%2B26490%2Fgithub.com%2Fjuntyr%2Frust-cuda.svg?type=shield
//! [fossa]: https://app.fossa.com/projects/custom%2B26490%2Fgithub.com%2Fjuntyr%2Frust-cuda?ref=badge_shield
//!
//! [Code Coverage]: https://img.shields.io/codecov/c/github/juntyr/rust-cuda?token=wfeAeybbbx
//! [codecov]: https://codecov.io/gh/juntyr/rust-cuda
//!
//! [Gitpod Ready-to-Code]: https://img.shields.io/badge/Gitpod-ready-blue?logo=gitpod
//! [gitpod]: https://gitpod.io/#https://github.com/juntyr/rust-cuda

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
