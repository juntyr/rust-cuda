//! [![CI Status]][workflow] [![MSRV]][repo] [![Rust Doc]][docs] [![License
//! Status]][fossa] [![Code Coverage]][codecov] [![Gitpod
//! Ready-to-Code]][gitpod]
//!
//! [CI Status]: https://img.shields.io/github/actions/workflow/status/juntyr/rust-cuda/ci.yml?branch=main
//! [workflow]: https://github.com/juntyr/rust-cuda/actions/workflows/ci.yml?query=branch%3Amain
//!
//! [MSRV]: https://img.shields.io/badge/MSRV-1.81.0--nightly-orange
//! [repo]: https://github.com/juntyr/rust-cuda
//!
//! [Rust Doc]: https://img.shields.io/badge/docs-main-blue
//! [docs]: https://juntyr.github.io/rust-cuda/rust_cuda/
//!
//! [License Status]: https://app.fossa.com/api/projects/custom%2B26490%2Fgithub.com%2Fjuntyr%2Frust-cuda.svg?type=shield
//! [fossa]: https://app.fossa.com/projects/custom%2B26490%2Fgithub.com%2Fjuntyr%2Frust-cuda?ref=badge_shield
//!
//! [Code Coverage]: https://img.shields.io/codecov/c/github/juntyr/rust-cuda?token=wfeAeybbbx
//! [codecov]: https://codecov.io/gh/juntyr/rust-cuda
//!
//! [Gitpod Ready-to-Code]: https://img.shields.io/badge/Gitpod-ready-blue?logo=gitpod
//! [gitpod]: https://gitpod.io/#https://github.com/juntyr/rust-cuda

#![allow(missing_docs)] // FIXME
#![allow(clippy::undocumented_unsafe_blocks)] // FIXME
#![allow(clippy::multiple_unsafe_ops_per_block)] // FIXME
#![allow(clippy::indexing_slicing)] // FIXME
#![cfg_attr(all(any(feature = "device", target_os = "cuda"), not(doc)), no_std)]
#![feature(auto_traits)]
#![feature(negative_impls)]
#![cfg_attr(all(feature = "device", not(doc)), feature(stdarch_nvptx))]
#![cfg_attr(feature = "device", feature(asm_experimental_arch))]
#![cfg_attr(feature = "device", feature(asm_const))]
#![feature(doc_auto_cfg)]
#![feature(doc_cfg)]
#![feature(marker_trait_attr)]
#![feature(const_type_name)]
#![feature(adt_const_params)]
#![feature(impl_trait_in_assoc_type)]
#![feature(ptr_metadata)]
#![feature(decl_macro)]
#![feature(let_chains)]
#![feature(sync_unsafe_cell)]
#![feature(never_type)]
#![feature(layout_for_ptr)]
#![feature(cfg_version)]
#![cfg_attr(any(feature = "host", feature = "device"), feature(slice_ptr_get))]
#![expect(incomplete_features)]
#![feature(generic_const_exprs)]
#![expect(internal_features)]
#![feature(core_intrinsics)]
#![feature(const_intrinsic_compare_bytes)]
#![doc(html_root_url = "https://juntyr.github.io/rust-cuda/")]

#[cfg(all(feature = "host", feature = "device", not(doc)))]
core::compile_error!("cannot enable the `host` and `device` features at the same time");

#[cfg(all(feature = "host", target_os = "cuda", not(doc)))]
core::compile_error!("cannot enable the `host` feature on a target with `target_os=\"cuda\"`");

#[cfg(all(feature = "device", not(target_os = "cuda"), not(doc)))]
core::compile_error!("cannot enable the `device` feature on a target without `target_os=\"cuda\"`");

pub mod alloc;
pub mod deps;
pub mod kernel;
pub mod lend;
pub mod safety;
pub mod utils;

#[cfg(feature = "host")]
pub mod host;

#[cfg(feature = "device")]
pub mod device;
