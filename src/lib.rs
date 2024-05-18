#![deny(clippy::pedantic)]
#![allow(clippy::useless_attribute)]
#![cfg_attr(not(feature = "host"), no_std)]
#![feature(auto_traits)]
#![feature(negative_impls)]
#![cfg_attr(
    any(all(not(feature = "host"), target_os = "cuda"), doc),
    feature(stdarch_nvptx)
)]
#![cfg_attr(any(feature = "alloc", doc), feature(allocator_api))]
#![feature(doc_cfg)]
#![feature(marker_trait_attr)]
#![feature(const_type_name)]
#![feature(adt_const_params)]
#![allow(incomplete_features)]
#![feature(generic_const_exprs)]
#![doc(html_root_url = "https://juntyr.github.io/rust-cuda/")]

#[doc(hidden)]
pub extern crate alloc;

pub extern crate rust_cuda_ptx_jit as ptx_jit;
pub extern crate rustacuda_core;

#[doc(hidden)]
#[macro_use]
pub extern crate const_type_layout;

#[cfg(feature = "derive")]
#[doc(cfg(feature = "derive"))]
pub extern crate rustacuda_derive;

pub mod common;

#[cfg(feature = "host")]
#[doc(cfg(feature = "host"))]
pub mod host;

#[cfg(feature = "host")]
#[doc(cfg(feature = "host"))]
pub extern crate rustacuda;

#[cfg(any(all(not(feature = "host"), target_os = "cuda"), doc))]
#[doc(cfg(all(not(feature = "host"), target_os = "cuda")))]
pub mod device;

pub mod utils;

pub mod safety;
