#![deny(clippy::pedantic)]
#![allow(clippy::useless_attribute)]
#![cfg_attr(not(feature = "host"), no_std)]
#![feature(associated_type_bounds)]
#![feature(auto_traits)]
#![feature(negative_impls)]
#![cfg_attr(
    any(all(not(feature = "host"), target_os = "cuda"), doc),
    feature(stdsimd)
)]
#![cfg_attr(
    any(all(not(feature = "host"), target_os = "cuda"), doc),
    feature(asm_experimental_arch)
)]
#![cfg_attr(
    any(all(not(feature = "host"), target_os = "cuda"), doc),
    feature(asm_const)
)]
#![feature(doc_cfg)]
#![feature(marker_trait_attr)]
#![feature(const_type_name)]
#![feature(offset_of)]
#![feature(adt_const_params)]
#![feature(impl_trait_in_assoc_type)]
#![feature(ptr_metadata)]
#![allow(incomplete_features)]
#![feature(generic_const_exprs)]
#![cfg_attr(target_os = "cuda", feature(slice_ptr_get))]
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
