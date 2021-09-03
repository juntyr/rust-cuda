#![deny(clippy::pedantic)]
#![allow(clippy::useless_attribute)]
#![no_std]
#![feature(associated_type_bounds)]
#![feature(auto_traits)]
#![feature(negative_impls)]
#![cfg_attr(
    any(all(not(feature = "host"), target_os = "cuda"), doc),
    feature(stdsimd)
)]
#![cfg_attr(any(feature = "alloc", doc), feature(allocator_api))]
#![feature(doc_cfg)]
#![feature(marker_trait_attr)]
#![allow(incomplete_features)]
#![feature(specialization)]

#[doc(hidden)]
pub extern crate alloc;

pub extern crate rust_cuda_ptx_jit as ptx_jit;
pub extern crate rustacuda_core;

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
