#![deny(clippy::pedantic)]
#![allow(clippy::useless_attribute)]
#![no_std]
#![feature(associated_type_bounds)]
#![cfg_attr(not(feature = "host"), feature(link_llvm_intrinsics))]
#![cfg_attr(not(feature = "host"), feature(core_intrinsics))]
#![cfg_attr(not(feature = "host"), feature(asm))]

#[doc(hidden)]
pub extern crate alloc;

pub extern crate rust_cuda_ptx_jit as ptx_jit;
pub extern crate rustacuda_core;

#[cfg(any(feature = "derive", doc))]
pub extern crate rustacuda_derive;

pub mod common;

#[cfg(any(feature = "host", doc))]
pub mod host;

#[cfg(any(all(not(feature = "host"), feature = "derive"), doc))]
pub mod host {
    pub use rust_cuda_derive::LendToCuda;
}

#[cfg(any(feature = "host", doc))]
pub extern crate rustacuda;

#[cfg(any(all(not(feature = "host"), target_os = "cuda"), doc))]
pub mod device;

pub mod utils;
