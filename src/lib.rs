#![deny(clippy::pedantic)]
#![allow(clippy::useless_attribute)]
#![no_std]
#![feature(associated_type_bounds)]
#![cfg_attr(not(feature = "host"), feature(link_llvm_intrinsics))]
#![cfg_attr(not(feature = "host"), feature(core_intrinsics))]
#![cfg_attr(not(feature = "host"), feature(asm))]

#[doc(hidden)]
pub extern crate alloc;

pub extern crate rustacuda_core;

pub mod common;

#[cfg(feature = "host")]
pub mod host;

#[cfg(all(not(feature = "host"), feature = "derive"))]
pub mod host {
    pub use rust_cuda_derive::LendToCuda;
}

#[cfg(feature = "host")]
pub extern crate rustacuda;

#[cfg(not(feature = "host"))]
pub mod device;

pub mod utils;
