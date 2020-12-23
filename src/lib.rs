#![deny(clippy::pedantic)]
#![no_std]
#![feature(associated_type_bounds)]
#![cfg_attr(not(feature = "host"), feature(link_llvm_intrinsics))]
#![cfg_attr(not(feature = "host"), feature(core_intrinsics))]
#![cfg_attr(not(feature = "host"), feature(asm))]

#[doc(hidden)]
pub extern crate alloc;

pub mod common;

#[cfg(feature = "host")]
pub mod host;

#[cfg(not(feature = "host"))]
pub mod device;

pub mod utils;
