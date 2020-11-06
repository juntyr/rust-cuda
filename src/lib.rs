#![deny(clippy::pedantic)]
#![no_std]
#![cfg_attr(not(feature = "host"), feature(link_llvm_intrinsics))]
#![cfg_attr(not(feature = "host"), feature(core_intrinsics))]
#![cfg_attr(not(feature = "host"), feature(asm))]

pub mod common;

#[cfg(feature = "host")]
pub mod host;

#[cfg(not(feature = "host"))]
pub mod device;
