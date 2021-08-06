#![deny(clippy::pedantic)]
#![allow(clippy::useless_attribute)]
#![no_std]
#![feature(associated_type_bounds)]
#![feature(auto_traits)]
#![feature(negative_impls)]
#![cfg_attr(any(not(feature = "host"), doc), feature(link_llvm_intrinsics))]
#![cfg_attr(any(not(feature = "host"), doc), feature(core_intrinsics))]
#![cfg_attr(any(not(feature = "host"), doc), feature(asm))]
#![feature(doc_cfg)]

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
