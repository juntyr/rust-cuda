#![deny(clippy::pedantic)]
#![cfg_attr(not(feature = "host"), no_std)]
#![feature(doc_cfg)]
#![doc(html_root_url = "https://juntyr.github.io/rust-cuda/")]

#[cfg(feature = "host")]
mod host;

#[cfg(feature = "host")]
pub use host::{compiler::PtxJITCompiler, compiler::PtxJITResult, kernel::CudaKernel};

#[cfg(any(not(feature = "host"), doc))]
#[doc(cfg(not(feature = "host")))]
mod device;
