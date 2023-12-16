#![deny(clippy::pedantic)]
#![cfg_attr(not(feature = "host"), no_std)]
#![feature(cfg_version)]
#![cfg_attr(not(version("1.76.0")), feature(ptr_from_ref))]
#![feature(doc_cfg)]
#![doc(html_root_url = "https://juntyr.github.io/rust-cuda/")]

#[cfg(feature = "host")]
mod host;

#[cfg(feature = "host")]
pub use host::{compiler::PtxJITCompiler, compiler::PtxJITResult, kernel::CudaKernel};

#[cfg(any(not(feature = "host"), doc))]
#[doc(cfg(not(feature = "host")))]
mod device;

pub fn arg_as_raw_bytes<T: ?Sized>(r: &T) -> *const [u8] {
    core::ptr::slice_from_raw_parts(
        core::ptr::from_ref(r).cast::<u8>(),
        core::mem::size_of_val(r),
    )
}
