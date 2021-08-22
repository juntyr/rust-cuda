use rustacuda_core::DeviceCopy;

#[cfg(feature = "derive")]
#[doc(cfg(feature = "derive"))]
pub use rust_cuda_derive::{specialise_kernel_entry, specialise_kernel_type};

use crate::common::{DevicePointerConst, DevicePointerMut, RustToCuda, RustToCudaCore};

pub mod nvptx;
pub mod utils;

/// # Safety
/// This is an internal trait and should NEVER be implemented manually
pub unsafe trait BorrowFromRust: RustToCuda {
    /// # Safety
    /// This function is only safe to call iff `cuda_repr` is the
    /// `DevicePointerConst` borrowed on the CPU using the corresponding
    /// `LendToCuda::lend_to_cuda`.
    unsafe fn with_borrow_from_rust<O, F: FnOnce(&Self) -> O>(
        cuda_repr: DevicePointerConst<<Self as RustToCudaCore>::CudaRepresentation>,
        inner: F,
    ) -> O;

    /// # Safety
    /// This function is only safe to call iff `cuda_repr_mut` is the
    /// `DevicePointerMut` borrowed on the CPU using the corresponding
    /// `LendToCuda::lend_to_cuda_mut`. Furthermore, since different GPU
    /// threads can access heap storage mutably inside the safe `inner` scope,
    /// there must not be any aliasing between concurrently running threads.
    unsafe fn with_borrow_from_rust_mut<O, F: FnOnce(&mut Self) -> O>(
        cuda_repr_mut: DevicePointerMut<<Self as RustToCudaCore>::CudaRepresentation>,
        inner: F,
    ) -> O;
}
