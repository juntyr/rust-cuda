use rustacuda_core::DeviceCopy;

#[cfg(feature = "derive")]
pub use rust_cuda_derive::{specialise_kernel_entry, specialise_kernel_type};

use crate::common::{DeviceBoxConst, DeviceBoxMut, RustToCuda};

pub mod nvptx;
pub mod utils;

/// # Safety
/// This is an internal trait and should NEVER be implemented manually
pub unsafe trait BorrowFromRust: RustToCuda {
    /// # Safety
    /// This function is only safe to call iff `cuda_repr` is the
    /// `DeviceBoxConst` borrowed on the CPU using the corresponding
    /// `LendToCuda::lend_to_cuda`.
    unsafe fn with_borrow_from_rust<O, F: FnOnce(&Self) -> O>(
        cuda_repr: DeviceBoxConst<<Self as RustToCuda>::CudaRepresentation>,
        inner: F,
    ) -> O;

    /// # Safety
    /// This function is only safe to call iff `cuda_repr_mut` is the
    /// `DeviceBoxMut` borrowed on the CPU using the corresponding
    /// `LendToCuda::lend_to_cuda_mut`. Furthermore, since different GPU
    /// threads can access heap storage mutably inside the safe `inner` scope,
    /// there must not be any aliasing between concurrently running threads.
    unsafe fn with_borrow_from_rust_mut<O, F: FnOnce(&mut Self) -> O>(
        cuda_repr_mut: DeviceBoxMut<<Self as RustToCuda>::CudaRepresentation>,
        inner: F,
    ) -> O;
}

#[repr(transparent)]
pub struct AnyDeviceBoxConst(*const core::ffi::c_void);

impl AnyDeviceBoxConst {
    #[must_use]
    /// # Safety
    ///
    /// This method is only safe iff this `AnyDeviceBoxConst` contains a `*const
    /// T`
    pub unsafe fn into<T: Sized + DeviceCopy>(self) -> DeviceBoxConst<T> {
        DeviceBoxConst(self.0.cast())
    }
}

#[repr(transparent)]
pub struct AnyDeviceBoxMut(*mut core::ffi::c_void);

impl AnyDeviceBoxMut {
    #[must_use]
    /// # Safety
    ///
    /// This method is only safe iff this `AnyDeviceBoxMut` contains a `*mut T`
    pub unsafe fn into<T: Sized + DeviceCopy>(self) -> DeviceBoxMut<T> {
        DeviceBoxMut(self.0.cast())
    }
}
