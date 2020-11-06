use crate::common::RustToCuda;

pub mod nvptx;
pub mod utils;

/// # Safety
/// This is an internal trait and should NEVER be implemented manually
pub unsafe trait BorrowFromRust: RustToCuda {
    /// # Safety
    /// This function is only safe to call iff `cuda_repr_ptr` is the `DevicePointer` borrowed on
    /// the CPU using the corresponding `LendToCuda::lend_to_cuda`.
    unsafe fn with_borrow_from_rust<O, F: FnOnce(&Self) -> O>(
        cuda_repr_ptr: *const <Self as RustToCuda>::CudaRepresentation,
        inner: F,
    ) -> O;

    /// # Safety
    /// This function is only safe to call iff `cuda_repr_ptr` is the `DevicePointer` borrowed on
    /// the CPU using the corresponding `LendToCuda::lend_to_cuda_mut`.
    /// Furthermore, since different GPU cores can access heap storage mutably inside the safe
    /// `inner` scope, there must not be any aliasing between concurrently running cores.
    unsafe fn with_borrow_from_rust_mut<O, F: FnOnce(&mut Self) -> O>(
        cuda_repr_ptr: *mut <Self as RustToCuda>::CudaRepresentation,
        inner: F,
    ) -> O;
}
