use rustacuda_core::DeviceCopy;

use super::DeviceAccessible;

#[doc(hidden)]
#[allow(clippy::module_name_repetitions)]
/// # Safety
///
/// This is an internal trait and should ONLY be derived automatically using
/// `#[derive(RustToCudaAsRust)]`
pub unsafe trait RustToCudaImpl {
    #[cfg(feature = "host")]
    type CudaAllocationImpl: crate::host::CudaAlloc;
    type CudaRepresentationImpl: CudaAsRustImpl<RustRepresentationImpl = Self>;

    #[cfg(feature = "host")]
    /// # Errors
    ///
    /// Returns a `rustacuda::errors::CudaError` iff an error occurs inside CUDA
    ///
    /// # Safety
    ///
    /// This is an internal function and should NEVER be called manually
    /// The returned `Self::CudaRepresentation` must NEVER be accessed on the
    ///  CPU  as it contains a GPU-resident copy of `self`.
    #[allow(clippy::type_complexity)]
    unsafe fn borrow_impl<A: crate::host::CudaAlloc>(
        &self,
        alloc: A,
    ) -> rustacuda::error::CudaResult<(
        DeviceAccessible<Self::CudaRepresentationImpl>,
        crate::host::CombinedCudaAlloc<Self::CudaAllocationImpl, A>,
    )>;

    #[cfg(feature = "host")]
    /// # Errors
    ///
    /// Returns a `rustacuda::errors::CudaError` iff an error occurs inside CUDA
    ///
    /// # Safety
    ///
    /// This is an internal function and should NEVER be called manually
    #[allow(clippy::type_complexity)]
    unsafe fn restore_impl<A: crate::host::CudaAlloc>(
        &mut self,
        alloc: crate::host::CombinedCudaAlloc<Self::CudaAllocationImpl, A>,
    ) -> rustacuda::error::CudaResult<A>;
}

#[doc(hidden)]
#[allow(clippy::module_name_repetitions)]
/// # Safety
///
/// This is an internal trait and should NEVER be implemented manually
pub unsafe trait CudaAsRustImpl: DeviceCopy {
    type RustRepresentationImpl: RustToCudaImpl<CudaRepresentationImpl = Self>;

    #[cfg(any(not(feature = "host"), doc))]
    /// # Safety
    ///
    /// This is an internal function and should NEVER be called manually
    unsafe fn as_rust_impl(this: &DeviceAccessible<Self>) -> Self::RustRepresentationImpl;
}
