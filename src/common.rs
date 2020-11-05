use rustacuda_core::DeviceCopy;

/// # Safety
/// This is an internal trait and should ONLY be derived automatically using `#[derive(RustToCuda)]`
pub unsafe trait RustToCuda {
    type CudaRepresentation: DeviceCopy + CudaAsRust<RustRepresentation = Self>;

    #[cfg(feature = "host")]
    type CudaAllocation: crate::host::CudaAlloc;

    #[cfg(feature = "host")]
    /// # Errors
    /// Returns a `rustacuda::errors::CudaError` iff an error occurs inside CUDA
    /// # Safety
    /// This is an internal function and should NEVER be called manually
    #[allow(clippy::type_complexity)]
    unsafe fn borrow<A: crate::host::CudaAlloc>(
        &self,
        alloc: A,
    ) -> rustacuda::error::CudaResult<(
        Self::CudaRepresentation,
        crate::host::CombinedCudaAlloc<Self::CudaAllocation, A>,
    )>;
}

/// # Safety
/// This is an internal trait and should NEVER be implemented manually
pub unsafe trait CudaAsRust {
    type RustRepresentation: RustToCuda<CudaRepresentation = Self>;

    #[cfg(not(feature = "host"))]
    /// # Safety
    /// This is an internal function and should NEVER be called manually
    unsafe fn as_rust(&self) -> Self::RustRepresentation;
}
