use rustacuda_core::DeviceCopy;

use super::{
    sealed::{CudaAsRustSealed, RustToCudaSealed},
    DeviceAccessible,
};

/// This is an internal trait which cannot be implemented directly.
/// Instead, any type which is either `StackOnly` or derives `RustToCudaAsRust`
///  automatically implements `RustToCuda` as well.
pub trait RustToCuda: RustToCudaSealed {
    #[cfg(feature = "host")]
    #[doc(cfg(feature = "host"))]
    type CudaAllocation: crate::host::CudaAlloc;
    type CudaRepresentation: CudaAsRust<RustRepresentation = Self>;

    #[cfg(feature = "host")]
    #[doc(cfg(feature = "host"))]
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
    unsafe fn borrow<A: crate::host::CudaAlloc>(
        &self,
        alloc: A,
    ) -> rustacuda::error::CudaResult<(
        DeviceAccessible<Self::CudaRepresentation>,
        crate::host::CombinedCudaAlloc<Self::CudaAllocation, A>,
    )>;

    #[cfg(feature = "host")]
    #[doc(cfg(feature = "host"))]
    /// # Errors
    ///
    /// Returns a `rustacuda::errors::CudaError` iff an error occurs inside CUDA
    ///
    /// # Safety
    ///
    /// This is an internal function and should NEVER be called manually
    #[allow(clippy::type_complexity)]
    unsafe fn restore<A: crate::host::CudaAlloc>(
        &mut self,
        alloc: crate::host::CombinedCudaAlloc<Self::CudaAllocation, A>,
    ) -> rustacuda::error::CudaResult<A>;
}

/// This is an internal trait which cannot be implemented directly.
/// Instead, any type which is either `StackOnly` or derives `RustToCudaAsRust`
///  automatically gets a `RustToCuda::CudaRepresentation` that implements
///  `CudaAsRust` as well.
pub trait CudaAsRust: CudaAsRustSealed + DeviceCopy {
    type RustRepresentation: RustToCuda<CudaRepresentation = Self>;

    #[cfg(any(not(feature = "host"), doc))]
    #[doc(cfg(not(feature = "host")))]
    /// # Safety
    ///
    /// This is an internal function and should NEVER be called manually
    unsafe fn as_rust(this: &DeviceAccessible<Self>) -> Self::RustRepresentation;
}

impl<T: RustToCudaSealed> RustToCuda for T {
    #[cfg(feature = "host")]
    type CudaAllocation = Self::CudaAllocationSealed;
    type CudaRepresentation = Self::CudaRepresentationSealed;

    #[cfg(feature = "host")]
    #[allow(clippy::type_complexity)]
    unsafe fn borrow<A: crate::host::CudaAlloc>(
        &self,
        alloc: A,
    ) -> rustacuda::error::CudaResult<(
        DeviceAccessible<Self::CudaRepresentation>,
        crate::host::CombinedCudaAlloc<Self::CudaAllocation, A>,
    )> {
        self.borrow_sealed(alloc)
    }

    #[cfg(feature = "host")]
    unsafe fn restore<A: crate::host::CudaAlloc>(
        &mut self,
        alloc: crate::host::CombinedCudaAlloc<Self::CudaAllocation, A>,
    ) -> rustacuda::error::CudaResult<A> {
        self.restore_sealed(alloc)
    }
}

impl<T: CudaAsRustSealed> CudaAsRust for T {
    type RustRepresentation = Self::RustRepresentationSealed;

    #[cfg(any(not(feature = "host"), doc))]
    unsafe fn as_rust(this: &DeviceAccessible<Self>) -> Self::RustRepresentation {
        CudaAsRustSealed::as_rust_sealed(this)
    }
}
