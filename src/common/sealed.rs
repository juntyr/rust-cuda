use rustacuda_core::DeviceCopy;

use super::DeviceAccessible;

#[doc(hidden)]
#[allow(clippy::module_name_repetitions)]
pub trait RustToCudaSealed {
    #[cfg(feature = "host")]
    type CudaAllocationSealed: crate::host::CudaAlloc;
    type CudaRepresentationSealed: CudaAsRustSealed<RustRepresentationSealed = Self>;

    #[cfg(feature = "host")]
    #[allow(clippy::type_complexity)]
    unsafe fn borrow_sealed<A: crate::host::CudaAlloc>(
        &self,
        alloc: A,
    ) -> rustacuda::error::CudaResult<(
        DeviceAccessible<Self::CudaRepresentationSealed>,
        crate::host::CombinedCudaAlloc<Self::CudaAllocationSealed, A>,
    )>;

    #[cfg(feature = "host")]
    unsafe fn restore_sealed<A: crate::host::CudaAlloc>(
        &mut self,
        alloc: crate::host::CombinedCudaAlloc<Self::CudaAllocationSealed, A>,
    ) -> rustacuda::error::CudaResult<A>;
}

#[doc(hidden)]
#[allow(clippy::module_name_repetitions)]
pub trait CudaAsRustSealed: DeviceCopy {
    type RustRepresentationSealed: RustToCudaSealed<CudaRepresentationSealed = Self>;

    #[cfg(not(feature = "host"))]
    unsafe fn as_rust_sealed(this: &DeviceAccessible<Self>) -> Self::RustRepresentationSealed;
}
