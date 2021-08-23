pub use r#final::Final;

use rustacuda_core::DeviceCopy;

use crate::common::{
    r#impl::{CudaAsRustImpl, RustToCudaImpl},
    CudaAsRust, DeviceAccessible, RustToCuda,
};

#[repr(transparent)]
#[doc(hidden)]
#[allow(clippy::module_name_repetitions)]
pub struct FinalCudaRepresentation<T: CudaAsRust>(DeviceAccessible<T>);

// Safety: If T is DeviceCopy, then the newtype struct can be DeviceCopy as well
unsafe impl<T: CudaAsRust + DeviceCopy> DeviceCopy for FinalCudaRepresentation<T> {}

unsafe impl<T: RustToCuda> RustToCudaImpl for Final<T> {
    #[cfg(feature = "host")]
    #[doc(cfg(feature = "host"))]
    type CudaAllocationImpl = T::CudaAllocation;
    type CudaRepresentationImpl = FinalCudaRepresentation<T::CudaRepresentation>;

    #[cfg(feature = "host")]
    #[doc(cfg(feature = "host"))]
    #[allow(clippy::type_complexity)]
    unsafe fn borrow_impl<A: crate::host::CudaAlloc>(
        &self,
        alloc: A,
    ) -> rustacuda::error::CudaResult<(
        DeviceAccessible<Self::CudaRepresentationImpl>,
        crate::host::CombinedCudaAlloc<Self::CudaAllocationImpl, A>,
    )> {
        let (cuda_repr, alloc) = (**self).borrow(alloc)?;

        Ok((
            DeviceAccessible::from(FinalCudaRepresentation(cuda_repr)),
            alloc,
        ))
    }

    #[cfg(feature = "host")]
    #[doc(cfg(feature = "host"))]
    unsafe fn restore_impl<A: crate::host::CudaAlloc>(
        &mut self,
        alloc: crate::host::CombinedCudaAlloc<Self::CudaAllocationImpl, A>,
    ) -> rustacuda::error::CudaResult<A> {
        // Safety: Final is a repr(transparent) newtype wrapper around T
        let inner: &mut T = &mut *(self as *mut Self).cast();

        inner.restore(alloc)
    }
}

unsafe impl<T: CudaAsRust> CudaAsRustImpl for FinalCudaRepresentation<T> {
    type RustRepresentationImpl = Final<T::RustRepresentation>;

    #[cfg(any(not(feature = "host"), doc))]
    #[doc(cfg(not(feature = "host")))]
    unsafe fn as_rust_impl(this: &DeviceAccessible<Self>) -> Self::RustRepresentationImpl {
        Final::new(CudaAsRust::as_rust(&this.0))
    }
}
