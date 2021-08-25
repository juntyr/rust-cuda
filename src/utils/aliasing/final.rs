pub use r#final::Final;

use rustacuda_core::DeviceCopy;

use crate::common::{
    r#impl::{CudaAsRustImpl, RustToCudaImpl},
    DeviceAccessible,
};

#[repr(transparent)]
#[doc(hidden)]
#[allow(clippy::module_name_repetitions)]
pub struct FinalCudaRepresentation<T: CudaAsRustImpl>(DeviceAccessible<T>);

// Safety: If T is DeviceCopy, then the newtype struct can be DeviceCopy as well
unsafe impl<T: CudaAsRustImpl> DeviceCopy for FinalCudaRepresentation<T> {}

unsafe impl<T: RustToCudaImpl> RustToCudaImpl for Final<T> {
    #[cfg(feature = "host")]
    #[doc(cfg(feature = "host"))]
    type CudaAllocationImpl = T::CudaAllocationImpl;
    type CudaRepresentationImpl = FinalCudaRepresentation<T::CudaRepresentationImpl>;

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
        let (cuda_repr, alloc) = (**self).borrow_impl(alloc)?;

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

        inner.restore_impl(alloc)
    }
}

unsafe impl<T: CudaAsRustImpl> CudaAsRustImpl for FinalCudaRepresentation<T> {
    type RustRepresentationImpl = Final<T::RustRepresentationImpl>;

    #[cfg(any(not(feature = "host"), doc))]
    #[doc(cfg(not(feature = "host")))]
    unsafe fn as_rust_impl(this: &DeviceAccessible<Self>) -> Self::RustRepresentationImpl {
        Final::new(CudaAsRustImpl::as_rust_impl(&this.0))
    }
}
