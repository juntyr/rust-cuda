pub use r#final::Final;

use rustacuda_core::DeviceCopy;

use crate::common::{CudaAsRust, DeviceAccessible, RustToCuda};

#[repr(transparent)]
#[doc(hidden)]
#[allow(clippy::module_name_repetitions)]
pub struct FinalCudaRepresentation<T: CudaAsRust>(DeviceAccessible<T>);

// Safety: If T is DeviceCopy, then the newtype struct can be DeviceCopy as well
unsafe impl<T: CudaAsRust + DeviceCopy> DeviceCopy for FinalCudaRepresentation<T> {}

unsafe impl<T: RustToCuda> RustToCuda for Final<T> {
    #[cfg(feature = "host")]
    #[doc(cfg(feature = "host"))]
    type CudaAllocation = T::CudaAllocation;
    type CudaRepresentation = FinalCudaRepresentation<T::CudaRepresentation>;

    #[cfg(feature = "host")]
    #[doc(cfg(feature = "host"))]
    #[allow(clippy::type_complexity)]
    unsafe fn borrow<A: crate::host::CudaAlloc>(
        &self,
        alloc: A,
    ) -> rustacuda::error::CudaResult<(
        DeviceAccessible<Self::CudaRepresentation>,
        crate::host::CombinedCudaAlloc<Self::CudaAllocation, A>,
    )> {
        let (cuda_repr, alloc) = (**self).borrow(alloc)?;

        Ok((
            DeviceAccessible::from(FinalCudaRepresentation(cuda_repr)),
            alloc,
        ))
    }

    #[cfg(feature = "host")]
    #[doc(cfg(feature = "host"))]
    unsafe fn restore<A: crate::host::CudaAlloc>(
        &mut self,
        alloc: crate::host::CombinedCudaAlloc<Self::CudaAllocation, A>,
    ) -> rustacuda::error::CudaResult<A> {
        // Safety: Final is a repr(transparent) newtype wrapper around T
        let inner: &mut T = &mut *(self as *mut Self).cast();

        inner.restore(alloc)
    }
}

unsafe impl<T: CudaAsRust> CudaAsRust for FinalCudaRepresentation<T> {
    type RustRepresentation = Final<T::RustRepresentation>;

    #[cfg(any(not(feature = "host"), doc))]
    #[doc(cfg(not(feature = "host")))]
    unsafe fn as_rust(this: &DeviceAccessible<Self>) -> Self::RustRepresentation {
        Final::new(CudaAsRust::as_rust(&this.0))
    }
}
