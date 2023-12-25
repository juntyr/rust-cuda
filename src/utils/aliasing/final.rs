use const_type_layout::TypeLayout;
use r#final::Final;

use crate::{
    lend::{CudaAsRust, RustToCuda, RustToCudaAsync},
    utils::ffi::DeviceAccessible,
};

#[doc(hidden)]
#[repr(transparent)]
#[derive(TypeLayout)]
#[allow(clippy::module_name_repetitions)]
pub struct FinalCudaRepresentation<T: CudaAsRust>(DeviceAccessible<T>);

// Safety: If [`T`] is [`CudaAsRust`], then the newtype struct is [`DeviceCopy`]
unsafe impl<T: CudaAsRust> rustacuda_core::DeviceCopy for FinalCudaRepresentation<T> {}

unsafe impl<T: RustToCuda> RustToCuda for Final<T> {
    type CudaAllocation = T::CudaAllocation;
    type CudaRepresentation = FinalCudaRepresentation<T::CudaRepresentation>;

    #[cfg(feature = "host")]
    #[allow(clippy::type_complexity)]
    unsafe fn borrow<A: crate::alloc::CudaAlloc>(
        &self,
        alloc: A,
    ) -> rustacuda::error::CudaResult<(
        DeviceAccessible<Self::CudaRepresentation>,
        crate::alloc::CombinedCudaAlloc<Self::CudaAllocation, A>,
    )> {
        let (cuda_repr, alloc) = (**self).borrow(alloc)?;

        Ok((
            DeviceAccessible::from(FinalCudaRepresentation(cuda_repr)),
            alloc,
        ))
    }

    #[cfg(feature = "host")]
    unsafe fn restore<A: crate::alloc::CudaAlloc>(
        &mut self,
        alloc: crate::alloc::CombinedCudaAlloc<Self::CudaAllocation, A>,
    ) -> rustacuda::error::CudaResult<A> {
        // Safety: Final is a repr(transparent) newtype wrapper around T
        let inner: &mut T = &mut *(self as *mut Self).cast();

        inner.restore(alloc)
    }
}

unsafe impl<T: RustToCudaAsync> RustToCudaAsync for Final<T> {
    #[cfg(feature = "host")]
    #[allow(clippy::type_complexity)]
    unsafe fn borrow_async<A: crate::alloc::CudaAlloc>(
        &self,
        alloc: A,
        stream: &rustacuda::stream::Stream,
    ) -> rustacuda::error::CudaResult<(
        DeviceAccessible<Self::CudaRepresentation>,
        crate::alloc::CombinedCudaAlloc<Self::CudaAllocation, A>,
    )> {
        let (cuda_repr, alloc) = (**self).borrow_async(alloc, stream)?;

        Ok((
            DeviceAccessible::from(FinalCudaRepresentation(cuda_repr)),
            alloc,
        ))
    }

    #[cfg(feature = "host")]
    unsafe fn restore_async<A: crate::alloc::CudaAlloc>(
        &mut self,
        alloc: crate::alloc::CombinedCudaAlloc<Self::CudaAllocation, A>,
        stream: &rustacuda::stream::Stream,
    ) -> rustacuda::error::CudaResult<A> {
        // Safety: Final is a repr(transparent) newtype wrapper around T
        let inner: &mut T = &mut *(self as *mut Self).cast();

        inner.restore_async(alloc, stream)
    }
}

unsafe impl<T: CudaAsRust> CudaAsRust for FinalCudaRepresentation<T> {
    type RustRepresentation = Final<T::RustRepresentation>;

    #[cfg(feature = "device")]
    unsafe fn as_rust(this: &DeviceAccessible<Self>) -> Self::RustRepresentation {
        Final::new(CudaAsRust::as_rust(&this.0))
    }
}
