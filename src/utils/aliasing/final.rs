pub use r#final::Final;

use rustacuda_core::DeviceCopy;

use crate::common::{CudaAsRust, RustToCuda};

#[repr(transparent)]
#[doc(hidden)]
#[allow(clippy::module_name_repetitions)]
pub struct FinalCudaRepresentation<T: CudaAsRust>(T);

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
    unsafe fn borrow_mut<A: crate::host::CudaAlloc>(
        &mut self,
        alloc: A,
    ) -> rustacuda::error::CudaResult<(
        Self::CudaRepresentation,
        crate::host::CombinedCudaAlloc<Self::CudaAllocation, A>,
    )> {
        let inner: &T = &*self;

        // This change in mutability is only safe iff the mutability is not
        //  exposed in `Self::CudaRepresentation`
        #[allow(clippy::cast_ref_to_mut)]
        let inner_mut: &mut T = &mut *(inner as *const T as *mut T);

        let (cuda_repr, alloc) = inner_mut.borrow_mut(alloc)?;

        Ok((FinalCudaRepresentation(cuda_repr), alloc))
    }

    #[cfg(feature = "host")]
    #[doc(cfg(feature = "host"))]
    unsafe fn un_borrow_mut<A: crate::host::CudaAlloc>(
        &mut self,
        cuda_repr: Self::CudaRepresentation,
        alloc: crate::host::CombinedCudaAlloc<Self::CudaAllocation, A>,
    ) -> rustacuda::error::CudaResult<A> {
        let inner: &T = &*self;

        // This change in mutability is only safe iff the mutability is not
        //  exposed in `Self::CudaRepresentation`
        #[allow(clippy::cast_ref_to_mut)]
        let inner_mut: &mut T = &mut *(inner as *const T as *mut T);

        inner_mut.un_borrow_mut(cuda_repr.0, alloc)
    }
}

unsafe impl<T: CudaAsRust> CudaAsRust for FinalCudaRepresentation<T> {
    type RustRepresentation = Final<T::RustRepresentation>;

    #[cfg(any(not(feature = "host"), doc))]
    #[doc(cfg(not(feature = "host")))]
    unsafe fn as_rust(&mut self) -> Self::RustRepresentation {
        Final::new(self.0.as_rust())
    }
}
