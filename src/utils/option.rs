use rustacuda_core::DeviceCopy;

use crate::common::{
    r#impl::{CudaAsRustImpl, RustToCudaImpl},
    DeviceAccessible, RustToCuda,
};

#[cfg(not(feature = "host"))]
use crate::common::CudaAsRust;

#[doc(hidden)]
#[allow(clippy::module_name_repetitions)]
#[repr(C, u8)]
pub enum OptionCudaRepresentation<T: RustToCuda> {
    None,
    Some(DeviceAccessible<<T as RustToCuda>::CudaRepresentation>),
}

// Safety: Since the CUDA representation of T is DeviceCopy,
//         the full enum is also DeviceCopy
unsafe impl<T: RustToCuda> DeviceCopy for OptionCudaRepresentation<T> {}

unsafe impl<T: RustToCuda> RustToCudaImpl for Option<T> {
    #[cfg(feature = "host")]
    #[doc(cfg(feature = "host"))]
    type CudaAllocationImpl = Option<<T as RustToCuda>::CudaAllocation>;
    type CudaRepresentationImpl = OptionCudaRepresentation<T>;

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
        let (cuda_repr, alloc) = match self {
            None => (
                OptionCudaRepresentation::None,
                crate::host::CombinedCudaAlloc::new(None, alloc),
            ),
            Some(value) => {
                let (cuda_repr, alloc) = value.borrow(alloc)?;

                let (alloc_front, alloc_tail) = alloc.split();

                (
                    OptionCudaRepresentation::Some(cuda_repr),
                    crate::host::CombinedCudaAlloc::new(Some(alloc_front), alloc_tail),
                )
            },
        };

        Ok((DeviceAccessible::from(cuda_repr), alloc))
    }

    #[cfg(feature = "host")]
    #[doc(cfg(feature = "host"))]
    unsafe fn restore_impl<A: crate::host::CudaAlloc>(
        &mut self,
        alloc: crate::host::CombinedCudaAlloc<Self::CudaAllocationImpl, A>,
    ) -> rustacuda::error::CudaResult<A> {
        let (alloc_front, alloc_tail) = alloc.split();

        match (self, alloc_front) {
            (Some(value), Some(alloc_front)) => {
                value.restore(crate::host::CombinedCudaAlloc::new(alloc_front, alloc_tail))
            },
            _ => Ok(alloc_tail),
        }
    }
}

unsafe impl<T: RustToCuda> CudaAsRustImpl for OptionCudaRepresentation<T> {
    type RustRepresentationImpl = Option<T>;

    #[cfg(any(not(feature = "host"), doc))]
    #[doc(cfg(not(feature = "host")))]
    unsafe fn as_rust_impl(this: &DeviceAccessible<Self>) -> Self::RustRepresentationImpl {
        match &**this {
            OptionCudaRepresentation::None => None,
            OptionCudaRepresentation::Some(value) => Some(CudaAsRust::as_rust(value)),
        }
    }
}
