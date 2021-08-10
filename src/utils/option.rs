use rustacuda_core::DeviceCopy;

use crate::{
    common::{CudaAsRust, RustToCuda},
    utils::stack::StackOnly,
};

#[doc(hidden)]
#[allow(clippy::module_name_repetitions)]
#[derive(Debug, Clone)]
#[repr(C, u8)]
pub enum OptionCudaRepresentation<T: StackOnly + Clone> {
    None,
    Some(T),
}

// Safety: Any type that is fully on the stack without any references
//         to the heap can be safely copied to the GPU
unsafe impl<T: StackOnly + Clone> DeviceCopy for OptionCudaRepresentation<T> {}

unsafe impl<T: StackOnly + Clone> RustToCuda for Option<T> {
    #[cfg(feature = "host")]
    #[doc(cfg(feature = "host"))]
    type CudaAllocation = crate::host::NullCudaAlloc;
    type CudaRepresentation = OptionCudaRepresentation<T>;

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
        let alloc = crate::host::CombinedCudaAlloc::new(crate::host::NullCudaAlloc, alloc);

        match self {
            None => Ok((OptionCudaRepresentation::None, alloc)),
            Some(value) => Ok((OptionCudaRepresentation::Some(value.clone()), alloc)),
        }
    }

    #[cfg(feature = "host")]
    #[doc(cfg(feature = "host"))]
    unsafe fn un_borrow_mut<A: crate::host::CudaAlloc>(
        &mut self,
        _cuda_repr: Self::CudaRepresentation,
        alloc: crate::host::CombinedCudaAlloc<Self::CudaAllocation, A>,
    ) -> rustacuda::error::CudaResult<A> {
        let (_alloc_front, alloc_tail) = alloc.split();

        Ok(alloc_tail)
    }
}

unsafe impl<T: StackOnly + Clone> CudaAsRust for OptionCudaRepresentation<T> {
    type RustRepresentation = Option<T>;

    #[cfg(any(not(feature = "host"), doc))]
    #[doc(cfg(not(feature = "host")))]
    unsafe fn as_rust(&mut self) -> Self::RustRepresentation {
        match self {
            OptionCudaRepresentation::None => None,
            OptionCudaRepresentation::Some(value) => Some(value.clone()),
        }
    }
}
