use rustacuda_core::DeviceCopy;

use crate::common::{
    r#impl::{CudaAsRustImpl, RustToCudaImpl},
    DeviceAccessible, RustToCudaProxy,
};

use super::stack::{StackOnly, StackOnlyWrapper};

#[doc(hidden)]
#[allow(clippy::module_name_repetitions)]
#[repr(C, u8)]
pub enum OptionCudaRepresentation<T: CudaAsRustImpl> {
    None,
    Some(DeviceAccessible<T>),
}

// Safety: Since the CUDA representation of T is DeviceCopy,
//         the full enum is also DeviceCopy
unsafe impl<T: CudaAsRustImpl> DeviceCopy for OptionCudaRepresentation<T> {}

unsafe impl<T: RustToCudaImpl> RustToCudaImpl for Option<T> {
    #[cfg(feature = "host")]
    #[doc(cfg(feature = "host"))]
    type CudaAllocationImpl = Option<<T as RustToCudaImpl>::CudaAllocationImpl>;
    type CudaRepresentationImpl =
        OptionCudaRepresentation<<T as RustToCudaImpl>::CudaRepresentationImpl>;

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
                let (cuda_repr, alloc) = value.borrow_impl(alloc)?;

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
                value.restore_impl(crate::host::CombinedCudaAlloc::new(alloc_front, alloc_tail))
            },
            _ => Ok(alloc_tail),
        }
    }
}

unsafe impl<T: CudaAsRustImpl> CudaAsRustImpl for OptionCudaRepresentation<T> {
    type RustRepresentationImpl = Option<<T as CudaAsRustImpl>::RustRepresentationImpl>;

    #[cfg(any(not(feature = "host"), doc))]
    #[doc(cfg(not(feature = "host")))]
    unsafe fn as_rust_impl(this: &DeviceAccessible<Self>) -> Self::RustRepresentationImpl {
        match &**this {
            OptionCudaRepresentation::None => None,
            OptionCudaRepresentation::Some(value) => Some(CudaAsRustImpl::as_rust_impl(value)),
        }
    }
}

impl<T: StackOnly> RustToCudaProxy<Option<T>> for Option<StackOnlyWrapper<T>> {
    fn from_ref(val: &Option<T>) -> &Self {
        // Safety: StackOnlyWrapper is a transparent newtype
        unsafe { &*(val as *const Option<T>).cast() }
    }

    fn from_mut(val: &mut Option<T>) -> &mut Self {
        // Safety: StackOnlyWrapper is a transparent newtype
        unsafe { &mut *(val as *mut Option<T>).cast() }
    }

    fn into(self) -> Option<T> {
        self.map(StackOnlyWrapper::into_inner)
    }
}
