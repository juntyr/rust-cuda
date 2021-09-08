use crate::{
    common::{CudaAsRust, DeviceAccessible, RustToCuda, RustToCudaProxy},
    utils::{SafeDeviceCopy, SafeDeviceCopyWrapper},
};

#[cfg(feature = "host")]
use crate::{host::CombinedCudaAlloc, host::CudaAlloc, rustacuda::error::CudaResult};

#[doc(hidden)]
#[allow(clippy::module_name_repetitions)]
#[repr(C, u8)]
pub enum OptionCudaRepresentation<T: CudaAsRust> {
    None,
    Some(DeviceAccessible<T>),
}

// Safety: Since the CUDA representation of T is DeviceCopy,
//         the full enum is also DeviceCopy
unsafe impl<T: CudaAsRust> rustacuda_core::DeviceCopy for OptionCudaRepresentation<T> {}

unsafe impl<T: RustToCuda> RustToCuda for Option<T> {
    #[cfg(feature = "host")]
    #[doc(cfg(feature = "host"))]
    type CudaAllocation = Option<<T as RustToCuda>::CudaAllocation>;
    type CudaRepresentation = OptionCudaRepresentation<<T as RustToCuda>::CudaRepresentation>;

    #[cfg(feature = "host")]
    #[doc(cfg(feature = "host"))]
    #[allow(clippy::type_complexity)]
    unsafe fn borrow<A: CudaAlloc>(
        &self,
        alloc: A,
    ) -> CudaResult<(
        DeviceAccessible<Self::CudaRepresentation>,
        CombinedCudaAlloc<Self::CudaAllocation, A>,
    )> {
        let (cuda_repr, alloc) = match self {
            None => (
                OptionCudaRepresentation::None,
                CombinedCudaAlloc::new(None, alloc),
            ),
            Some(value) => {
                let (cuda_repr, alloc) = value.borrow(alloc)?;

                let (alloc_front, alloc_tail) = alloc.split();

                (
                    OptionCudaRepresentation::Some(cuda_repr),
                    CombinedCudaAlloc::new(Some(alloc_front), alloc_tail),
                )
            },
        };

        Ok((DeviceAccessible::from(cuda_repr), alloc))
    }

    #[cfg(feature = "host")]
    #[doc(cfg(feature = "host"))]
    unsafe fn restore<A: CudaAlloc>(
        &mut self,
        alloc: CombinedCudaAlloc<Self::CudaAllocation, A>,
    ) -> CudaResult<A> {
        let (alloc_front, alloc_tail) = alloc.split();

        match (self, alloc_front) {
            (Some(value), Some(alloc_front)) => {
                value.restore(CombinedCudaAlloc::new(alloc_front, alloc_tail))
            },
            _ => Ok(alloc_tail),
        }
    }
}

unsafe impl<T: CudaAsRust> CudaAsRust for OptionCudaRepresentation<T> {
    type RustRepresentation = Option<<T as CudaAsRust>::RustRepresentation>;

    #[cfg(any(not(feature = "host"), doc))]
    #[doc(cfg(not(feature = "host")))]
    unsafe fn as_rust(this: &DeviceAccessible<Self>) -> Self::RustRepresentation {
        match &**this {
            OptionCudaRepresentation::None => None,
            OptionCudaRepresentation::Some(value) => Some(CudaAsRust::as_rust(value)),
        }
    }
}

impl<T: SafeDeviceCopy> RustToCudaProxy<Option<T>> for Option<SafeDeviceCopyWrapper<T>> {
    fn from_ref(val: &Option<T>) -> &Self {
        // Safety: `SafeDeviceCopyWrapper` is a transparent newtype
        unsafe { &*(val as *const Option<T>).cast() }
    }

    fn from_mut(val: &mut Option<T>) -> &mut Self {
        // Safety: `SafeDeviceCopyWrapper` is a transparent newtype
        unsafe { &mut *(val as *mut Option<T>).cast() }
    }

    fn into(self) -> Option<T> {
        self.map(SafeDeviceCopyWrapper::into_inner)
    }
}
