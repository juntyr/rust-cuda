use core::mem::MaybeUninit;

use const_type_layout::TypeGraphLayout;

use crate::{
    common::{CudaAsRust, DeviceAccessible, RustToCuda, RustToCudaProxy},
    safety::SafeDeviceCopy,
    utils::device_copy::SafeDeviceCopyWrapper,
};

#[cfg(feature = "host")]
use crate::{host::CombinedCudaAlloc, host::CudaAlloc, rustacuda::error::CudaResult};

#[doc(hidden)]
#[allow(clippy::module_name_repetitions)]
#[derive(TypeLayout)]
#[repr(C)]
pub struct OptionCudaRepresentation<T: CudaAsRust> {
    maybe: MaybeUninit<DeviceAccessible<T>>,
    present: bool,
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
                OptionCudaRepresentation {
                    maybe: MaybeUninit::uninit(),
                    present: false,
                },
                CombinedCudaAlloc::new(None, alloc),
            ),
            Some(value) => {
                let (cuda_repr, alloc) = value.borrow(alloc)?;

                let (alloc_front, alloc_tail) = alloc.split();

                (
                    OptionCudaRepresentation {
                        maybe: MaybeUninit::new(cuda_repr),
                        present: true,
                    },
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
        if this.present {
            Some(CudaAsRust::as_rust(this.maybe.assume_init_ref()))
        } else {
            None
        }
    }
}

impl<T: SafeDeviceCopy + ~const TypeGraphLayout> RustToCudaProxy<Option<T>>
    for Option<SafeDeviceCopyWrapper<T>>
{
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
