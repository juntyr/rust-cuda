use core::mem::MaybeUninit;

use const_type_layout::{TypeGraphLayout, TypeLayout};

#[cfg(feature = "host")]
use rustacuda::error::CudaResult;

use crate::{
    lend::{CudaAsRust, RustToCuda, RustToCudaAsync, RustToCudaAsyncProxy, RustToCudaProxy},
    safety::PortableBitSemantics,
    utils::{adapter::RustToCudaWithPortableBitCopySemantics, ffi::DeviceAccessible},
};

#[cfg(feature = "host")]
use crate::alloc::{CombinedCudaAlloc, CudaAlloc};

#[doc(hidden)]
#[allow(clippy::module_name_repetitions)]
#[derive(TypeLayout)]
#[repr(C)]
pub struct OptionCudaRepresentation<T: CudaAsRust> {
    maybe: MaybeUninit<DeviceAccessible<T>>,
    present: bool,
}

unsafe impl<T: RustToCuda> RustToCuda for Option<T> {
    type CudaAllocation = Option<<T as RustToCuda>::CudaAllocation>;
    type CudaRepresentation = OptionCudaRepresentation<<T as RustToCuda>::CudaRepresentation>;

    #[cfg(feature = "host")]
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

unsafe impl<T: RustToCudaAsync> RustToCudaAsync for Option<T> {
    type CudaAllocationAsync = Option<<T as RustToCudaAsync>::CudaAllocationAsync>;

    #[cfg(feature = "host")]
    #[allow(clippy::type_complexity)]
    unsafe fn borrow_async<A: CudaAlloc>(
        &self,
        alloc: A,
        stream: &rustacuda::stream::Stream,
    ) -> CudaResult<(
        DeviceAccessible<Self::CudaRepresentation>,
        CombinedCudaAlloc<Self::CudaAllocationAsync, A>,
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
                let (cuda_repr, alloc) = value.borrow_async(alloc, stream)?;

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
    unsafe fn restore_async<A: CudaAlloc>(
        &mut self,
        alloc: CombinedCudaAlloc<Self::CudaAllocationAsync, A>,
        stream: &rustacuda::stream::Stream,
    ) -> CudaResult<A> {
        let (alloc_front, alloc_tail) = alloc.split();

        match (self, alloc_front) {
            (Some(value), Some(alloc_front)) => {
                value.restore_async(CombinedCudaAlloc::new(alloc_front, alloc_tail), stream)
            },
            _ => Ok(alloc_tail),
        }
    }
}

unsafe impl<T: CudaAsRust> CudaAsRust for OptionCudaRepresentation<T> {
    type RustRepresentation = Option<<T as CudaAsRust>::RustRepresentation>;

    #[cfg(feature = "device")]
    unsafe fn as_rust(this: &DeviceAccessible<Self>) -> Self::RustRepresentation {
        if this.present {
            Some(CudaAsRust::as_rust(this.maybe.assume_init_ref()))
        } else {
            None
        }
    }
}

impl<T: Copy + PortableBitSemantics + TypeGraphLayout> RustToCudaProxy<Option<T>>
    for Option<RustToCudaWithPortableBitCopySemantics<T>>
{
    fn from_ref(val: &Option<T>) -> &Self {
        // Safety: [`RustToCudaWithPortableBitCopySemantics`] is a transparent newtype
        unsafe { &*core::ptr::from_ref(val).cast() }
    }

    fn from_mut(val: &mut Option<T>) -> &mut Self {
        // Safety: [`RustToCudaWithPortableBitCopySemantics`] is a transparent newtype
        unsafe { &mut *core::ptr::from_mut(val).cast() }
    }

    fn into(self) -> Option<T> {
        self.map(RustToCudaWithPortableBitCopySemantics::into_inner)
    }
}

impl<T: Copy + PortableBitSemantics + TypeGraphLayout> RustToCudaAsyncProxy<Option<T>>
    for Option<RustToCudaWithPortableBitCopySemantics<T>>
{
    fn from_ref(val: &Option<T>) -> &Self {
        <Self as RustToCudaProxy<Option<T>>>::from_ref(val)
    }

    fn from_mut(val: &mut Option<T>) -> &mut Self {
        <Self as RustToCudaProxy<Option<T>>>::from_mut(val)
    }

    fn into(self) -> Option<T> {
        <Self as RustToCudaProxy<Option<T>>>::into(self)
    }
}
