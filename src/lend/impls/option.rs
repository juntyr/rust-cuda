use core::mem::MaybeUninit;

use const_type_layout::{TypeGraphLayout, TypeLayout};

#[cfg(feature = "host")]
use rustacuda::error::CudaResult;

use crate::{
    lend::{CudaAsRust, RustToCuda, RustToCudaAsync, RustToCudaProxy},
    safety::PortableBitSemantics,
    utils::{adapter::RustToCudaWithPortableBitCopySemantics, ffi::DeviceAccessible},
};

#[cfg(feature = "host")]
use crate::{
    alloc::{CombinedCudaAlloc, CudaAlloc},
    utils::r#async::{Async, CompletionFnMut, NoCompletion},
};

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
    unsafe fn borrow_async<'stream, A: CudaAlloc>(
        &self,
        alloc: A,
        stream: crate::host::Stream<'stream>,
    ) -> CudaResult<(
        Async<'_, 'stream, DeviceAccessible<Self::CudaRepresentation>>,
        CombinedCudaAlloc<Self::CudaAllocationAsync, A>,
    )> {
        let (cuda_repr, alloc) = match self {
            None => (
                Async::ready(
                    DeviceAccessible::from(OptionCudaRepresentation {
                        maybe: MaybeUninit::uninit(),
                        present: false,
                    }),
                    stream,
                ),
                CombinedCudaAlloc::new(None, alloc),
            ),
            Some(value) => {
                let (cuda_repr, alloc) = value.borrow_async(alloc, stream)?;

                let (cuda_repr, completion) = unsafe { cuda_repr.unwrap_unchecked()? };

                let (alloc_front, alloc_tail) = alloc.split();
                let alloc = CombinedCudaAlloc::new(Some(alloc_front), alloc_tail);

                let option_cuda_repr = DeviceAccessible::from(OptionCudaRepresentation {
                    maybe: MaybeUninit::new(cuda_repr),
                    present: true,
                });

                let r#async = if matches!(completion, Some(NoCompletion)) {
                    Async::pending(option_cuda_repr, stream, NoCompletion)?
                } else {
                    Async::ready(option_cuda_repr, stream)
                };

                (r#async, alloc)
            },
        };

        Ok((cuda_repr, alloc))
    }

    #[cfg(feature = "host")]
    unsafe fn restore_async<'a, 'stream, A: CudaAlloc, O>(
        mut this: owning_ref::BoxRefMut<'a, O, Self>,
        alloc: CombinedCudaAlloc<Self::CudaAllocationAsync, A>,
        stream: crate::host::Stream<'stream>,
    ) -> CudaResult<(
        Async<'a, 'stream, owning_ref::BoxRefMut<'a, O, Self>, CompletionFnMut<'a, Self>>,
        A,
    )> {
        let (alloc_front, alloc_tail) = alloc.split();

        if let (Some(_), Some(alloc_front)) = (&mut *this, alloc_front) {
            let this_backup = unsafe { std::mem::ManuallyDrop::new(std::ptr::read(&this)) };

            #[allow(clippy::option_if_let_else)]
            let (r#async, alloc_tail) = RustToCudaAsync::restore_async(
                this.map_mut(|value| match value {
                    Some(value) => value,
                    None => unreachable!(), // TODO
                }),
                CombinedCudaAlloc::new(alloc_front, alloc_tail),
                stream,
            )?;

            let (value, on_completion) = unsafe { r#async.unwrap_unchecked()? };

            std::mem::forget(value);
            let this = std::mem::ManuallyDrop::into_inner(this_backup);

            if let Some(on_completion) = on_completion {
                let r#async = Async::<_, CompletionFnMut<'a, Self>>::pending(
                    this,
                    stream,
                    Box::new(|this: &mut Self| {
                        if let Some(value) = this {
                            on_completion(value)?;
                        }

                        Ok(())
                    }),
                )?;
                Ok((r#async, alloc_tail))
            } else {
                let r#async = Async::ready(this, stream);
                Ok((r#async, alloc_tail))
            }
        } else {
            let r#async = Async::ready(this, stream);
            Ok((r#async, alloc_tail))
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
