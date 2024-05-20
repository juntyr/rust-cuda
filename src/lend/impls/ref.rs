use core::marker::PhantomData;
#[cfg(feature = "host")]
use std::mem::ManuallyDrop;

use const_type_layout::{TypeGraphLayout, TypeLayout};

#[cfg(feature = "host")]
use rustacuda::{error::CudaResult, memory::DeviceBox, memory::LockedBox};

use crate::{
    lend::{CudaAsRust, RustToCuda, RustToCudaAsync},
    safety::PortableBitSemantics,
    utils::ffi::DeviceConstPointer,
};

#[cfg(any(feature = "host", feature = "device"))]
use crate::utils::ffi::DeviceAccessible;

#[cfg(feature = "host")]
use crate::{
    alloc::{CombinedCudaAlloc, CudaAlloc},
    host::CudaDropWrapper,
    utils::adapter::DeviceCopyWithPortableBitSemantics,
    utils::r#async::{Async, CompletionFnMut, NoCompletion},
};

#[doc(hidden)]
#[repr(transparent)]
#[derive(TypeLayout)]
#[allow(clippy::module_name_repetitions)]
pub struct RefCudaRepresentation<'a, T: 'a + PortableBitSemantics + TypeGraphLayout> {
    data: DeviceConstPointer<T>,
    _marker: PhantomData<&'a T>,
}

unsafe impl<'a, T: PortableBitSemantics + TypeGraphLayout> RustToCuda for &'a T {
    #[cfg(all(feature = "host", not(doc)))]
    type CudaAllocation = CudaDropWrapper<DeviceBox<DeviceCopyWithPortableBitSemantics<T>>>;
    #[cfg(any(not(feature = "host"), doc))]
    type CudaAllocation = crate::alloc::SomeCudaAlloc;
    type CudaRepresentation = RefCudaRepresentation<'a, T>;

    #[cfg(feature = "host")]
    #[allow(clippy::type_complexity)]
    unsafe fn borrow<A: CudaAlloc>(
        &self,
        alloc: A,
    ) -> CudaResult<(
        DeviceAccessible<Self::CudaRepresentation>,
        CombinedCudaAlloc<Self::CudaAllocation, A>,
    )> {
        let mut device_box = CudaDropWrapper::from(DeviceBox::new(
            DeviceCopyWithPortableBitSemantics::from_ref(&**self),
        )?);

        Ok((
            DeviceAccessible::from(RefCudaRepresentation {
                data: DeviceConstPointer(device_box.as_device_ptr().as_raw().cast()),
                _marker: PhantomData::<&'a T>,
            }),
            CombinedCudaAlloc::new(device_box, alloc),
        ))
    }

    #[cfg(feature = "host")]
    unsafe fn restore<A: CudaAlloc>(
        &mut self,
        alloc: CombinedCudaAlloc<Self::CudaAllocation, A>,
    ) -> CudaResult<A> {
        let (_alloc_front, alloc_tail) = alloc.split();
        Ok(alloc_tail)
    }
}

unsafe impl<'a, T: PortableBitSemantics + TypeGraphLayout> RustToCudaAsync for &'a T {
    #[cfg(all(feature = "host", not(doc)))]
    type CudaAllocationAsync = CombinedCudaAlloc<
        CudaDropWrapper<LockedBox<DeviceCopyWithPortableBitSemantics<ManuallyDrop<T>>>>,
        CudaDropWrapper<DeviceBox<DeviceCopyWithPortableBitSemantics<ManuallyDrop<T>>>>,
    >;
    #[cfg(any(not(feature = "host"), doc))]
    type CudaAllocationAsync = crate::alloc::SomeCudaAlloc;

    #[cfg(feature = "host")]
    unsafe fn borrow_async<'stream, A: CudaAlloc>(
        &self,
        alloc: A,
        stream: crate::host::Stream<'stream>,
    ) -> rustacuda::error::CudaResult<(
        Async<'_, 'stream, DeviceAccessible<Self::CudaRepresentation>>,
        CombinedCudaAlloc<Self::CudaAllocationAsync, A>,
    )> {
        use rustacuda::memory::AsyncCopyDestination;

        let locked_box = unsafe {
            let mut uninit = CudaDropWrapper::from(LockedBox::<
                DeviceCopyWithPortableBitSemantics<ManuallyDrop<T>>,
            >::uninitialized()?);
            std::ptr::copy_nonoverlapping(
                std::ptr::from_ref::<T>(&**self)
                    .cast::<DeviceCopyWithPortableBitSemantics<ManuallyDrop<T>>>(),
                uninit.as_mut_ptr(),
                1,
            );
            uninit
        };

        let mut device_box = CudaDropWrapper::from(DeviceBox::<
            DeviceCopyWithPortableBitSemantics<ManuallyDrop<T>>,
        >::uninitialized()?);
        device_box.async_copy_from(&*locked_box, &stream)?;

        Ok((
            Async::pending(
                DeviceAccessible::from(RefCudaRepresentation {
                    data: DeviceConstPointer(device_box.as_device_ptr().as_raw().cast()),
                    _marker: PhantomData::<&T>,
                }),
                stream,
                NoCompletion,
            )?,
            CombinedCudaAlloc::new(CombinedCudaAlloc::new(locked_box, device_box), alloc),
        ))
    }

    #[cfg(feature = "host")]
    unsafe fn restore_async<'b, 'stream, A: CudaAlloc, O>(
        this: owning_ref::BoxRefMut<'b, O, Self>,
        alloc: CombinedCudaAlloc<Self::CudaAllocationAsync, A>,
        stream: crate::host::Stream<'stream>,
    ) -> CudaResult<(
        Async<'b, 'stream, owning_ref::BoxRefMut<'b, O, Self>, CompletionFnMut<'b, Self>>,
        A,
    )> {
        let (_alloc_front, alloc_tail) = alloc.split();
        let r#async = Async::ready(this, stream);
        Ok((r#async, alloc_tail))
    }
}

unsafe impl<'a, T: PortableBitSemantics + TypeGraphLayout> CudaAsRust
    for RefCudaRepresentation<'a, T>
{
    type RustRepresentation = &'a T;

    #[cfg(feature = "device")]
    unsafe fn as_rust(this: &DeviceAccessible<Self>) -> Self::RustRepresentation {
        &*this.data.0
    }
}
