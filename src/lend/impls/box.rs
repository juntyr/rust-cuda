#[cfg(feature = "host")]
use std::mem::ManuallyDrop;

use const_type_layout::{TypeGraphLayout, TypeLayout};

#[cfg(feature = "host")]
use rustacuda::{error::CudaResult, memory::DeviceBox, memory::LockedBox};

use crate::{
    deps::alloc::boxed::Box,
    lend::{CudaAsRust, RustToCuda, RustToCudaAsync},
    safety::PortableBitSemantics,
    utils::ffi::DeviceOwnedPointer,
};

#[cfg(any(feature = "host", feature = "device"))]
use crate::utils::ffi::DeviceAccessible;

#[cfg(feature = "host")]
use crate::{
    alloc::{CombinedCudaAlloc, CudaAlloc},
    host::CudaDropWrapper,
    utils::adapter::DeviceCopyWithPortableBitSemantics,
    utils::r#async::Async,
};

#[doc(hidden)]
#[repr(transparent)]
#[derive(TypeLayout)]
#[allow(clippy::module_name_repetitions)]
pub struct BoxCudaRepresentation<T: PortableBitSemantics + TypeGraphLayout>(DeviceOwnedPointer<T>);

unsafe impl<T: PortableBitSemantics + TypeGraphLayout> RustToCuda for Box<T> {
    #[cfg(all(feature = "host", not(doc)))]
    type CudaAllocation = CudaDropWrapper<DeviceBox<DeviceCopyWithPortableBitSemantics<T>>>;
    #[cfg(any(not(feature = "host"), doc))]
    type CudaAllocation = crate::alloc::SomeCudaAlloc;
    type CudaRepresentation = BoxCudaRepresentation<T>;

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
            DeviceAccessible::from(BoxCudaRepresentation(DeviceOwnedPointer(
                device_box.as_device_ptr().as_raw_mut().cast(),
            ))),
            CombinedCudaAlloc::new(device_box, alloc),
        ))
    }

    #[cfg(feature = "host")]
    unsafe fn restore<A: CudaAlloc>(
        &mut self,
        alloc: CombinedCudaAlloc<Self::CudaAllocation, A>,
    ) -> CudaResult<A> {
        use rustacuda::memory::CopyDestination;

        let (alloc_front, alloc_tail) = alloc.split();

        alloc_front.copy_to(DeviceCopyWithPortableBitSemantics::from_mut(&mut **self))?;

        core::mem::drop(alloc_front);

        Ok(alloc_tail)
    }
}

unsafe impl<T: PortableBitSemantics + TypeGraphLayout> RustToCudaAsync for Box<T> {
    #[cfg(all(feature = "host", not(doc)))]
    type CudaAllocationAsync = CombinedCudaAlloc<
        CudaDropWrapper<LockedBox<DeviceCopyWithPortableBitSemantics<ManuallyDrop<T>>>>,
        CudaDropWrapper<DeviceBox<DeviceCopyWithPortableBitSemantics<ManuallyDrop<T>>>>,
    >;
    #[cfg(any(not(feature = "host"), doc))]
    type CudaAllocationAsync = crate::alloc::SomeCudaAlloc;

    #[cfg(feature = "host")]
    unsafe fn borrow_async<A: CudaAlloc>(
        &self,
        alloc: A,
        stream: &rustacuda::stream::Stream,
    ) -> rustacuda::error::CudaResult<(
        DeviceAccessible<Self::CudaRepresentation>,
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
        device_box.async_copy_from(&*locked_box, stream)?;

        Ok((
            DeviceAccessible::from(BoxCudaRepresentation(DeviceOwnedPointer(
                device_box.as_device_ptr().as_raw_mut().cast(),
            ))),
            CombinedCudaAlloc::new(CombinedCudaAlloc::new(locked_box, device_box), alloc),
        ))
    }

    #[cfg(feature = "host")]
    unsafe fn restore_async<'a, 'stream, A: CudaAlloc, O>(
        this: owning_ref::BoxRefMut<'a, O, Self>,
        alloc: CombinedCudaAlloc<Self::CudaAllocationAsync, A>,
        stream: &'stream rustacuda::stream::Stream,
    ) -> CudaResult<(
        Async<'stream, owning_ref::BoxRefMut<'a, O, Self>, Self::CudaAllocationAsync>,
        A,
    )> {
        use rustacuda::memory::AsyncCopyDestination;

        let (alloc_front, alloc_tail) = alloc.split();
        let (mut locked_box, device_box) = alloc_front.split();

        device_box.async_copy_to(&mut *locked_box, stream)?;

        let r#async = crate::utils::r#async::Async::pending(
            this,
            stream,
            CombinedCudaAlloc::new(locked_box, device_box),
            move |this, alloc| {
                let data: &mut T = &mut *this;
                let (locked_box, device_box) = alloc.split();

                std::mem::drop(device_box);
                // Safety: equivalent to *data = *locked_box since
                //         LockedBox<ManuallyDrop<T>> doesn't drop T
                unsafe {
                    std::ptr::copy_nonoverlapping(locked_box.as_ptr().cast::<T>(), data, 1);
                }
                std::mem::drop(locked_box);
                Ok(())
            },
        )?;

        Ok((r#async, alloc_tail))
    }
}

unsafe impl<T: PortableBitSemantics + TypeGraphLayout> CudaAsRust for BoxCudaRepresentation<T> {
    type RustRepresentation = Box<T>;

    #[cfg(feature = "device")]
    unsafe fn as_rust(this: &DeviceAccessible<Self>) -> Self::RustRepresentation {
        crate::deps::alloc::boxed::Box::from_raw(this.0 .0)
    }
}
