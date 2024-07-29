use core::marker::PhantomData;
#[cfg(feature = "host")]
use std::mem::ManuallyDrop;

use crate::{deps::alloc::boxed::Box, lend::RustToCudaAsync, utils::ffi::DeviceOwnedPointer};

use const_type_layout::{TypeGraphLayout, TypeLayout};

#[cfg(feature = "host")]
use rustacuda::{error::CudaResult, memory::DeviceBuffer, memory::LockedBuffer};

use crate::{
    lend::{CudaAsRust, RustToCuda},
    safety::PortableBitSemantics,
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
#[expect(clippy::module_name_repetitions)]
#[derive(TypeLayout)]
#[repr(C)]
pub struct BoxedSliceCudaRepresentation<T: PortableBitSemantics + TypeGraphLayout> {
    data: DeviceOwnedPointer<T>,
    len: usize,
    _marker: PhantomData<T>,
}

unsafe impl<T: PortableBitSemantics + TypeGraphLayout> RustToCuda for Box<[T]> {
    #[cfg(all(feature = "host", not(doc)))]
    type CudaAllocation =
        crate::host::CudaDropWrapper<DeviceBuffer<DeviceCopyWithPortableBitSemantics<T>>>;
    #[cfg(any(not(feature = "host"), doc))]
    type CudaAllocation = crate::alloc::SomeCudaAlloc;
    type CudaRepresentation = BoxedSliceCudaRepresentation<T>;

    #[cfg(feature = "host")]
    unsafe fn borrow<A: CudaAlloc>(
        &self,
        alloc: A,
    ) -> CudaResult<(
        DeviceAccessible<Self::CudaRepresentation>,
        CombinedCudaAlloc<Self::CudaAllocation, A>,
    )> {
        let mut device_buffer = CudaDropWrapper::from(DeviceBuffer::from_slice(
            DeviceCopyWithPortableBitSemantics::from_slice(self),
        )?);

        Ok((
            DeviceAccessible::from(BoxedSliceCudaRepresentation {
                data: DeviceOwnedPointer(device_buffer.as_mut_ptr().cast()),
                len: device_buffer.len(),
                _marker: PhantomData::<T>,
            }),
            CombinedCudaAlloc::new(device_buffer, alloc),
        ))
    }

    #[cfg(feature = "host")]
    unsafe fn restore<A: CudaAlloc>(
        &mut self,
        alloc: CombinedCudaAlloc<Self::CudaAllocation, A>,
    ) -> CudaResult<A> {
        use rustacuda::memory::CopyDestination;

        let (alloc_front, alloc_tail) = alloc.split();

        alloc_front.copy_to(DeviceCopyWithPortableBitSemantics::from_mut_slice(self))?;

        core::mem::drop(alloc_front);

        Ok(alloc_tail)
    }
}

unsafe impl<T: PortableBitSemantics + TypeGraphLayout> RustToCudaAsync for Box<[T]> {
    #[cfg(all(feature = "host", not(doc)))]
    type CudaAllocationAsync = CombinedCudaAlloc<
        CudaDropWrapper<LockedBuffer<DeviceCopyWithPortableBitSemantics<ManuallyDrop<T>>>>,
        CudaDropWrapper<DeviceBuffer<DeviceCopyWithPortableBitSemantics<ManuallyDrop<T>>>>,
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

        let locked_buffer = unsafe {
            let mut uninit = CudaDropWrapper::from(LockedBuffer::<
                DeviceCopyWithPortableBitSemantics<ManuallyDrop<T>>,
            >::uninitialized(self.len())?);
            std::ptr::copy_nonoverlapping(
                self.as_ref()
                    .as_ptr()
                    .cast::<DeviceCopyWithPortableBitSemantics<ManuallyDrop<T>>>(),
                uninit.as_mut_ptr(),
                self.len(),
            );
            uninit
        };

        let mut device_buffer = CudaDropWrapper::from(DeviceBuffer::<
            DeviceCopyWithPortableBitSemantics<ManuallyDrop<T>>,
        >::uninitialized(self.len())?);
        device_buffer.async_copy_from(&*locked_buffer, &stream)?;

        Ok((
            Async::pending(
                DeviceAccessible::from(BoxedSliceCudaRepresentation {
                    data: DeviceOwnedPointer(device_buffer.as_mut_ptr().cast()),
                    len: device_buffer.len(),
                    _marker: PhantomData::<T>,
                }),
                stream,
                NoCompletion,
            )?,
            CombinedCudaAlloc::new(CombinedCudaAlloc::new(locked_buffer, device_buffer), alloc),
        ))
    }

    #[cfg(feature = "host")]
    unsafe fn restore_async<'a, 'stream, A: CudaAlloc, O>(
        this: owning_ref::BoxRefMut<'a, O, Self>,
        alloc: CombinedCudaAlloc<Self::CudaAllocationAsync, A>,
        stream: crate::host::Stream<'stream>,
    ) -> CudaResult<(
        Async<'a, 'stream, owning_ref::BoxRefMut<'a, O, Self>, CompletionFnMut<'a, Self>>,
        A,
    )> {
        use rustacuda::memory::AsyncCopyDestination;

        let (alloc_front, alloc_tail) = alloc.split();
        let (mut locked_buffer, device_buffer) = alloc_front.split();

        device_buffer.async_copy_to(&mut *locked_buffer, &stream)?;

        let r#async = crate::utils::r#async::Async::<_, CompletionFnMut<'a, Self>>::pending(
            this,
            stream,
            Box::new(move |this: &mut Self| {
                let data: &mut [T] = &mut *this;
                std::mem::drop(device_buffer);
                // Safety: equivalent to data.copy_from_slice(&*locked_buffer)
                //         since LockedBox<ManuallyDrop<T>> doesn't drop T
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        locked_buffer.as_ptr().cast::<T>(),
                        data.as_mut_ptr(),
                        data.len(),
                    );
                }
                std::mem::drop(locked_buffer);
                Ok(())
            }),
        )?;

        Ok((r#async, alloc_tail))
    }
}

unsafe impl<T: PortableBitSemantics + TypeGraphLayout> CudaAsRust
    for BoxedSliceCudaRepresentation<T>
{
    type RustRepresentation = Box<[T]>;

    #[cfg(feature = "device")]
    unsafe fn as_rust(this: &DeviceAccessible<Self>) -> Self::RustRepresentation {
        crate::deps::alloc::boxed::Box::from_raw(core::slice::from_raw_parts_mut(
            this.data.0,
            this.len,
        ))
    }
}
