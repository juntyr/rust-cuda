use core::sync::atomic::AtomicUsize;
#[cfg(feature = "host")]
use std::mem::{ManuallyDrop, MaybeUninit};

use const_type_layout::{TypeGraphLayout, TypeLayout};

#[cfg(feature = "host")]
use cust::{
    error::CudaResult,
    memory::LockedBuffer,
    memory::{DeviceBox, DeviceBuffer},
};
use cust_core::DeviceCopy;

use crate::{
    deps::alloc::sync::Arc,
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
    utils::r#async::CompletionFnMut,
    utils::r#async::NoCompletion,
};

#[doc(hidden)]
#[expect(clippy::module_name_repetitions)]
#[derive(TypeLayout)]
#[repr(C)]
pub struct ArcedSliceCudaRepresentation<T: PortableBitSemantics + TypeGraphLayout> {
    data: DeviceOwnedPointer<_ArcInner<T>>,
    len: usize,
}

// must be kept in sync (hehe)
#[doc(hidden)]
#[derive(TypeLayout)]
#[repr(C)]
pub struct _ArcInner<T: ?Sized> {
    strong: AtomicUsize,
    weak: AtomicUsize,
    data: T,
}

#[derive(Copy, Clone)]
#[repr(C)]
struct _ArcInnerHeader {
    strong: _AtomicUsize,
    weak: _AtomicUsize,
}

#[derive(Copy, Clone)]
#[repr(C, align(8))]
struct _AtomicUsize {
    v: usize,
}

unsafe impl DeviceCopy for _ArcInnerHeader {}

unsafe impl<T: PortableBitSemantics + TypeGraphLayout> RustToCuda for Arc<[T]> {
    #[cfg(all(feature = "host", not(doc)))]
    type CudaAllocation = CudaDropWrapper<DeviceBuffer<DeviceCopyWithPortableBitSemantics<T>>>;
    #[cfg(any(not(feature = "host"), doc))]
    type CudaAllocation = crate::alloc::SomeCudaAlloc;
    type CudaRepresentation = ArcedSliceCudaRepresentation<T>;

    #[cfg(feature = "host")]
    unsafe fn borrow<A: CudaAlloc>(
        &self,
        alloc: A,
    ) -> CudaResult<(
        DeviceAccessible<Self::CudaRepresentation>,
        CombinedCudaAlloc<Self::CudaAllocation, A>,
    )> {
        use cust::memory::{CopyDestination, DevicePointer, DeviceSlice};

        let data_ptr: *const T = std::ptr::from_ref(&**self).as_ptr();
        let offset = std::mem::offset_of!(_ArcInner<[T; 42]>, data);
        let arc_ptr: *const _ArcInner<[T; 42]> = data_ptr.byte_sub(offset).cast();

        let header_len = (offset + (std::mem::align_of::<T>() - 1)) / std::mem::align_of::<T>();

        let mut device_buffer = CudaDropWrapper::from(DeviceBuffer::<
            DeviceCopyWithPortableBitSemantics<T>,
        >::uninitialized(
            header_len + self.len()
        )?);
        let (header, buffer): (&mut DeviceSlice<_>, &mut DeviceSlice<_>) =
            device_buffer.split_at_mut(header_len);
        buffer.copy_from(std::slice::from_raw_parts(self.as_ptr().cast(), self.len()))?;
        let header = DeviceSlice::from_raw_parts_mut(
            DevicePointer::wrap(header.as_mut_ptr().cast::<u8>()),
            header.len() * std::mem::size_of::<T>(),
        );
        let (_, header) = header.split_at_mut(header.len() - offset);
        let (header, _) = header.split_at_mut(std::mem::size_of::<_ArcInnerHeader>());
        #[expect(clippy::cast_ptr_alignment)]
        let mut header: ManuallyDrop<DeviceBox<_ArcInnerHeader>> = ManuallyDrop::new(
            DeviceBox::from_raw(header.as_mut_ptr().cast::<_ArcInnerHeader>()),
        );
        header.copy_from(&*arc_ptr.cast::<_ArcInnerHeader>())?;

        Ok((
            DeviceAccessible::from(ArcedSliceCudaRepresentation {
                data: DeviceOwnedPointer(header.as_device_ptr().as_mut_ptr().cast()),
                len: self.len(),
            }),
            CombinedCudaAlloc::new(device_buffer, alloc),
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

unsafe impl<T: PortableBitSemantics + TypeGraphLayout> RustToCudaAsync for Arc<[T]> {
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
    ) -> cust::error::CudaResult<(
        Async<'_, 'stream, DeviceAccessible<Self::CudaRepresentation>>,
        CombinedCudaAlloc<Self::CudaAllocationAsync, A>,
    )> {
        use cust::memory::AsyncCopyDestination;

        let data_ptr: *const T = std::ptr::from_ref(&**self).as_ptr();
        let offset = std::mem::offset_of!(_ArcInner<[T; 42]>, data);
        let arc_ptr: *const _ArcInner<[T; 42]> = data_ptr.byte_sub(offset).cast();

        let header_len = (offset + (std::mem::align_of::<T>() - 1)) / std::mem::align_of::<T>();

        let locked_buffer = unsafe {
            let mut locked_buffer =
                CudaDropWrapper::from(LockedBuffer::<
                    DeviceCopyWithPortableBitSemantics<ManuallyDrop<T>>,
                >::uninitialized(header_len + self.len())?);
            let locked_buffer_slice: &mut [MaybeUninit<T>] = std::slice::from_raw_parts_mut(
                locked_buffer.as_mut_slice().as_mut_ptr().cast(),
                locked_buffer.as_slice().len(),
            );
            let (header, buffer) = locked_buffer_slice.split_at_mut(header_len);
            std::ptr::copy_nonoverlapping(self.as_ptr().cast(), buffer.as_mut_ptr(), self.len());
            let header = std::slice::from_raw_parts_mut(
                header.as_mut_ptr().cast::<MaybeUninit<u8>>(),
                header.len() * std::mem::size_of::<T>(),
            );
            let (_, header) = header.split_at_mut(header.len() - offset);
            let (header, _) = header.split_at_mut(std::mem::size_of::<_ArcInnerHeader>());
            let header: *mut MaybeUninit<_ArcInnerHeader> = header.as_mut_ptr().cast();
            std::ptr::copy_nonoverlapping(
                &*arc_ptr.cast::<MaybeUninit<_ArcInnerHeader>>(),
                header,
                1,
            );

            locked_buffer
        };

        let mut device_buffer =
            CudaDropWrapper::from(DeviceBuffer::<
                DeviceCopyWithPortableBitSemantics<ManuallyDrop<T>>,
            >::uninitialized(locked_buffer.len())?);
        device_buffer.async_copy_from(&*locked_buffer, &stream)?;

        Ok((
            Async::pending(
                DeviceAccessible::from(ArcedSliceCudaRepresentation {
                    data: DeviceOwnedPointer(
                        device_buffer
                            .as_device_ptr()
                            .as_mut_ptr()
                            .byte_add(header_len * std::mem::size_of::<T>() - offset)
                            .cast(),
                    ),
                    len: self.len(),
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
        let (_alloc_front, alloc_tail) = alloc.split();
        let r#async = Async::ready(this, stream);
        Ok((r#async, alloc_tail))
    }
}

unsafe impl<T: PortableBitSemantics + TypeGraphLayout> CudaAsRust
    for ArcedSliceCudaRepresentation<T>
{
    type RustRepresentation = Arc<[T]>;

    #[cfg(feature = "device")]
    unsafe fn as_rust(this: &DeviceAccessible<Self>) -> Self::RustRepresentation {
        crate::deps::alloc::sync::Arc::from_raw(core::ptr::slice_from_raw_parts(
            core::ptr::addr_of!((*(this.data.0)).data),
            this.len,
        ))
    }
}
