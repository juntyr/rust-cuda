use core::sync::atomic::AtomicUsize;
#[cfg(feature = "host")]
use std::mem::ManuallyDrop;

use const_type_layout::{TypeGraphLayout, TypeLayout};

#[cfg(feature = "host")]
use rustacuda::{error::CudaResult, memory::DeviceBox, memory::LockedBox};

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
#[repr(transparent)]
#[derive(TypeLayout)]
#[allow(clippy::module_name_repetitions)]
pub struct ArcCudaRepresentation<T: PortableBitSemantics + TypeGraphLayout>(
    DeviceOwnedPointer<_ArcInner<T>>,
);

// must be kept in sync (hehe)
#[doc(hidden)]
#[derive(TypeLayout)]
#[repr(C)]
pub struct _ArcInner<T: ?Sized> {
    strong: AtomicUsize,
    weak: AtomicUsize,
    data: T,
}

unsafe impl<T: PortableBitSemantics + TypeGraphLayout> RustToCuda for Arc<T> {
    #[cfg(all(feature = "host", not(doc)))]
    type CudaAllocation =
        CudaDropWrapper<DeviceBox<DeviceCopyWithPortableBitSemantics<_ArcInner<T>>>>;
    #[cfg(any(not(feature = "host"), doc))]
    type CudaAllocation = crate::alloc::SomeCudaAlloc;
    type CudaRepresentation = ArcCudaRepresentation<T>;

    #[cfg(feature = "host")]
    #[allow(clippy::type_complexity)]
    unsafe fn borrow<A: CudaAlloc>(
        &self,
        alloc: A,
    ) -> CudaResult<(
        DeviceAccessible<Self::CudaRepresentation>,
        CombinedCudaAlloc<Self::CudaAllocation, A>,
    )> {
        let inner = ManuallyDrop::new(_ArcInner {
            strong: AtomicUsize::new(1),
            weak: AtomicUsize::new(1),
            data: std::ptr::read(&**self),
        });

        let mut device_box = CudaDropWrapper::from(DeviceBox::new(
            DeviceCopyWithPortableBitSemantics::from_ref(&*inner),
        )?);

        Ok((
            DeviceAccessible::from(ArcCudaRepresentation(DeviceOwnedPointer(
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
        let (_alloc_front, alloc_tail) = alloc.split();
        Ok(alloc_tail)
    }
}

unsafe impl<T: PortableBitSemantics + TypeGraphLayout> RustToCudaAsync for Arc<T> {
    #[cfg(all(feature = "host", not(doc)))]
    type CudaAllocationAsync = CombinedCudaAlloc<
        CudaDropWrapper<LockedBox<DeviceCopyWithPortableBitSemantics<ManuallyDrop<_ArcInner<T>>>>>,
        CudaDropWrapper<DeviceBox<DeviceCopyWithPortableBitSemantics<ManuallyDrop<_ArcInner<T>>>>>,
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
            let inner = ManuallyDrop::new(_ArcInner {
                strong: AtomicUsize::new(1),
                weak: AtomicUsize::new(1),
                data: std::ptr::read(&**self),
            });

            let mut uninit = CudaDropWrapper::from(LockedBox::<
                DeviceCopyWithPortableBitSemantics<ManuallyDrop<_ArcInner<T>>>,
            >::uninitialized()?);
            std::ptr::copy_nonoverlapping(
                std::ptr::from_ref(DeviceCopyWithPortableBitSemantics::from_ref(&inner)),
                uninit.as_mut_ptr(),
                1,
            );

            uninit
        };

        let mut device_box = CudaDropWrapper::from(DeviceBox::<
            DeviceCopyWithPortableBitSemantics<ManuallyDrop<_ArcInner<T>>>,
        >::uninitialized()?);
        device_box.async_copy_from(&*locked_box, &stream)?;

        Ok((
            Async::pending(
                DeviceAccessible::from(ArcCudaRepresentation(DeviceOwnedPointer(
                    device_box.as_device_ptr().as_raw_mut().cast(),
                ))),
                stream,
                NoCompletion,
            )?,
            CombinedCudaAlloc::new(CombinedCudaAlloc::new(locked_box, device_box), alloc),
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

unsafe impl<T: PortableBitSemantics + TypeGraphLayout> CudaAsRust for ArcCudaRepresentation<T> {
    type RustRepresentation = Arc<T>;

    #[cfg(feature = "device")]
    unsafe fn as_rust(this: &DeviceAccessible<Self>) -> Self::RustRepresentation {
        crate::deps::alloc::sync::Arc::from_raw(core::ptr::addr_of!((*(this.0 .0)).data))
    }
}
