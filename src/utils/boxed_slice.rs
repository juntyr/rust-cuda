use alloc::boxed::Box;

use rustacuda_core::DeviceCopy;

use crate::common::{
    r#impl::{CudaAsRustImpl, RustToCudaImpl},
    DeviceAccessible,
};

#[doc(hidden)]
#[allow(clippy::module_name_repetitions)]
#[derive(Debug)]
#[repr(C)]
pub struct BoxedSliceCudaRepresentation<T: DeviceCopy>(*mut T, usize);

// Safety: This repr(C) struct only contains a device-owned pointer
unsafe impl<T: DeviceCopy> DeviceCopy for BoxedSliceCudaRepresentation<T> {}

unsafe impl<T: DeviceCopy> RustToCudaImpl for Box<[T]> {
    #[cfg(feature = "host")]
    #[doc(cfg(feature = "host"))]
    type CudaAllocationImpl =
        crate::host::CudaDropWrapper<crate::rustacuda::memory::DeviceBuffer<T>>;
    type CudaRepresentationImpl = BoxedSliceCudaRepresentation<T>;

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
        let mut device_buffer =
            crate::host::CudaDropWrapper::from(rustacuda::memory::DeviceBuffer::from_slice(self)?);

        Ok((
            DeviceAccessible::from(BoxedSliceCudaRepresentation(
                device_buffer.as_mut_ptr(),
                device_buffer.len(),
            )),
            crate::host::CombinedCudaAlloc::new(device_buffer, alloc),
        ))
    }

    #[cfg(feature = "host")]
    #[doc(cfg(feature = "host"))]
    unsafe fn restore_impl<A: crate::host::CudaAlloc>(
        &mut self,
        alloc: crate::host::CombinedCudaAlloc<Self::CudaAllocationImpl, A>,
    ) -> rustacuda::error::CudaResult<A> {
        use rustacuda::prelude::CopyDestination;

        let (alloc_front, alloc_tail) = alloc.split();

        alloc_front.copy_to(self)?;

        core::mem::drop(alloc_front);

        Ok(alloc_tail)
    }
}

unsafe impl<T: DeviceCopy> CudaAsRustImpl for BoxedSliceCudaRepresentation<T> {
    type RustRepresentationImpl = Box<[T]>;

    #[cfg(any(not(feature = "host"), doc))]
    #[doc(cfg(not(feature = "host")))]
    unsafe fn as_rust_impl(this: &DeviceAccessible<Self>) -> Self::RustRepresentationImpl {
        alloc::boxed::Box::from_raw(core::slice::from_raw_parts_mut(this.0, this.1))
    }
}
