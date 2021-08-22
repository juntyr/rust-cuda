use alloc::boxed::Box;

use rustacuda_core::DeviceCopy;

use crate::common::{CudaAsRust, DeviceAccessible, RustToCuda};

#[doc(hidden)]
#[allow(clippy::module_name_repetitions)]
#[derive(Debug)]
#[repr(C)]
pub struct BoxedSliceCudaRepresentation<T: DeviceCopy>(*mut T, usize);

// Safety: This repr(C) struct only contains a device-owned pointer
unsafe impl<T: DeviceCopy> DeviceCopy for BoxedSliceCudaRepresentation<T> {}

unsafe impl<T: DeviceCopy> RustToCuda for Box<[T]> {
    #[cfg(feature = "host")]
    #[doc(cfg(feature = "host"))]
    type CudaAllocation = crate::host::CudaDropWrapper<crate::rustacuda::memory::DeviceBuffer<T>>;
    type CudaRepresentation = BoxedSliceCudaRepresentation<T>;

    #[cfg(feature = "host")]
    #[doc(cfg(feature = "host"))]
    #[allow(clippy::type_complexity)]
    unsafe fn borrow<A: crate::host::CudaAlloc>(
        &self,
        alloc: A,
    ) -> rustacuda::error::CudaResult<(
        DeviceAccessible<Self::CudaRepresentation>,
        crate::host::CombinedCudaAlloc<Self::CudaAllocation, A>,
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
    unsafe fn restore<A: crate::host::CudaAlloc>(
        &mut self,
        alloc: crate::host::CombinedCudaAlloc<Self::CudaAllocation, A>,
    ) -> rustacuda::error::CudaResult<A> {
        use rustacuda::prelude::CopyDestination;

        let (alloc_front, alloc_tail) = alloc.split();

        alloc_front.copy_to(self)?;

        core::mem::drop(alloc_front);

        Ok(alloc_tail)
    }
}

unsafe impl<T: DeviceCopy> CudaAsRust for BoxedSliceCudaRepresentation<T> {
    type RustRepresentation = Box<[T]>;

    #[cfg(any(not(feature = "host"), doc))]
    #[doc(cfg(not(feature = "host")))]
    unsafe fn as_rust(this: &DeviceAccessible<Self>) -> Self::RustRepresentation {
        alloc::boxed::Box::from_raw(core::slice::from_raw_parts_mut(this.0, this.1))
    }
}
