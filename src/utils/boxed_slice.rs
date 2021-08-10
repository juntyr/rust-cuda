use alloc::boxed::Box;

use rustacuda_core::DeviceCopy;

use crate::common::{CudaAsRust, RustToCuda};

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
    unsafe fn borrow_mut<A: crate::host::CudaAlloc>(
        &mut self,
        alloc: A,
    ) -> rustacuda::error::CudaResult<(
        Self::CudaRepresentation,
        crate::host::CombinedCudaAlloc<Self::CudaAllocation, A>,
    )> {
        let mut device_buffer =
            crate::host::CudaDropWrapper::from(rustacuda::memory::DeviceBuffer::from_slice(self)?);

        Ok((
            BoxedSliceCudaRepresentation(device_buffer.as_mut_ptr(), device_buffer.len()),
            crate::host::CombinedCudaAlloc::new(device_buffer, alloc),
        ))
    }

    #[cfg(feature = "host")]
    #[doc(cfg(feature = "host"))]
    unsafe fn un_borrow_mut<A: crate::host::CudaAlloc>(
        &mut self,
        _cuda_repr: Self::CudaRepresentation,
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
    unsafe fn as_rust(&mut self) -> Self::RustRepresentation {
        alloc::boxed::Box::from_raw(core::slice::from_raw_parts_mut(self.0, self.1))
    }
}
