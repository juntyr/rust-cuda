use alloc::boxed::Box;

use const_type_layout::TypeLayout;

use crate::{
    common::{CudaAsRust, DeviceAccessible, RustToCuda},
    memory::SafeDeviceCopy,
};

#[cfg(feature = "host")]
use crate::{
    host::CombinedCudaAlloc, host::CudaAlloc, host::CudaDropWrapper, rustacuda::error::CudaResult,
    rustacuda::memory::DeviceBuffer, utils::device_copy::SafeDeviceCopyWrapper,
};

#[doc(hidden)]
#[allow(clippy::module_name_repetitions)]
#[derive(Debug, TypeLayout)]
#[repr(C)]
pub struct BoxedSliceCudaRepresentation<T: SafeDeviceCopy + TypeLayout>(*mut T, usize);

// Safety: This repr(C) struct only contains a device-owned pointer
unsafe impl<T: SafeDeviceCopy + TypeLayout> rustacuda_core::DeviceCopy
    for BoxedSliceCudaRepresentation<T>
{
}

unsafe impl<T: SafeDeviceCopy + TypeLayout> RustToCuda for Box<[T]> {
    #[cfg(feature = "host")]
    #[doc(cfg(feature = "host"))]
    type CudaAllocation = CudaDropWrapper<DeviceBuffer<SafeDeviceCopyWrapper<T>>>;
    type CudaRepresentation = BoxedSliceCudaRepresentation<T>;

    #[cfg(feature = "host")]
    #[doc(cfg(feature = "host"))]
    #[allow(clippy::type_complexity)]
    unsafe fn borrow<A: CudaAlloc>(
        &self,
        alloc: A,
    ) -> CudaResult<(
        DeviceAccessible<Self::CudaRepresentation>,
        CombinedCudaAlloc<Self::CudaAllocation, A>,
    )> {
        let mut device_buffer = CudaDropWrapper::from(DeviceBuffer::from_slice(
            SafeDeviceCopyWrapper::from_slice(self),
        )?);

        Ok((
            DeviceAccessible::from(BoxedSliceCudaRepresentation(
                device_buffer.as_mut_ptr().cast(),
                device_buffer.len(),
            )),
            CombinedCudaAlloc::new(device_buffer, alloc),
        ))
    }

    #[cfg(feature = "host")]
    #[doc(cfg(feature = "host"))]
    unsafe fn restore<A: CudaAlloc>(
        &mut self,
        alloc: CombinedCudaAlloc<Self::CudaAllocation, A>,
    ) -> CudaResult<A> {
        use rustacuda::memory::CopyDestination;

        let (alloc_front, alloc_tail) = alloc.split();

        alloc_front.copy_to(SafeDeviceCopyWrapper::from_mut_slice(self))?;

        core::mem::drop(alloc_front);

        Ok(alloc_tail)
    }
}

unsafe impl<T: SafeDeviceCopy + TypeLayout> CudaAsRust for BoxedSliceCudaRepresentation<T> {
    type RustRepresentation = Box<[T]>;

    #[cfg(any(not(feature = "host"), doc))]
    #[doc(cfg(not(feature = "host")))]
    unsafe fn as_rust(this: &DeviceAccessible<Self>) -> Self::RustRepresentation {
        alloc::boxed::Box::from_raw(core::slice::from_raw_parts_mut(this.0, this.1))
    }
}
