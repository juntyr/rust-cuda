use alloc::boxed::Box;

use const_type_layout::TypeGraphLayout;

use crate::{
    common::{CudaAsRust, DeviceAccessible, RustToCuda},
    safety::SafeDeviceCopy,
};

#[cfg(feature = "host")]
use crate::{
    host::CombinedCudaAlloc, host::CudaAlloc, host::CudaDropWrapper, rustacuda::error::CudaResult,
    rustacuda::memory::DeviceBox, utils::device_copy::SafeDeviceCopyWrapper,
};

#[doc(hidden)]
#[repr(transparent)]
#[derive(TypeLayout)]
#[allow(clippy::module_name_repetitions)]
#[layout(bound = "T: SafeDeviceCopy + ~const TypeGraphLayout")]
pub struct BoxCudaRepresentation<T: SafeDeviceCopy + ~const TypeGraphLayout>(*mut T);

// Safety: This repr(C) struct only contains a device-owned pointer
unsafe impl<T: SafeDeviceCopy + ~const TypeGraphLayout> rustacuda_core::DeviceCopy
    for BoxCudaRepresentation<T>
{
}

unsafe impl<T: SafeDeviceCopy + ~const TypeGraphLayout> RustToCuda for Box<T> {
    #[cfg(feature = "host")]
    #[doc(cfg(feature = "host"))]
    type CudaAllocation = CudaDropWrapper<DeviceBox<SafeDeviceCopyWrapper<T>>>;
    type CudaRepresentation = BoxCudaRepresentation<T>;

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
        let mut device_box =
            CudaDropWrapper::from(DeviceBox::new(SafeDeviceCopyWrapper::from_ref(&**self))?);

        Ok((
            DeviceAccessible::from(BoxCudaRepresentation(
                device_box.as_device_ptr().as_raw_mut().cast(),
            )),
            CombinedCudaAlloc::new(device_box, alloc),
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

        alloc_front.copy_to(SafeDeviceCopyWrapper::from_mut(&mut **self))?;

        core::mem::drop(alloc_front);

        Ok(alloc_tail)
    }
}

unsafe impl<T: SafeDeviceCopy + ~const TypeGraphLayout> CudaAsRust for BoxCudaRepresentation<T> {
    type RustRepresentation = Box<T>;

    #[cfg(any(not(feature = "host"), doc))]
    #[doc(cfg(not(feature = "host")))]
    unsafe fn as_rust(this: &DeviceAccessible<Self>) -> Self::RustRepresentation {
        alloc::boxed::Box::from_raw(this.0)
    }
}
