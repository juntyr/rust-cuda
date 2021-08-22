use alloc::boxed::Box;

use rustacuda_core::DeviceCopy;

use crate::common::{CudaAsRust, CudaAsRustCore, DeviceAccessible, RustToCuda, RustToCudaAlloc, RustToCudaCore};

use super::stack::DeviceCopy2;

#[repr(transparent)]
#[doc(hidden)]
#[allow(clippy::module_name_repetitions)]
pub struct BoxCudaRepresentation<T: DeviceCopy>(*mut T);

// Safety: This repr(C) struct only contains a device-owned pointer
unsafe impl<T: DeviceCopy> DeviceCopy for BoxCudaRepresentation<T> {}

impl<T: DeviceCopy> RustToCudaCore for Box<T> {
    type CudaRepresentation = BoxCudaRepresentation<T>;
}

#[cfg(feature = "host")]
#[doc(cfg(feature = "host"))]
impl<T: DeviceCopy> RustToCudaAlloc for Box<T> {
    type CudaAllocation = crate::host::CudaDropWrapper<crate::rustacuda::memory::DeviceBox<T>>;
}

unsafe impl<T: DeviceCopy> RustToCuda for Box<T> {
    //#[cfg(feature = "host")]
    //#[doc(cfg(feature = "host"))]
    //type CudaAllocation = crate::host::CudaDropWrapper<crate::rustacuda::memory::DeviceBox<T>>;
    //type CudaRepresentation = BoxCudaRepresentation<T>;

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
        let mut device_box =
            crate::host::CudaDropWrapper::from(rustacuda::memory::DeviceBox::new(&**self)?);

        Ok((
            DeviceAccessible::from(BoxCudaRepresentation(
                device_box.as_device_ptr().as_raw_mut(),
            )),
            crate::host::CombinedCudaAlloc::new(device_box, alloc),
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

        alloc_front.copy_to(&mut **self)?;

        core::mem::drop(alloc_front);

        Ok(alloc_tail)
    }
}

impl<T: DeviceCopy> CudaAsRustCore for BoxCudaRepresentation<T> {
    type RustRepresentation = Box<T>;
}

unsafe impl<T: DeviceCopy> CudaAsRust for BoxCudaRepresentation<T> {
    //type RustRepresentation = Box<T>;

    #[cfg(any(not(feature = "host"), doc))]
    #[doc(cfg(not(feature = "host")))]
    unsafe fn as_rust(this: &DeviceAccessible<Self>) -> Self::RustRepresentation {
        alloc::boxed::Box::from_raw(this.0)
    }
}
