use alloc::boxed::Box;

use rustacuda_core::DeviceCopy;

use crate::common::{CudaAsRust, RustToCuda};

#[repr(transparent)]
#[doc(hidden)]
#[allow(clippy::module_name_repetitions)]
pub struct BoxCudaRepresentation<T: DeviceCopy + Sized>(*mut T);

// Safety: This repr(C) struct only contains a device-owned pointer
unsafe impl<T: DeviceCopy + Sized> DeviceCopy for BoxCudaRepresentation<T> {}

unsafe impl<T: DeviceCopy + Sized> RustToCuda for Box<T> {
    #[cfg(feature = "host")]
    #[doc(cfg(feature = "host"))]
    type CudaAllocation = crate::host::CudaDropWrapper<crate::rustacuda::memory::DeviceBox<T>>;
    type CudaRepresentation = BoxCudaRepresentation<T>;

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
        let mut device_box =
            crate::host::CudaDropWrapper::from(rustacuda::memory::DeviceBox::new(&**self)?);

        Ok((
            BoxCudaRepresentation(device_box.as_device_ptr().as_raw_mut()),
            crate::host::CombinedCudaAlloc::new(device_box, alloc),
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

        alloc_front.copy_to(&mut **self)?;

        core::mem::drop(alloc_front);

        Ok(alloc_tail)
    }
}

unsafe impl<T: DeviceCopy + Sized> CudaAsRust for BoxCudaRepresentation<T> {
    type RustRepresentation = Box<T>;

    #[cfg(any(not(feature = "host"), doc))]
    #[doc(cfg(not(feature = "host")))]
    unsafe fn as_rust(&mut self) -> Self::RustRepresentation {
        alloc::boxed::Box::from_raw(self.0)
    }
}
