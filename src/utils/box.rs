use alloc::boxed::Box;

use rustacuda_core::DeviceCopy;

use crate::common::{
    r#impl::{CudaAsRustImpl, RustToCudaImpl},
    DeviceAccessible,
};

#[repr(transparent)]
#[doc(hidden)]
#[allow(clippy::module_name_repetitions)]
pub struct BoxCudaRepresentation<T: DeviceCopy>(*mut T);

// Safety: This repr(C) struct only contains a device-owned pointer
unsafe impl<T: DeviceCopy> DeviceCopy for BoxCudaRepresentation<T> {}

unsafe impl<T: DeviceCopy> RustToCudaImpl for Box<T> {
    #[cfg(feature = "host")]
    #[doc(cfg(feature = "host"))]
    type CudaAllocationImpl = crate::host::CudaDropWrapper<crate::rustacuda::memory::DeviceBox<T>>;
    type CudaRepresentationImpl = BoxCudaRepresentation<T>;

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
    unsafe fn restore_impl<A: crate::host::CudaAlloc>(
        &mut self,
        alloc: crate::host::CombinedCudaAlloc<Self::CudaAllocationImpl, A>,
    ) -> rustacuda::error::CudaResult<A> {
        use rustacuda::prelude::CopyDestination;

        let (alloc_front, alloc_tail) = alloc.split();

        alloc_front.copy_to(&mut **self)?;

        core::mem::drop(alloc_front);

        Ok(alloc_tail)
    }
}

unsafe impl<T: DeviceCopy> CudaAsRustImpl for BoxCudaRepresentation<T> {
    type RustRepresentationImpl = Box<T>;

    #[cfg(any(not(feature = "host"), doc))]
    #[doc(cfg(not(feature = "host")))]
    unsafe fn as_rust_impl(this: &DeviceAccessible<Self>) -> Self::RustRepresentationImpl {
        alloc::boxed::Box::from_raw(this.0)
    }
}
