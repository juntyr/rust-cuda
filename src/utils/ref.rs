use core::marker::PhantomData;

use const_type_layout::{TypeGraphLayout, TypeLayout};

#[cfg(feature = "host")]
use rustacuda::{error::CudaResult, memory::DeviceBox};

use crate::{
    common::{CudaAsRust, RustToCuda},
    safety::SafeDeviceCopy,
};

#[cfg(any(feature = "host", feature = "device"))]
use crate::common::DeviceAccessible;

#[cfg(feature = "host")]
use crate::{
    common::{CombinedCudaAlloc, CudaAlloc},
    host::CudaDropWrapper,
    utils::device_copy::SafeDeviceCopyWrapper,
};

#[doc(hidden)]
#[repr(transparent)]
#[derive(TypeLayout)]
#[allow(clippy::module_name_repetitions)]
pub struct RefCudaRepresentation<'a, T: 'a + SafeDeviceCopy + TypeGraphLayout> {
    data: *const T,
    _marker: PhantomData<&'a T>,
}

// Safety: This repr(C) struct only contains a device-owned pointer
unsafe impl<'a, T: SafeDeviceCopy + TypeGraphLayout> rustacuda_core::DeviceCopy
    for RefCudaRepresentation<'a, T>
{
}

unsafe impl<'a, T: SafeDeviceCopy + TypeGraphLayout> RustToCuda for &'a T {
    #[cfg(all(feature = "host", not(doc)))]
    type CudaAllocation = crate::host::CudaDropWrapper<DeviceBox<SafeDeviceCopyWrapper<T>>>;
    #[cfg(any(not(feature = "host"), doc))]
    type CudaAllocation = crate::common::SomeCudaAlloc;
    type CudaRepresentation = RefCudaRepresentation<'a, T>;

    #[cfg(feature = "host")]
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
            DeviceAccessible::from(RefCudaRepresentation {
                data: device_box.as_device_ptr().as_raw().cast(),
                _marker: PhantomData::<&'a T>,
            }),
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

unsafe impl<'a, T: SafeDeviceCopy + TypeGraphLayout> CudaAsRust for RefCudaRepresentation<'a, T> {
    type RustRepresentation = &'a T;

    #[cfg(feature = "device")]
    unsafe fn as_rust(this: &DeviceAccessible<Self>) -> Self::RustRepresentation {
        &*this.data
    }
}
