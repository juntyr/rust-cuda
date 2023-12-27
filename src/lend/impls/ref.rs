use core::marker::PhantomData;

use const_type_layout::{TypeGraphLayout, TypeLayout};

#[cfg(feature = "host")]
use rustacuda::{error::CudaResult, memory::DeviceBox};

use crate::{
    lend::{CudaAsRust, RustToCuda},
    safety::PortableBitSemantics,
    utils::ffi::DeviceConstPointer,
};

#[cfg(any(feature = "host", feature = "device"))]
use crate::utils::ffi::DeviceAccessible;

#[cfg(feature = "host")]
use crate::{
    alloc::{CombinedCudaAlloc, CudaAlloc},
    host::CudaDropWrapper,
    utils::adapter::DeviceCopyWithPortableBitSemantics,
};

#[doc(hidden)]
#[repr(transparent)]
#[derive(TypeLayout)]
#[allow(clippy::module_name_repetitions)]
pub struct RefCudaRepresentation<'a, T: 'a + PortableBitSemantics + TypeGraphLayout> {
    data: DeviceConstPointer<T>,
    _marker: PhantomData<&'a T>,
}

unsafe impl<'a, T: PortableBitSemantics + TypeGraphLayout> RustToCuda for &'a T {
    #[cfg(all(feature = "host", not(doc)))]
    type CudaAllocation = CudaDropWrapper<DeviceBox<DeviceCopyWithPortableBitSemantics<T>>>;
    #[cfg(any(not(feature = "host"), doc))]
    type CudaAllocation = crate::alloc::SomeCudaAlloc;
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
        let mut device_box = CudaDropWrapper::from(DeviceBox::new(
            DeviceCopyWithPortableBitSemantics::from_ref(&**self),
        )?);

        Ok((
            DeviceAccessible::from(RefCudaRepresentation {
                data: DeviceConstPointer(device_box.as_device_ptr().as_raw().cast()),
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

unsafe impl<'a, T: PortableBitSemantics + TypeGraphLayout> CudaAsRust
    for RefCudaRepresentation<'a, T>
{
    type RustRepresentation = &'a T;

    #[cfg(feature = "device")]
    unsafe fn as_rust(this: &DeviceAccessible<Self>) -> Self::RustRepresentation {
        &*this.data.0
    }
}
