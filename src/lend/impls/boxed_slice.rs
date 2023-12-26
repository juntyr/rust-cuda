use core::marker::PhantomData;

use crate::{deps::alloc::boxed::Box, utils::ffi::DeviceOwnedPointer};

use const_type_layout::{TypeGraphLayout, TypeLayout};

#[cfg(feature = "host")]
use rustacuda::{error::CudaResult, memory::DeviceBuffer};

use crate::{
    lend::{CudaAsRust, RustToCuda},
    safety::PortableBitSemantics,
};

#[cfg(any(feature = "host", feature = "device"))]
use crate::utils::ffi::DeviceAccessible;

#[cfg(feature = "host")]
use crate::{
    alloc::{CombinedCudaAlloc, CudaAlloc},
    host::CudaDropWrapper,
    utils::device_copy::SafeDeviceCopyWrapper,
};

#[doc(hidden)]
#[allow(clippy::module_name_repetitions)]
#[derive(TypeLayout)]
#[repr(C)]
pub struct BoxedSliceCudaRepresentation<T: PortableBitSemantics + TypeGraphLayout> {
    data: DeviceOwnedPointer<T>,
    len: usize,
    _marker: PhantomData<T>,
}

unsafe impl<T: PortableBitSemantics + TypeGraphLayout> RustToCuda for Box<[T]> {
    #[cfg(all(feature = "host", not(doc)))]
    type CudaAllocation = crate::host::CudaDropWrapper<DeviceBuffer<SafeDeviceCopyWrapper<T>>>;
    #[cfg(any(not(feature = "host"), doc))]
    type CudaAllocation = crate::alloc::SomeCudaAlloc;
    type CudaRepresentation = BoxedSliceCudaRepresentation<T>;

    #[cfg(feature = "host")]
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
            DeviceAccessible::from(BoxedSliceCudaRepresentation {
                data: DeviceOwnedPointer(device_buffer.as_mut_ptr().cast()),
                len: device_buffer.len(),
                _marker: PhantomData::<T>,
            }),
            CombinedCudaAlloc::new(device_buffer, alloc),
        ))
    }

    #[cfg(feature = "host")]
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

unsafe impl<T: PortableBitSemantics + TypeGraphLayout> CudaAsRust
    for BoxedSliceCudaRepresentation<T>
{
    type RustRepresentation = Box<[T]>;

    #[cfg(feature = "device")]
    unsafe fn as_rust(this: &DeviceAccessible<Self>) -> Self::RustRepresentation {
        crate::deps::alloc::boxed::Box::from_raw(core::slice::from_raw_parts_mut(
            this.data.0,
            this.len,
        ))
    }
}
