use core::marker::PhantomData;

use const_type_layout::{TypeGraphLayout, TypeLayout};

#[cfg(feature = "host")]
use rustacuda::{error::CudaResult, memory::DeviceBuffer};

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
#[allow(clippy::module_name_repetitions)]
#[derive(TypeLayout)]
#[repr(C)]
pub struct SliceRefCudaRepresentation<'a, T: 'a + PortableBitSemantics + TypeGraphLayout> {
    data: DeviceConstPointer<T>,
    len: usize,
    _marker: PhantomData<&'a [T]>,
}

unsafe impl<'a, T: PortableBitSemantics + TypeGraphLayout> RustToCuda for &'a [T] {
    #[cfg(all(feature = "host", not(doc)))]
    type CudaAllocation =
        crate::host::CudaDropWrapper<DeviceBuffer<DeviceCopyWithPortableBitSemantics<T>>>;
    #[cfg(any(not(feature = "host"), doc))]
    type CudaAllocation = crate::alloc::SomeCudaAlloc;
    type CudaRepresentation = SliceRefCudaRepresentation<'a, T>;

    #[cfg(feature = "host")]
    #[allow(clippy::type_complexity)]
    unsafe fn borrow<A: CudaAlloc>(
        &self,
        alloc: A,
    ) -> CudaResult<(
        DeviceAccessible<Self::CudaRepresentation>,
        CombinedCudaAlloc<Self::CudaAllocation, A>,
    )> {
        let device_buffer = CudaDropWrapper::from(DeviceBuffer::from_slice(
            DeviceCopyWithPortableBitSemantics::from_slice(self),
        )?);

        Ok((
            DeviceAccessible::from(SliceRefCudaRepresentation {
                data: DeviceConstPointer(device_buffer.as_ptr().cast()),
                len: device_buffer.len(),
                _marker: PhantomData::<&'a [T]>,
            }),
            CombinedCudaAlloc::new(device_buffer, alloc),
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
    for SliceRefCudaRepresentation<'a, T>
{
    type RustRepresentation = &'a [T];

    #[cfg(feature = "device")]
    unsafe fn as_rust(this: &DeviceAccessible<Self>) -> Self::RustRepresentation {
        core::slice::from_raw_parts(this.data.0, this.len)
    }
}
