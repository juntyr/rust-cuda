use core::marker::PhantomData;

use const_type_layout::{TypeGraphLayout, TypeLayout};

#[cfg(feature = "host")]
use rustacuda::{error::CudaResult, memory::DeviceBuffer};

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
#[allow(clippy::module_name_repetitions)]
#[derive(Debug, TypeLayout)]
#[repr(C)]
pub struct SliceRefMutCudaRepresentation<'a, T: 'a + SafeDeviceCopy + TypeGraphLayout> {
    data: *mut T,
    len: usize,
    _marker: PhantomData<&'a mut [T]>,
}

// Safety: This repr(C) struct only contains a device-owned pointer and a usize
unsafe impl<'a, T: SafeDeviceCopy + TypeGraphLayout> rustacuda_core::DeviceCopy
    for SliceRefMutCudaRepresentation<'a, T>
{
}

unsafe impl<'a, T: SafeDeviceCopy + TypeGraphLayout> RustToCuda for &'a mut [T] {
    #[cfg(all(feature = "host", not(doc)))]
    type CudaAllocation = crate::host::CudaDropWrapper<DeviceBuffer<SafeDeviceCopyWrapper<T>>>;
    #[cfg(any(not(feature = "host"), doc))]
    type CudaAllocation = crate::common::SomeCudaAlloc;
    type CudaRepresentation = SliceRefMutCudaRepresentation<'a, T>;

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
            DeviceAccessible::from(SliceRefMutCudaRepresentation {
                data: device_buffer.as_mut_ptr().cast(),
                len: device_buffer.len(),
                _marker: PhantomData::<&'a mut [T]>,
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

unsafe impl<'a, T: SafeDeviceCopy + TypeGraphLayout> CudaAsRust
    for SliceRefMutCudaRepresentation<'a, T>
{
    type RustRepresentation = &'a mut [T];

    #[cfg(feature = "device")]
    unsafe fn as_rust(this: &DeviceAccessible<Self>) -> Self::RustRepresentation {
        core::slice::from_raw_parts_mut(this.data, this.len)
    }
}
