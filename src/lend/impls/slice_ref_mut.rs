use core::marker::PhantomData;

use const_type_layout::{TypeGraphLayout, TypeLayout};

#[cfg(feature = "host")]
use rustacuda::{error::CudaResult, memory::DeviceBuffer};

use crate::{
    lend::{CudaAsRust, RustToCuda},
    safety::PortableBitSemantics,
    utils::ffi::DeviceMutPointer,
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
#[expect(clippy::module_name_repetitions)]
#[derive(TypeLayout)]
#[repr(C)]
pub struct SliceRefMutCudaRepresentation<'a, T: 'a + PortableBitSemantics + TypeGraphLayout> {
    data: DeviceMutPointer<T>,
    len: usize,
    _marker: PhantomData<&'a mut [T]>,
}

unsafe impl<'a, T: PortableBitSemantics + TypeGraphLayout> RustToCuda for &'a mut [T] {
    #[cfg(all(feature = "host", not(doc)))]
    type CudaAllocation =
        crate::host::CudaDropWrapper<DeviceBuffer<DeviceCopyWithPortableBitSemantics<T>>>;
    #[cfg(any(not(feature = "host"), doc))]
    type CudaAllocation = crate::alloc::SomeCudaAlloc;
    type CudaRepresentation = SliceRefMutCudaRepresentation<'a, T>;

    #[cfg(feature = "host")]
    unsafe fn borrow<A: CudaAlloc>(
        &self,
        alloc: A,
    ) -> CudaResult<(
        DeviceAccessible<Self::CudaRepresentation>,
        CombinedCudaAlloc<Self::CudaAllocation, A>,
    )> {
        let mut device_buffer = CudaDropWrapper::from(DeviceBuffer::from_slice(
            DeviceCopyWithPortableBitSemantics::from_slice(self),
        )?);

        Ok((
            DeviceAccessible::from(SliceRefMutCudaRepresentation {
                data: DeviceMutPointer(device_buffer.as_mut_ptr().cast()),
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

        alloc_front.copy_to(DeviceCopyWithPortableBitSemantics::from_mut_slice(self))?;

        core::mem::drop(alloc_front);

        Ok(alloc_tail)
    }
}

// &mut [T] cannot implement RustToCudaAsync since the slice, potentially with
//  garbage data, would remain accessible after failing a mutable restore

unsafe impl<'a, T: PortableBitSemantics + TypeGraphLayout> CudaAsRust
    for SliceRefMutCudaRepresentation<'a, T>
{
    type RustRepresentation = &'a mut [T];

    #[cfg(feature = "device")]
    unsafe fn as_rust(this: &DeviceAccessible<Self>) -> Self::RustRepresentation {
        core::slice::from_raw_parts_mut(this.data.0, this.len)
    }
}
