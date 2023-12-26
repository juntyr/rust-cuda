use core::marker::PhantomData;

use const_type_layout::{TypeGraphLayout, TypeLayout};

#[cfg(feature = "host")]
use rustacuda::{error::CudaResult, memory::DeviceBox};

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
    utils::device_copy::SafeDeviceCopyWrapper,
};

#[doc(hidden)]
#[repr(transparent)]
#[derive(TypeLayout)]
#[allow(clippy::module_name_repetitions)]
pub struct RefMutCudaRepresentation<'a, T: 'a + PortableBitSemantics + TypeGraphLayout> {
    data: DeviceMutPointer<T>,
    _marker: PhantomData<&'a mut T>,
}

unsafe impl<'a, T: PortableBitSemantics + TypeGraphLayout> RustToCuda for &'a mut T {
    #[cfg(all(feature = "host", not(doc)))]
    type CudaAllocation = CudaDropWrapper<DeviceBox<SafeDeviceCopyWrapper<T>>>;
    #[cfg(any(not(feature = "host"), doc))]
    type CudaAllocation = crate::alloc::SomeCudaAlloc;
    type CudaRepresentation = RefMutCudaRepresentation<'a, T>;

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
            DeviceAccessible::from(RefMutCudaRepresentation {
                data: DeviceMutPointer(device_box.as_device_ptr().as_raw_mut().cast()),
                _marker: PhantomData::<&'a mut T>,
            }),
            CombinedCudaAlloc::new(device_box, alloc),
        ))
    }

    #[cfg(feature = "host")]
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

unsafe impl<'a, T: PortableBitSemantics + TypeGraphLayout> CudaAsRust
    for RefMutCudaRepresentation<'a, T>
{
    type RustRepresentation = &'a mut T;

    #[cfg(feature = "device")]
    unsafe fn as_rust(this: &DeviceAccessible<Self>) -> Self::RustRepresentation {
        let data: *mut T = this.data.0;
        &mut *data
    }
}
