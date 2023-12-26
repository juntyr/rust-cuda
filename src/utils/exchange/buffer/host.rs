use std::{
    cell::UnsafeCell,
    ops::{Deref, DerefMut},
};

use const_type_layout::TypeGraphLayout;
use rustacuda::{
    error::CudaResult,
    memory::{DeviceBuffer, LockedBuffer},
};

use crate::{
    alloc::{CombinedCudaAlloc, CudaAlloc, NoCudaAlloc},
    host::CudaDropWrapper,
    safety::PortableBitSemantics,
    utils::{
        device_copy::SafeDeviceCopyWrapper,
        ffi::{DeviceAccessible, DeviceMutPointer},
    },
};

use super::{common::CudaExchangeBufferCudaRepresentation, CudaExchangeItem};

#[allow(clippy::module_name_repetitions)]
pub struct CudaExchangeBufferHost<
    T: PortableBitSemantics + TypeGraphLayout,
    const M2D: bool,
    const M2H: bool,
> {
    host_buffer:
        CudaDropWrapper<LockedBuffer<SafeDeviceCopyWrapper<CudaExchangeItem<T, M2D, M2H>>>>,
    device_buffer: UnsafeCell<
        CudaDropWrapper<DeviceBuffer<SafeDeviceCopyWrapper<CudaExchangeItem<T, M2D, M2H>>>>,
    >,
}

impl<T: Clone + PortableBitSemantics + TypeGraphLayout, const M2D: bool, const M2H: bool>
    CudaExchangeBufferHost<T, M2D, M2H>
{
    /// # Errors
    /// Returns a [`rustacuda::error::CudaError`] iff an error occurs inside
    /// CUDA
    pub fn new(elem: &T, capacity: usize) -> CudaResult<Self> {
        // Safety: CudaExchangeItem is a `repr(transparent)` wrapper around T
        let elem: &CudaExchangeItem<T, M2D, M2H> = unsafe { &*(elem as *const T).cast() };

        let host_buffer = CudaDropWrapper::from(LockedBuffer::new(
            SafeDeviceCopyWrapper::from_ref(elem),
            capacity,
        )?);
        let device_buffer = UnsafeCell::new(CudaDropWrapper::from(DeviceBuffer::from_slice(
            host_buffer.as_slice(),
        )?));

        Ok(Self {
            host_buffer,
            device_buffer,
        })
    }
}

impl<T: PortableBitSemantics + TypeGraphLayout, const M2D: bool, const M2H: bool>
    CudaExchangeBufferHost<T, M2D, M2H>
{
    /// # Errors
    /// Returns a [`rustacuda::error::CudaError`] iff an error occurs inside
    /// CUDA
    pub fn from_vec(vec: Vec<T>) -> CudaResult<Self> {
        let host_buffer = unsafe {
            let mut uninit: CudaDropWrapper<LockedBuffer<SafeDeviceCopyWrapper<_>>> =
                CudaDropWrapper::from(LockedBuffer::uninitialized(vec.len())?);

            for (i, src) in vec.into_iter().enumerate() {
                uninit
                    .as_mut_ptr()
                    .add(i)
                    .write(SafeDeviceCopyWrapper::from(CudaExchangeItem(src)));
            }

            uninit
        };

        let device_buffer = UnsafeCell::new(CudaDropWrapper::from(DeviceBuffer::from_slice(
            host_buffer.as_slice(),
        )?));

        Ok(Self {
            host_buffer,
            device_buffer,
        })
    }
}

impl<T: PortableBitSemantics + TypeGraphLayout, const M2D: bool, const M2H: bool> Deref
    for CudaExchangeBufferHost<T, M2D, M2H>
{
    type Target = [CudaExchangeItem<T, M2D, M2H>];

    fn deref(&self) -> &Self::Target {
        SafeDeviceCopyWrapper::into_slice(self.host_buffer.as_slice())
    }
}

impl<T: PortableBitSemantics + TypeGraphLayout, const M2D: bool, const M2H: bool> DerefMut
    for CudaExchangeBufferHost<T, M2D, M2H>
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        SafeDeviceCopyWrapper::into_mut_slice(self.host_buffer.as_mut_slice())
    }
}

impl<T: PortableBitSemantics + TypeGraphLayout, const M2D: bool, const M2H: bool>
    CudaExchangeBufferHost<T, M2D, M2H>
{
    #[allow(clippy::type_complexity)]
    pub unsafe fn borrow<A: CudaAlloc>(
        &self,
        alloc: A,
    ) -> rustacuda::error::CudaResult<(
        DeviceAccessible<CudaExchangeBufferCudaRepresentation<T, M2D, M2H>>,
        CombinedCudaAlloc<NoCudaAlloc, A>,
    )> {
        // Safety: device_buffer is inside an UnsafeCell
        //         borrow checks must be satisfied through LendToCuda
        let device_buffer = &mut *self.device_buffer.get();

        if M2D {
            // Only move the buffer contents to the device if needed

            rustacuda::memory::CopyDestination::copy_from(
                &mut ***device_buffer,
                self.host_buffer.as_slice(),
            )?;
        }

        Ok((
            DeviceAccessible::from(CudaExchangeBufferCudaRepresentation(
                DeviceMutPointer(device_buffer.as_mut_ptr().cast()),
                device_buffer.len(),
            )),
            CombinedCudaAlloc::new(NoCudaAlloc, alloc),
        ))
    }

    #[allow(clippy::type_complexity)]
    pub unsafe fn restore<A: CudaAlloc>(
        &mut self,
        alloc: CombinedCudaAlloc<NoCudaAlloc, A>,
    ) -> rustacuda::error::CudaResult<A> {
        let (_alloc_front, alloc_tail) = alloc.split();

        if M2H {
            // Only move the buffer contents back to the host if needed

            rustacuda::memory::CopyDestination::copy_to(
                &***self.device_buffer.get_mut(),
                self.host_buffer.as_mut_slice(),
            )?;
        }

        Ok(alloc_tail)
    }
}

impl<T: PortableBitSemantics + TypeGraphLayout, const M2D: bool, const M2H: bool>
    CudaExchangeBufferHost<T, M2D, M2H>
{
    #[allow(clippy::type_complexity)]
    pub unsafe fn borrow_async<A: CudaAlloc>(
        &self,
        alloc: A,
        stream: &rustacuda::stream::Stream,
    ) -> rustacuda::error::CudaResult<(
        DeviceAccessible<CudaExchangeBufferCudaRepresentation<T, M2D, M2H>>,
        CombinedCudaAlloc<NoCudaAlloc, A>,
    )> {
        // Safety: device_buffer is inside an UnsafeCell
        //         borrow checks must be satisfied through LendToCuda
        let device_buffer = &mut *self.device_buffer.get();

        if M2D {
            // Only move the buffer contents to the device if needed

            rustacuda::memory::AsyncCopyDestination::async_copy_from(
                &mut ***device_buffer,
                self.host_buffer.as_slice(),
                stream,
            )?;
        }

        Ok((
            DeviceAccessible::from(CudaExchangeBufferCudaRepresentation(
                DeviceMutPointer(device_buffer.as_mut_ptr().cast()),
                device_buffer.len(),
            )),
            CombinedCudaAlloc::new(NoCudaAlloc, alloc),
        ))
    }

    #[allow(clippy::type_complexity)]
    pub unsafe fn restore_async<A: CudaAlloc>(
        &mut self,
        alloc: CombinedCudaAlloc<NoCudaAlloc, A>,
        stream: &rustacuda::stream::Stream,
    ) -> rustacuda::error::CudaResult<A> {
        let (_alloc_front, alloc_tail) = alloc.split();

        if M2H {
            // Only move the buffer contents back to the host if needed

            rustacuda::memory::AsyncCopyDestination::async_copy_to(
                &***self.device_buffer.get_mut(),
                self.host_buffer.as_mut_slice(),
                stream,
            )?;
        }

        Ok(alloc_tail)
    }
}
