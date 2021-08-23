use alloc::vec::Vec;
use core::{
    cell::UnsafeCell,
    ops::{Deref, DerefMut},
};

use rustacuda::{
    error::CudaResult,
    memory::{DeviceBuffer, LockedBuffer},
};

use rustacuda_core::DeviceCopy;

use crate::{
    common::{r#impl::RustToCudaImpl, DeviceAccessible},
    host::{CombinedCudaAlloc, CudaAlloc, CudaDropWrapper, NullCudaAlloc},
};

use super::CudaExchangeBufferCudaRepresentation;

#[allow(clippy::module_name_repetitions)]
pub struct CudaExchangeBufferHost<T: DeviceCopy> {
    host_buffer: CudaDropWrapper<LockedBuffer<T>>,
    device_buffer: UnsafeCell<CudaDropWrapper<DeviceBuffer<T>>>,
}

impl<T: Clone + DeviceCopy> CudaExchangeBufferHost<T> {
    /// # Errors
    /// Returns a `rustacuda::errors::CudaError` iff an error occurs inside CUDA
    pub fn new(elem: &T, capacity: usize) -> CudaResult<Self> {
        let host_buffer = CudaDropWrapper::from(LockedBuffer::new(elem, capacity)?);
        let device_buffer = UnsafeCell::new(CudaDropWrapper::from(DeviceBuffer::from_slice(
            host_buffer.as_slice(),
        )?));

        Ok(Self {
            host_buffer,
            device_buffer,
        })
    }
}

impl<T: DeviceCopy> CudaExchangeBufferHost<T> {
    /// # Errors
    /// Returns a `rustacuda::errors::CudaError` iff an error occurs inside CUDA
    pub fn from_vec(vec: Vec<T>) -> CudaResult<Self> {
        let mut host_buffer_uninit =
            CudaDropWrapper::from(unsafe { LockedBuffer::uninitialized(vec.len())? });

        for (src, dst) in vec.into_iter().zip(host_buffer_uninit.iter_mut()) {
            *dst = src;
        }

        let host_buffer = host_buffer_uninit;

        let device_buffer = UnsafeCell::new(CudaDropWrapper::from(DeviceBuffer::from_slice(
            host_buffer.as_slice(),
        )?));

        Ok(Self {
            host_buffer,
            device_buffer,
        })
    }
}

impl<T: DeviceCopy> Deref for CudaExchangeBufferHost<T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        self.host_buffer.as_slice()
    }
}

impl<T: DeviceCopy> DerefMut for CudaExchangeBufferHost<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.host_buffer.as_mut_slice()
    }
}

unsafe impl<T: DeviceCopy> RustToCudaImpl for CudaExchangeBufferHost<T> {
    type CudaAllocationImpl = NullCudaAlloc;
    type CudaRepresentationImpl = CudaExchangeBufferCudaRepresentation<T>;

    #[allow(clippy::type_complexity)]
    unsafe fn borrow_impl<A: CudaAlloc>(
        &self,
        alloc: A,
    ) -> rustacuda::error::CudaResult<(
        DeviceAccessible<Self::CudaRepresentationImpl>,
        CombinedCudaAlloc<Self::CudaAllocationImpl, A>,
    )> {
        use rustacuda::memory::CopyDestination;

        // Safety: device_buffer is inside an UnsafeCell
        //         borrow checks must be satisfied through LendToCuda
        let device_buffer = &mut *self.device_buffer.get();

        device_buffer.copy_from(self.host_buffer.as_slice())?;

        Ok((
            DeviceAccessible::from(CudaExchangeBufferCudaRepresentation(
                device_buffer.as_mut_ptr(),
                device_buffer.len(),
            )),
            CombinedCudaAlloc::new(NullCudaAlloc, alloc),
        ))
    }

    #[allow(clippy::type_complexity)]
    unsafe fn restore_impl<A: CudaAlloc>(
        &mut self,
        alloc: CombinedCudaAlloc<Self::CudaAllocationImpl, A>,
    ) -> rustacuda::error::CudaResult<A> {
        use rustacuda::memory::CopyDestination;

        let (_alloc_front, alloc_tail) = alloc.split();

        self.device_buffer
            .get_mut()
            .copy_to(self.host_buffer.as_mut_slice())?;

        Ok(alloc_tail)
    }
}
