use core::ops::{Deref, DerefMut};

use rustacuda::{
    error::CudaResult,
    memory::{DeviceBuffer, LockedBuffer},
};

use rustacuda_core::DeviceCopy;

use crate::{
    common::{DeviceOwnedSlice, RustToCuda},
    host::{CombinedCudaAlloc, CudaAlloc, CudaDropWrapper, NullCudaAlloc},
};

use super::CudaExchangeBufferCudaRepresentation;

#[allow(clippy::module_name_repetitions)]
pub struct CudaExchangeBufferHost<T: Clone + DeviceCopy> {
    host_buffer: CudaDropWrapper<LockedBuffer<T>>,
    device_buffer: CudaDropWrapper<DeviceBuffer<T>>,
}

impl<T: Clone + DeviceCopy> CudaExchangeBufferHost<T> {
    /// # Errors
    /// Returns a `rustacuda::errors::CudaError` iff an error occurs inside CUDA
    pub fn new(elem: &T, capacity: usize) -> CudaResult<Self> {
        let host_buffer = CudaDropWrapper::from(LockedBuffer::new(elem, capacity)?);
        let device_buffer =
            CudaDropWrapper::from(DeviceBuffer::from_slice(host_buffer.as_slice())?);

        Ok(Self {
            host_buffer,
            device_buffer,
        })
    }
}

impl<T: Clone + DeviceCopy> Deref for CudaExchangeBufferHost<T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        self.host_buffer.as_slice()
    }
}

impl<T: Clone + DeviceCopy> DerefMut for CudaExchangeBufferHost<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.host_buffer.as_mut_slice()
    }
}

unsafe impl<T: Clone + DeviceCopy> RustToCuda for CudaExchangeBufferHost<T> {
    type CudaAllocation = NullCudaAlloc;
    type CudaRepresentation = CudaExchangeBufferCudaRepresentation<T>;

    #[allow(clippy::type_complexity)]
    unsafe fn borrow_mut<A: CudaAlloc>(
        &mut self,
        alloc: A,
    ) -> rustacuda::error::CudaResult<(
        Self::CudaRepresentation,
        CombinedCudaAlloc<Self::CudaAllocation, A>,
    )> {
        use rustacuda::memory::CopyDestination;

        self.device_buffer.copy_from(self.host_buffer.as_slice())?;

        Ok((
            CudaExchangeBufferCudaRepresentation(DeviceOwnedSlice::from(&mut self.device_buffer)),
            CombinedCudaAlloc::new(NullCudaAlloc, alloc),
        ))
    }

    #[allow(clippy::type_complexity)]
    unsafe fn un_borrow_mut<A: CudaAlloc>(
        &mut self,
        _cuda_repr: Self::CudaRepresentation,
        alloc: CombinedCudaAlloc<Self::CudaAllocation, A>,
    ) -> rustacuda::error::CudaResult<A> {
        use rustacuda::memory::CopyDestination;

        let (_alloc_front, alloc_tail) = alloc.split();

        self.device_buffer
            .copy_to(self.host_buffer.as_mut_slice())?;

        Ok(alloc_tail)
    }
}
