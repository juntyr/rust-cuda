use alloc::vec::Vec;
use core::{
    cell::UnsafeCell,
    ops::{Deref, DerefMut},
};

use rustacuda::{
    error::CudaResult,
    memory::{DeviceBuffer, LockedBuffer},
};

use crate::{
    common::{DeviceAccessible, RustToCuda},
    host::{CombinedCudaAlloc, CudaAlloc, CudaDropWrapper, NullCudaAlloc},
    memory::SafeDeviceCopy,
};

use super::{common::CudaExchangeBufferCudaRepresentation, CudaExchangeItem};

#[allow(clippy::module_name_repetitions)]
#[doc(cfg(feature = "host"))]
/// When the `host` feature is **not** set,
/// [`CudaExchangeBuffer`](super::CudaExchangeBuffer)
/// refers to
/// [`CudaExchangeBufferDevice`](super::CudaExchangeBufferDevice)
/// instead.
/// [`CudaExchangeBufferHost`](Self) is never exposed directly.
pub struct CudaExchangeBufferHost<T: SafeDeviceCopy, const M2D: bool, const M2H: bool> {
    host_buffer: CudaDropWrapper<LockedBuffer<CudaExchangeItem<T, M2D, M2H>>>,
    device_buffer: UnsafeCell<CudaDropWrapper<DeviceBuffer<CudaExchangeItem<T, M2D, M2H>>>>,
}

impl<T: Clone + SafeDeviceCopy, const M2D: bool, const M2H: bool>
    CudaExchangeBufferHost<T, M2D, M2H>
{
    /// # Errors
    /// Returns a `rustacuda::errors::CudaError` iff an error occurs inside CUDA
    pub fn new(elem: &T, capacity: usize) -> CudaResult<Self> {
        // Safety: CudaExchangeItem is a `repr(transparent)` wrapper around T
        let elem: &CudaExchangeItem<T, M2D, M2H> = unsafe { &*(elem as *const T).cast() };

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

impl<T: SafeDeviceCopy, const M2D: bool, const M2H: bool> CudaExchangeBufferHost<T, M2D, M2H> {
    /// # Errors
    /// Returns a `rustacuda::errors::CudaError` iff an error occurs inside CUDA
    pub fn from_vec(vec: Vec<T>) -> CudaResult<Self> {
        let mut host_buffer_uninit =
            CudaDropWrapper::from(unsafe { LockedBuffer::uninitialized(vec.len())? });

        for (src, dst) in vec.into_iter().zip(host_buffer_uninit.iter_mut()) {
            *dst = CudaExchangeItem(src);
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

impl<T: SafeDeviceCopy, const M2D: bool, const M2H: bool> Deref
    for CudaExchangeBufferHost<T, M2D, M2H>
{
    type Target = [CudaExchangeItem<T, M2D, M2H>];

    fn deref(&self) -> &Self::Target {
        self.host_buffer.as_slice()
    }
}

impl<T: SafeDeviceCopy, const M2D: bool, const M2H: bool> DerefMut
    for CudaExchangeBufferHost<T, M2D, M2H>
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.host_buffer.as_mut_slice()
    }
}

unsafe impl<T: SafeDeviceCopy, const M2D: bool, const M2H: bool> RustToCuda
    for CudaExchangeBufferHost<T, M2D, M2H>
{
    type CudaAllocation = NullCudaAlloc;
    type CudaRepresentation = CudaExchangeBufferCudaRepresentation<T, M2D, M2H>;

    #[allow(clippy::type_complexity)]
    unsafe fn borrow<A: CudaAlloc>(
        &self,
        alloc: A,
    ) -> rustacuda::error::CudaResult<(
        DeviceAccessible<Self::CudaRepresentation>,
        CombinedCudaAlloc<Self::CudaAllocation, A>,
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
                device_buffer.as_mut_ptr(),
                device_buffer.len(),
            )),
            CombinedCudaAlloc::new(NullCudaAlloc, alloc),
        ))
    }

    #[allow(clippy::type_complexity)]
    unsafe fn restore<A: CudaAlloc>(
        &mut self,
        alloc: CombinedCudaAlloc<Self::CudaAllocation, A>,
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
