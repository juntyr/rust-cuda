use std::{
    cell::UnsafeCell,
    ops::{Deref, DerefMut},
};

use const_type_layout::TypeGraphLayout;
use cust::{
    error::CudaResult,
    memory::{DeviceBuffer, LockedBuffer},
};

use crate::{
    alloc::{CombinedCudaAlloc, CudaAlloc, NoCudaAlloc},
    host::CudaDropWrapper,
    safety::{PortableBitSemantics, StackOnly},
    utils::{
        adapter::DeviceCopyWithPortableBitSemantics,
        ffi::{DeviceAccessible, DeviceMutPointer},
        r#async::{Async, CompletionFnMut, NoCompletion},
    },
};

use super::{common::CudaExchangeBufferCudaRepresentation, CudaExchangeItem};

#[expect(clippy::module_name_repetitions)]
pub struct CudaExchangeBufferHost<
    T: StackOnly + PortableBitSemantics + TypeGraphLayout,
    const M2D: bool,
    const M2H: bool,
> {
    host_buffer: CudaDropWrapper<
        LockedBuffer<DeviceCopyWithPortableBitSemantics<CudaExchangeItem<T, M2D, M2H>>>,
    >,
    device_buffer: UnsafeCell<
        CudaDropWrapper<
            DeviceBuffer<DeviceCopyWithPortableBitSemantics<CudaExchangeItem<T, M2D, M2H>>>,
        >,
    >,
}

impl<
        T: Clone + StackOnly + PortableBitSemantics + TypeGraphLayout,
        const M2D: bool,
        const M2H: bool,
    > CudaExchangeBufferHost<T, M2D, M2H>
{
    /// # Errors
    /// Returns a [`cust::error::CudaError`] iff an error occurs inside
    /// CUDA
    pub fn new(elem: &T, capacity: usize) -> CudaResult<Self> {
        // Safety: CudaExchangeItem is a `repr(transparent)` wrapper around T
        let elem: &CudaExchangeItem<T, M2D, M2H> = unsafe { &*std::ptr::from_ref(elem).cast() };

        let host_buffer = CudaDropWrapper::from(LockedBuffer::new(
            DeviceCopyWithPortableBitSemantics::from_ref(elem),
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

impl<T: StackOnly + PortableBitSemantics + TypeGraphLayout, const M2D: bool, const M2H: bool>
    CudaExchangeBufferHost<T, M2D, M2H>
{
    /// # Errors
    /// Returns a [`cust::error::CudaError`] iff an error occurs inside
    /// CUDA
    pub fn from_vec(vec: Vec<T>) -> CudaResult<Self> {
        let host_buffer = unsafe {
            let mut uninit: CudaDropWrapper<LockedBuffer<DeviceCopyWithPortableBitSemantics<_>>> =
                CudaDropWrapper::from(LockedBuffer::uninitialized(vec.len())?);

            let uninit_ptr: *mut DeviceCopyWithPortableBitSemantics<CudaExchangeItem<T, M2D, M2H>> =
                uninit.as_mut_ptr();

            for (i, src) in vec.into_iter().enumerate() {
                uninit_ptr
                    .add(i)
                    .write(DeviceCopyWithPortableBitSemantics::from(CudaExchangeItem(
                        src,
                    )));
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

impl<T: StackOnly + PortableBitSemantics + TypeGraphLayout, const M2D: bool, const M2H: bool> Deref
    for CudaExchangeBufferHost<T, M2D, M2H>
{
    type Target = [CudaExchangeItem<T, M2D, M2H>];

    fn deref(&self) -> &Self::Target {
        DeviceCopyWithPortableBitSemantics::into_slice(self.host_buffer.as_slice())
    }
}

impl<T: StackOnly + PortableBitSemantics + TypeGraphLayout, const M2D: bool, const M2H: bool>
    DerefMut for CudaExchangeBufferHost<T, M2D, M2H>
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        DeviceCopyWithPortableBitSemantics::into_mut_slice(self.host_buffer.as_mut_slice())
    }
}

impl<T: StackOnly + PortableBitSemantics + TypeGraphLayout, const M2D: bool, const M2H: bool>
    CudaExchangeBufferHost<T, M2D, M2H>
{
    #[expect(clippy::type_complexity)]
    pub unsafe fn borrow<A: CudaAlloc>(
        &self,
        alloc: A,
    ) -> cust::error::CudaResult<(
        DeviceAccessible<CudaExchangeBufferCudaRepresentation<T, M2D, M2H>>,
        CombinedCudaAlloc<NoCudaAlloc, A>,
    )> {
        // Safety: device_buffer is inside an UnsafeCell
        //         borrow checks must be satisfied through LendToCuda
        let device_buffer = &mut *self.device_buffer.get();

        if M2D {
            // Only move the buffer contents to the device if needed

            cust::memory::CopyDestination::copy_from(
                &mut ***device_buffer,
                self.host_buffer.as_slice(),
            )?;
        }

        Ok((
            DeviceAccessible::from(CudaExchangeBufferCudaRepresentation(
                DeviceMutPointer(device_buffer.as_device_ptr().as_mut_ptr().cast()),
                device_buffer.len(),
            )),
            CombinedCudaAlloc::new(NoCudaAlloc, alloc),
        ))
    }

    pub unsafe fn restore<A: CudaAlloc>(
        &mut self,
        alloc: CombinedCudaAlloc<NoCudaAlloc, A>,
    ) -> cust::error::CudaResult<A> {
        let (_alloc_front, alloc_tail) = alloc.split();

        if M2H {
            // Only move the buffer contents back to the host if needed

            cust::memory::CopyDestination::copy_to(
                &***self.device_buffer.get_mut(),
                self.host_buffer.as_mut_slice(),
            )?;
        }

        Ok(alloc_tail)
    }
}

impl<T: StackOnly + PortableBitSemantics + TypeGraphLayout, const M2D: bool, const M2H: bool>
    CudaExchangeBufferHost<T, M2D, M2H>
{
    #[expect(clippy::type_complexity)]
    pub unsafe fn borrow_async<'stream, A: CudaAlloc>(
        &self,
        alloc: A,
        stream: crate::host::Stream<'stream>,
    ) -> cust::error::CudaResult<(
        Async<'_, 'stream, DeviceAccessible<CudaExchangeBufferCudaRepresentation<T, M2D, M2H>>>,
        CombinedCudaAlloc<NoCudaAlloc, A>,
    )> {
        // Safety: device_buffer is inside an UnsafeCell
        //         borrow checks must be satisfied through LendToCuda
        let device_buffer = &mut *self.device_buffer.get();

        if M2D {
            // Only move the buffer contents to the device if needed

            cust::memory::AsyncCopyDestination::async_copy_from(
                &mut ***device_buffer,
                self.host_buffer.as_slice(),
                &stream,
            )?;
        }

        let cuda_repr = DeviceAccessible::from(CudaExchangeBufferCudaRepresentation(
            DeviceMutPointer(device_buffer.as_device_ptr().as_mut_ptr().cast()),
            device_buffer.len(),
        ));

        let r#async = if M2D {
            Async::pending(cuda_repr, stream, NoCompletion)?
        } else {
            Async::ready(cuda_repr, stream)
        };

        Ok((r#async, CombinedCudaAlloc::new(NoCudaAlloc, alloc)))
    }

    #[expect(clippy::type_complexity)]
    pub unsafe fn restore_async<'a, 'stream, A: CudaAlloc, O>(
        mut this: owning_ref::BoxRefMut<'a, O, Self>,
        alloc: CombinedCudaAlloc<NoCudaAlloc, A>,
        stream: crate::host::Stream<'stream>,
    ) -> cust::error::CudaResult<(
        Async<'a, 'stream, owning_ref::BoxRefMut<'a, O, Self>, CompletionFnMut<'a, Self>>,
        A,
    )> {
        let (_alloc_front, alloc_tail) = alloc.split();

        if M2H {
            // Only move the buffer contents back to the host if needed

            let this: &mut Self = &mut this;

            cust::memory::AsyncCopyDestination::async_copy_to(
                &***this.device_buffer.get_mut(),
                this.host_buffer.as_mut_slice(),
                &stream,
            )?;
        }

        let r#async = if M2H {
            Async::<_, CompletionFnMut<'a, Self>>::pending(this, stream, Box::new(|_this| Ok(())))?
        } else {
            Async::ready(this, stream)
        };

        Ok((r#async, alloc_tail))
    }
}
