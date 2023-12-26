use std::{
    future::{Future, IntoFuture},
    marker::PhantomData,
    ops::{Deref, DerefMut},
    sync::{Arc, Mutex},
    task::{Poll, Waker},
};

use rustacuda::{
    error::{CudaError, CudaResult},
    event::{Event, EventFlags, EventStatus},
    memory::{AsyncCopyDestination, CopyDestination, DeviceBox, LockedBox},
    stream::{Stream, StreamWaitEventFlags},
};

use crate::{
    alloc::{CombinedCudaAlloc, EmptyCudaAlloc, NoCudaAlloc},
    host::{
        CudaDropWrapper, HostAndDeviceConstRef, HostAndDeviceConstRefAsync, HostAndDeviceMutRef,
        HostAndDeviceMutRefAsync,
    },
    lend::{RustToCuda, RustToCudaAsync},
    utils::{device_copy::SafeDeviceCopyWrapper, ffi::DeviceAccessible},
};

#[allow(clippy::module_name_repetitions)]
pub struct ExchangeWrapperOnHost<T: RustToCuda<CudaAllocation: EmptyCudaAlloc>> {
    value: T,
    device_box: CudaDropWrapper<
        DeviceBox<SafeDeviceCopyWrapper<DeviceAccessible<<T as RustToCuda>::CudaRepresentation>>>,
    >,
    locked_cuda_repr: CudaDropWrapper<
        LockedBox<SafeDeviceCopyWrapper<DeviceAccessible<<T as RustToCuda>::CudaRepresentation>>>,
    >,
    move_event: CudaDropWrapper<Event>,
}

#[allow(clippy::module_name_repetitions)]
pub struct ExchangeWrapperOnHostAsync<'stream, T: RustToCuda<CudaAllocation: EmptyCudaAlloc>> {
    value: T,
    device_box: CudaDropWrapper<
        DeviceBox<SafeDeviceCopyWrapper<DeviceAccessible<<T as RustToCuda>::CudaRepresentation>>>,
    >,
    locked_cuda_repr: CudaDropWrapper<
        LockedBox<SafeDeviceCopyWrapper<DeviceAccessible<<T as RustToCuda>::CudaRepresentation>>>,
    >,
    move_event: CudaDropWrapper<Event>,
    stream: PhantomData<&'stream Stream>,
    waker: Arc<Mutex<Option<Waker>>>,
}

#[allow(clippy::module_name_repetitions)]
pub struct ExchangeWrapperOnDevice<T: RustToCuda<CudaAllocation: EmptyCudaAlloc>> {
    value: T,
    device_box: CudaDropWrapper<
        DeviceBox<SafeDeviceCopyWrapper<DeviceAccessible<<T as RustToCuda>::CudaRepresentation>>>,
    >,
    locked_cuda_repr: CudaDropWrapper<
        LockedBox<SafeDeviceCopyWrapper<DeviceAccessible<<T as RustToCuda>::CudaRepresentation>>>,
    >,
    null_alloc: CombinedCudaAlloc<<T as RustToCuda>::CudaAllocation, NoCudaAlloc>,
    move_event: CudaDropWrapper<Event>,
}

#[allow(clippy::module_name_repetitions)]
pub struct ExchangeWrapperOnDeviceAsync<'stream, T: RustToCuda<CudaAllocation: EmptyCudaAlloc>> {
    value: T,
    device_box: CudaDropWrapper<
        DeviceBox<SafeDeviceCopyWrapper<DeviceAccessible<<T as RustToCuda>::CudaRepresentation>>>,
    >,
    locked_cuda_repr: CudaDropWrapper<
        LockedBox<SafeDeviceCopyWrapper<DeviceAccessible<<T as RustToCuda>::CudaRepresentation>>>,
    >,
    null_alloc: CombinedCudaAlloc<<T as RustToCuda>::CudaAllocation, NoCudaAlloc>,
    move_event: CudaDropWrapper<Event>,
    stream: &'stream Stream,
    waker: Arc<Mutex<Option<Waker>>>,
}

impl<T: RustToCuda<CudaAllocation: EmptyCudaAlloc>> ExchangeWrapperOnHost<T> {
    /// # Errors
    /// Returns a [`rustacuda::error::CudaError`] iff an error occurs inside
    /// CUDA
    pub fn new(value: T) -> CudaResult<Self> {
        // Safety: The uninitialised memory is never exposed
        //         To access the device memory, [`Self::move_to_device`] has to
        //          be called first, which initialised the memory.
        let device_box = CudaDropWrapper::from(unsafe { DeviceBox::uninitialized() }?);

        let (cuda_repr, _null_alloc) = unsafe { value.borrow(NoCudaAlloc) }?;
        let locked_cuda_repr = unsafe {
            let mut uninit = CudaDropWrapper::from(LockedBox::<
                SafeDeviceCopyWrapper<DeviceAccessible<<T as RustToCuda>::CudaRepresentation>>,
            >::uninitialized()?);
            uninit
                .as_mut_ptr()
                .write(SafeDeviceCopyWrapper::from(cuda_repr));
            uninit
        };

        let move_event = Event::new(EventFlags::DISABLE_TIMING)?.into();

        Ok(Self {
            value,
            device_box,
            locked_cuda_repr,
            move_event,
        })
    }

    /// Moves the data synchronously to the CUDA device, where it can then be
    /// lent out immutably via [`ExchangeWrapperOnDevice::as_ref`], or mutably
    /// via [`ExchangeWrapperOnDevice::as_mut`].
    ///
    /// To avoid aliasing, each CUDA thread will get access to its own shallow
    /// copy of the data. Hence,
    /// - any shallow changes to the data will NOT be reflected back to the CPU
    /// - any deep changes to the data WILL be reflected back to the CPU
    ///
    /// # Errors
    /// Returns a [`rustacuda::error::CudaError`] iff an error occurs inside
    /// CUDA
    pub fn move_to_device(mut self) -> CudaResult<ExchangeWrapperOnDevice<T>> {
        let (cuda_repr, null_alloc) = unsafe { self.value.borrow(NoCudaAlloc) }?;
        **self.locked_cuda_repr = SafeDeviceCopyWrapper::from(cuda_repr);

        self.device_box.copy_from(&**self.locked_cuda_repr)?;

        Ok(ExchangeWrapperOnDevice {
            value: self.value,
            device_box: self.device_box,
            locked_cuda_repr: self.locked_cuda_repr,
            null_alloc,
            move_event: self.move_event,
        })
    }
}

impl<T: RustToCudaAsync<CudaAllocation: EmptyCudaAlloc>> ExchangeWrapperOnHost<T> {
    /// Moves the data asynchronously to the CUDA device.
    ///
    /// To avoid aliasing, each CUDA thread will get access to its own shallow
    /// copy of the data. Hence,
    /// - any shallow changes to the data will NOT be reflected back to the CPU
    /// - any deep changes to the data WILL be reflected back to the CPU
    ///
    /// # Errors
    /// Returns a [`rustacuda::error::CudaError`] iff an error occurs inside
    /// CUDA
    pub fn move_to_device_async(
        mut self,
        stream: &Stream,
    ) -> CudaResult<ExchangeWrapperOnDeviceAsync<'_, T>> {
        let (cuda_repr, null_alloc) = unsafe { self.value.borrow_async(NoCudaAlloc, stream) }?;
        **self.locked_cuda_repr = SafeDeviceCopyWrapper::from(cuda_repr);

        // Safety: The device value is not safely exposed until either
        // - the passed-in [`Stream`] is synchronised
        // - the kernel is launched on the passed-in [`Stream`]
        unsafe {
            self.device_box
                .async_copy_from(&*self.locked_cuda_repr, stream)
        }?;
        self.move_event.record(stream)?;

        let waker: Arc<Mutex<Option<Waker>>> = Arc::new(Mutex::new(None));

        let waker_callback = waker.clone();
        stream.add_callback(Box::new(move |_| {
            if let Ok(mut w) = waker_callback.lock() {
                if let Some(w) = w.take() {
                    w.wake();
                }
            }
        }))?;

        Ok(ExchangeWrapperOnDeviceAsync {
            value: self.value,
            device_box: self.device_box,
            locked_cuda_repr: self.locked_cuda_repr,
            null_alloc,
            move_event: self.move_event,
            stream,
            waker,
        })
    }
}

impl<'stream, T: RustToCuda<CudaAllocation: EmptyCudaAlloc>>
    ExchangeWrapperOnHostAsync<'stream, T>
{
    /// Synchronises the host CPU thread until the data has moved to the CPU.
    ///
    /// # Errors
    /// Returns a [`rustacuda::error::CudaError`] iff an error occurs inside
    /// CUDA
    pub fn sync_to_host(self) -> CudaResult<ExchangeWrapperOnHost<T>> {
        self.move_event.synchronize()?;

        Ok(ExchangeWrapperOnHost {
            value: self.value,
            device_box: self.device_box,
            locked_cuda_repr: self.locked_cuda_repr,
            move_event: self.move_event,
        })
    }

    /// Moves the asynchronous data move to a different [`Stream`].
    ///
    /// # Errors
    /// Returns a [`rustacuda::error::CudaError`] iff an error occurs inside
    /// CUDA
    pub fn move_to_stream(self, stream: &Stream) -> CudaResult<ExchangeWrapperOnHostAsync<'_, T>> {
        stream.wait_event(&self.move_event, StreamWaitEventFlags::DEFAULT)?;
        self.move_event.record(stream)?;

        let waker_callback = self.waker.clone();
        stream.add_callback(Box::new(move |_| {
            if let Ok(mut w) = waker_callback.lock() {
                if let Some(w) = w.take() {
                    w.wake();
                }
            }
        }))?;

        Ok(ExchangeWrapperOnHostAsync {
            value: self.value,
            device_box: self.device_box,
            locked_cuda_repr: self.locked_cuda_repr,
            move_event: self.move_event,
            stream: PhantomData::<&Stream>,
            waker: self.waker,
        })
    }
}

impl<'stream, T: RustToCuda<CudaAllocation: EmptyCudaAlloc>> IntoFuture
    for ExchangeWrapperOnHostAsync<'stream, T>
{
    type Output = CudaResult<ExchangeWrapperOnHost<T>>;

    type IntoFuture = impl Future<Output = Self::Output>;

    fn into_future(self) -> Self::IntoFuture {
        let mut wrapper = Some(self);

        core::future::poll_fn(move |cx| match &wrapper {
            Some(inner) => match inner.move_event.query() {
                Ok(EventStatus::NotReady) => inner.waker.lock().map_or_else(
                    |_| Poll::Ready(Err(CudaError::OperatingSystemError)),
                    |mut w| {
                        *w = Some(cx.waker().clone());
                        Poll::Pending
                    },
                ),
                Ok(EventStatus::Ready) => match wrapper.take() {
                    Some(inner) => Poll::Ready(Ok(ExchangeWrapperOnHost {
                        value: inner.value,
                        device_box: inner.device_box,
                        locked_cuda_repr: inner.locked_cuda_repr,
                        move_event: inner.move_event,
                    })),
                    None => Poll::Ready(Err(CudaError::AlreadyAcquired)),
                },
                Err(err) => Poll::Ready(Err(err)),
            },
            None => Poll::Ready(Err(CudaError::AlreadyAcquired)),
        })
    }
}

impl<T: RustToCuda<CudaAllocation: EmptyCudaAlloc>> Deref for ExchangeWrapperOnHost<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.value
    }
}

impl<T: RustToCuda<CudaAllocation: EmptyCudaAlloc>> DerefMut for ExchangeWrapperOnHost<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.value
    }
}

impl<'stream, T: RustToCuda<CudaAllocation: EmptyCudaAlloc>>
    ExchangeWrapperOnDeviceAsync<'stream, T>
{
    /// Synchronises the host CPU thread until the data has moved to the GPU.
    ///
    /// # Errors
    /// Returns a [`rustacuda::error::CudaError`] iff an error occurs inside
    /// CUDA
    pub fn sync_to_device(self) -> CudaResult<ExchangeWrapperOnDevice<T>> {
        self.move_event.synchronize()?;

        Ok(ExchangeWrapperOnDevice {
            value: self.value,
            device_box: self.device_box,
            locked_cuda_repr: self.locked_cuda_repr,
            null_alloc: self.null_alloc,
            move_event: self.move_event,
        })
    }

    /// Moves the asynchronous data move to a different [`Stream`].
    ///
    /// # Errors
    /// Returns a [`rustacuda::error::CudaError`] iff an error occurs inside
    /// CUDA
    pub fn move_to_stream(
        self,
        stream: &Stream,
    ) -> CudaResult<ExchangeWrapperOnDeviceAsync<'_, T>> {
        stream.wait_event(&self.move_event, StreamWaitEventFlags::DEFAULT)?;
        self.move_event.record(stream)?;

        let waker_callback = self.waker.clone();
        stream.add_callback(Box::new(move |_| {
            if let Ok(mut w) = waker_callback.lock() {
                if let Some(w) = w.take() {
                    w.wake();
                }
            }
        }))?;

        Ok(ExchangeWrapperOnDeviceAsync {
            value: self.value,
            device_box: self.device_box,
            locked_cuda_repr: self.locked_cuda_repr,
            null_alloc: self.null_alloc,
            move_event: self.move_event,
            stream,
            waker: self.waker,
        })
    }

    pub fn as_ref_async(
        &self,
    ) -> HostAndDeviceConstRefAsync<DeviceAccessible<<T as RustToCuda>::CudaRepresentation>> {
        // Safety: `device_box` contains exactly the device copy of `locked_cuda_repr`
        unsafe {
            HostAndDeviceConstRefAsync::new(
                &*self.device_box,
                (**self.locked_cuda_repr).into_ref(),
                self.stream,
            )
        }
    }

    pub fn as_mut_async(
        &mut self,
    ) -> HostAndDeviceMutRefAsync<DeviceAccessible<<T as RustToCuda>::CudaRepresentation>> {
        // Safety: `device_box` contains exactly the device copy of `locked_cuda_repr`
        unsafe {
            HostAndDeviceMutRefAsync::new(
                &mut self.device_box,
                (**self.locked_cuda_repr).into_mut(),
                self.stream,
            )
        }
    }

    /// Moves the data synchronously back to the host CPU device.
    ///
    /// To avoid aliasing, each CUDA thread only got access to its own shallow
    /// copy of the data. Hence,
    /// - any shallow changes to the data will NOT be reflected back to the CPU
    /// - any deep changes to the data WILL be reflected back to the CPU
    ///
    /// # Errors
    /// Returns a [`rustacuda::error::CudaError`] iff an error occurs inside
    /// CUDA
    pub fn move_to_host(mut self) -> CudaResult<ExchangeWrapperOnHost<T>> {
        // Reflect deep changes back to the CPU
        let _null_alloc: NoCudaAlloc = unsafe { self.value.restore(self.null_alloc) }?;

        // Note: Shallow changes are not reflected back to the CPU

        Ok(ExchangeWrapperOnHost {
            value: self.value,
            device_box: self.device_box,
            locked_cuda_repr: self.locked_cuda_repr,
            move_event: self.move_event,
        })
    }
}

impl<'stream, T: RustToCudaAsync<CudaAllocation: EmptyCudaAlloc>>
    ExchangeWrapperOnDeviceAsync<'stream, T>
{
    /// Moves the data asynchronously back to the host CPU device.
    ///
    /// To avoid aliasing, each CUDA thread only got access to its own shallow
    /// copy of the data. Hence,
    /// - any shallow changes to the data will NOT be reflected back to the CPU
    /// - any deep changes to the data WILL be reflected back to the CPU
    ///
    /// # Errors
    /// Returns a [`rustacuda::error::CudaError`] iff an error occurs inside
    /// CUDA
    pub fn move_to_host_async(
        mut self,
        stream: &'stream Stream,
    ) -> CudaResult<ExchangeWrapperOnHostAsync<'stream, T>> {
        // Reflect deep changes back to the CPU
        let _null_alloc: NoCudaAlloc =
            unsafe { self.value.restore_async(self.null_alloc, stream) }?;

        // Note: Shallow changes are not reflected back to the CPU

        self.move_event.record(stream)?;

        let waker: Arc<Mutex<Option<Waker>>> = Arc::new(Mutex::new(None));

        let waker_callback = waker.clone();
        stream.add_callback(Box::new(move |_| {
            if let Ok(mut w) = waker_callback.lock() {
                if let Some(w) = w.take() {
                    w.wake();
                }
            }
        }))?;

        Ok(ExchangeWrapperOnHostAsync {
            value: self.value,
            device_box: self.device_box,
            locked_cuda_repr: self.locked_cuda_repr,
            move_event: self.move_event,
            stream: PhantomData::<&'stream Stream>,
            waker,
        })
    }
}

impl<'stream, T: RustToCuda<CudaAllocation: EmptyCudaAlloc>> IntoFuture
    for ExchangeWrapperOnDeviceAsync<'stream, T>
{
    type Output = CudaResult<ExchangeWrapperOnDevice<T>>;

    type IntoFuture = impl Future<Output = Self::Output>;

    fn into_future(self) -> Self::IntoFuture {
        let mut wrapper = Some(self);

        core::future::poll_fn(move |cx| match &wrapper {
            Some(inner) => match inner.move_event.query() {
                Ok(EventStatus::NotReady) => inner.waker.lock().map_or_else(
                    |_| Poll::Ready(Err(CudaError::OperatingSystemError)),
                    |mut w| {
                        *w = Some(cx.waker().clone());
                        Poll::Pending
                    },
                ),
                Ok(EventStatus::Ready) => match wrapper.take() {
                    Some(inner) => Poll::Ready(Ok(ExchangeWrapperOnDevice {
                        value: inner.value,
                        device_box: inner.device_box,
                        locked_cuda_repr: inner.locked_cuda_repr,
                        null_alloc: inner.null_alloc,
                        move_event: inner.move_event,
                    })),
                    None => Poll::Ready(Err(CudaError::AlreadyAcquired)),
                },
                Err(err) => Poll::Ready(Err(err)),
            },
            None => Poll::Ready(Err(CudaError::AlreadyAcquired)),
        })
    }
}

impl<T: RustToCuda<CudaAllocation: EmptyCudaAlloc>> ExchangeWrapperOnDevice<T> {
    /// Moves the data synchronously back to the host CPU device.
    ///
    /// To avoid aliasing, each CUDA thread only got access to its own shallow
    /// copy of the data. Hence,
    /// - any shallow changes to the data will NOT be reflected back to the CPU
    /// - any deep changes to the data WILL be reflected back to the CPU
    ///
    /// # Errors
    /// Returns a [`rustacuda::error::CudaError`] iff an error occurs inside
    /// CUDA
    pub fn move_to_host(mut self) -> CudaResult<ExchangeWrapperOnHost<T>> {
        // Reflect deep changes back to the CPU
        let _null_alloc: NoCudaAlloc = unsafe { self.value.restore(self.null_alloc) }?;

        // Note: Shallow changes are not reflected back to the CPU

        Ok(ExchangeWrapperOnHost {
            value: self.value,
            device_box: self.device_box,
            locked_cuda_repr: self.locked_cuda_repr,
            move_event: self.move_event,
        })
    }

    pub fn as_ref(
        &self,
    ) -> HostAndDeviceConstRef<DeviceAccessible<<T as RustToCuda>::CudaRepresentation>> {
        // Safety: `device_box` contains exactly the device copy of `locked_cuda_repr`
        unsafe {
            HostAndDeviceConstRef::new(&self.device_box, (**self.locked_cuda_repr).into_ref())
        }
    }

    pub fn as_mut(
        &mut self,
    ) -> HostAndDeviceMutRef<DeviceAccessible<<T as RustToCuda>::CudaRepresentation>> {
        // Safety: `device_box` contains exactly the device copy of `locked_cuda_repr`
        unsafe {
            HostAndDeviceMutRef::new(&mut self.device_box, (**self.locked_cuda_repr).into_mut())
        }
    }
}

impl<T: RustToCudaAsync<CudaAllocation: EmptyCudaAlloc>> ExchangeWrapperOnDevice<T> {
    /// Moves the data asynchronously back to the host CPU device.
    ///
    /// To avoid aliasing, each CUDA thread only got access to its own shallow
    /// copy of the data. Hence,
    /// - any shallow changes to the data will NOT be reflected back to the CPU
    /// - any deep changes to the data WILL be reflected back to the CPU
    ///
    /// # Errors
    /// Returns a [`rustacuda::error::CudaError`] iff an error occurs inside
    /// CUDA
    pub fn move_to_host_async(
        mut self,
        stream: &Stream,
    ) -> CudaResult<ExchangeWrapperOnHostAsync<'_, T>> {
        // Reflect deep changes back to the CPU
        let _null_alloc: NoCudaAlloc =
            unsafe { self.value.restore_async(self.null_alloc, stream) }?;

        // Note: Shallow changes are not reflected back to the CPU

        self.move_event.record(stream)?;

        let waker: Arc<Mutex<Option<Waker>>> = Arc::new(Mutex::new(None));

        let waker_callback = waker.clone();
        stream.add_callback(Box::new(move |_| {
            if let Ok(mut w) = waker_callback.lock() {
                if let Some(w) = w.take() {
                    w.wake();
                }
            }
        }))?;

        Ok(ExchangeWrapperOnHostAsync {
            value: self.value,
            device_box: self.device_box,
            locked_cuda_repr: self.locked_cuda_repr,
            move_event: self.move_event,
            stream: PhantomData::<&Stream>,
            waker,
        })
    }
}
