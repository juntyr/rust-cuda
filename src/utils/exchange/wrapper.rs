use core::{
    marker::PhantomData,
    ops::{Deref, DerefMut},
};

use rustacuda::{
    error::CudaResult,
    event::{Event, EventFlags},
    memory::DeviceBox,
    stream::Stream,
};

use crate::{
    common::{DeviceAccessible, RustToCuda, RustToCudaAsync},
    host::{
        CombinedCudaAlloc, CudaDropWrapper, EmptyCudaAlloc, HostAndDeviceConstRef,
        HostAndDeviceMutRef, HostDeviceBox, HostLockedBox, NullCudaAlloc,
    },
};

#[allow(clippy::module_name_repetitions)]
pub struct ExchangeWrapperOnHost<T: RustToCuda<CudaAllocation: EmptyCudaAlloc>> {
    value: T,
    device_box: HostDeviceBox<DeviceAccessible<<T as RustToCuda>::CudaRepresentation>>,
    locked_cuda_repr: HostLockedBox<DeviceAccessible<<T as RustToCuda>::CudaRepresentation>>,
    move_event: CudaDropWrapper<Event>,
}

#[allow(clippy::module_name_repetitions)]
pub struct ExchangeWrapperOnHostAsync<'stream, T: RustToCuda<CudaAllocation: EmptyCudaAlloc>> {
    value: T,
    device_box: HostDeviceBox<DeviceAccessible<<T as RustToCuda>::CudaRepresentation>>,
    locked_cuda_repr: HostLockedBox<DeviceAccessible<<T as RustToCuda>::CudaRepresentation>>,
    move_event: CudaDropWrapper<Event>,
    stream: PhantomData<&'stream Stream>,
}

#[allow(clippy::module_name_repetitions)]
pub struct ExchangeWrapperOnDevice<T: RustToCuda<CudaAllocation: EmptyCudaAlloc>> {
    value: T,
    device_box: HostDeviceBox<DeviceAccessible<<T as RustToCuda>::CudaRepresentation>>,
    locked_cuda_repr: HostLockedBox<DeviceAccessible<<T as RustToCuda>::CudaRepresentation>>,
    null_alloc: CombinedCudaAlloc<<T as RustToCuda>::CudaAllocation, NullCudaAlloc>,
    move_event: CudaDropWrapper<Event>,
}

#[allow(clippy::module_name_repetitions)]
pub struct ExchangeWrapperOnDeviceAsync<'stream, T: RustToCuda<CudaAllocation: EmptyCudaAlloc>> {
    value: T,
    device_box: HostDeviceBox<DeviceAccessible<<T as RustToCuda>::CudaRepresentation>>,
    locked_cuda_repr: HostLockedBox<DeviceAccessible<<T as RustToCuda>::CudaRepresentation>>,
    null_alloc: CombinedCudaAlloc<<T as RustToCuda>::CudaAllocation, NullCudaAlloc>,
    move_event: CudaDropWrapper<Event>,
    stream: PhantomData<&'stream Stream>,
}

impl<T: RustToCuda<CudaAllocation: EmptyCudaAlloc>> ExchangeWrapperOnHost<T> {
    /// # Errors
    /// Returns a [`rustacuda::error::CudaError`] iff an error occurs inside
    /// CUDA
    pub fn new(value: T) -> CudaResult<Self> {
        // Safety: The uninitialised memory is never exposed
        //         To access the device memory, [`Self::move_to_device`] has to be
        // called first,           which initialised the memory.
        let device_box = unsafe { DeviceBox::uninitialized() }?.into();

        let (cuda_repr, _null_alloc) = unsafe { value.borrow(NullCudaAlloc) }?;
        let locked_cuda_repr = HostLockedBox::new(cuda_repr)?;

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
        let (cuda_repr, null_alloc) = unsafe { self.value.borrow(NullCudaAlloc) }?;
        *self.locked_cuda_repr = cuda_repr;

        self.device_box.copy_from(&self.locked_cuda_repr)?;

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
    pub fn move_to_device_async<'stream>(
        mut self,
        stream: &'stream Stream,
    ) -> CudaResult<ExchangeWrapperOnDeviceAsync<'stream, T>> {
        let (cuda_repr, null_alloc) = unsafe { self.value.borrow_async(NullCudaAlloc, stream) }?;
        *self.locked_cuda_repr = cuda_repr;

        // Safety: The device value is not safely exposed until either
        // - the passed-in [`Stream`] is synchronised
        // - the kernel is launched on the passed-in [`Stream`]
        unsafe {
            self.device_box
                .async_copy_from(&self.locked_cuda_repr, stream)
        }?;
        self.move_event.record(stream)?;

        Ok(ExchangeWrapperOnDeviceAsync {
            value: self.value,
            device_box: self.device_box,
            locked_cuda_repr: self.locked_cuda_repr,
            null_alloc,
            move_event: self.move_event,
            stream: PhantomData::<&'stream Stream>,
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
        let _null_alloc: NullCudaAlloc = unsafe { self.value.restore(self.null_alloc) }?;

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
        unsafe { HostAndDeviceConstRef::new(&self.device_box, &self.locked_cuda_repr) }
    }

    pub fn as_mut(
        &mut self,
    ) -> HostAndDeviceMutRef<DeviceAccessible<<T as RustToCuda>::CudaRepresentation>> {
        // Safety: `device_box` contains exactly the device copy of `locked_cuda_repr`
        unsafe { HostAndDeviceMutRef::new(&mut self.device_box, &mut self.locked_cuda_repr) }
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
    pub fn move_to_host_async<'stream>(
        mut self,
        stream: &'stream Stream,
    ) -> CudaResult<ExchangeWrapperOnHostAsync<'stream, T>> {
        // Reflect deep changes back to the CPU
        let _null_alloc: NullCudaAlloc =
            unsafe { self.value.restore_async(self.null_alloc, stream) }?;

        // Note: Shallow changes are not reflected back to the CPU

        self.move_event.record(stream)?;

        Ok(ExchangeWrapperOnHostAsync {
            value: self.value,
            device_box: self.device_box,
            locked_cuda_repr: self.locked_cuda_repr,
            move_event: self.move_event,
            stream: PhantomData::<&'stream Stream>,
        })
    }
}
