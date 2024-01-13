use std::ops::{Deref, DerefMut};

use rustacuda::{
    error::CudaResult,
    memory::{AsyncCopyDestination, CopyDestination, DeviceBox, LockedBox},
};

use crate::{
    alloc::{EmptyCudaAlloc, NoCudaAlloc},
    host::{CudaDropWrapper, HostAndDeviceConstRef, HostAndDeviceMutRef, Stream},
    lend::{RustToCuda, RustToCudaAsync},
    safety::SafeMutableAliasing,
    utils::{
        adapter::DeviceCopyWithPortableBitSemantics,
        ffi::DeviceAccessible,
        r#async::{Async, AsyncProj, CompletionFnMut, NoCompletion},
    },
};

#[allow(clippy::module_name_repetitions)]
pub struct ExchangeWrapperOnHost<T: RustToCuda<CudaAllocation: EmptyCudaAlloc>> {
    value: Box<T>,
    device_box: CudaDropWrapper<
        DeviceBox<
            DeviceCopyWithPortableBitSemantics<
                DeviceAccessible<<T as RustToCuda>::CudaRepresentation>,
            >,
        >,
    >,
    locked_cuda_repr: CudaDropWrapper<
        LockedBox<
            DeviceCopyWithPortableBitSemantics<
                DeviceAccessible<<T as RustToCuda>::CudaRepresentation>,
            >,
        >,
    >,
}

#[allow(clippy::module_name_repetitions)]
pub struct ExchangeWrapperOnDevice<T: RustToCuda<CudaAllocation: EmptyCudaAlloc>> {
    value: Box<T>,
    device_box: CudaDropWrapper<
        DeviceBox<
            DeviceCopyWithPortableBitSemantics<
                DeviceAccessible<<T as RustToCuda>::CudaRepresentation>,
            >,
        >,
    >,
    locked_cuda_repr: CudaDropWrapper<
        LockedBox<
            DeviceCopyWithPortableBitSemantics<
                DeviceAccessible<<T as RustToCuda>::CudaRepresentation>,
            >,
        >,
    >,
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
                DeviceCopyWithPortableBitSemantics<
                    DeviceAccessible<<T as RustToCuda>::CudaRepresentation>,
                >,
            >::uninitialized()?);
            uninit
                .as_mut_ptr()
                .write(DeviceCopyWithPortableBitSemantics::from(cuda_repr));
            uninit
        };

        Ok(Self {
            value: Box::new(value),
            device_box,
            locked_cuda_repr,
        })
    }

    /// Moves the data synchronously to the CUDA device, where it can then be
    /// lent out immutably via [`ExchangeWrapperOnDevice::as_ref`], or mutably
    /// via [`ExchangeWrapperOnDevice::as_mut`].
    ///
    /// # Errors
    /// Returns a [`rustacuda::error::CudaError`] iff an error occurs inside
    /// CUDA
    pub fn move_to_device(mut self) -> CudaResult<ExchangeWrapperOnDevice<T>> {
        let (cuda_repr, null_alloc) = unsafe { self.value.borrow(NoCudaAlloc) }?;
        **self.locked_cuda_repr = DeviceCopyWithPortableBitSemantics::from(cuda_repr);

        self.device_box.copy_from(&**self.locked_cuda_repr)?;

        let _: NoCudaAlloc = null_alloc.into();

        Ok(ExchangeWrapperOnDevice {
            value: self.value,
            device_box: self.device_box,
            locked_cuda_repr: self.locked_cuda_repr,
        })
    }
}

impl<T: RustToCudaAsync<CudaAllocationAsync: EmptyCudaAlloc, CudaAllocation: EmptyCudaAlloc>>
    ExchangeWrapperOnHost<T>
{
    #[allow(clippy::needless_lifetimes)] // keep 'stream explicit
    /// Moves the data asynchronously to the CUDA device.
    ///
    /// # Errors
    /// Returns a [`rustacuda::error::CudaError`] iff an error occurs inside
    /// CUDA
    pub fn move_to_device_async<'stream>(
        mut self,
        stream: &'stream Stream,
    ) -> CudaResult<Async<'static, 'stream, ExchangeWrapperOnDevice<T>, NoCompletion>> {
        let (cuda_repr, _null_alloc) = unsafe { self.value.borrow_async(NoCudaAlloc, stream) }?;
        let (cuda_repr, _completion): (_, Option<NoCompletion>) =
            unsafe { cuda_repr.unwrap_unchecked()? };

        **self.locked_cuda_repr = DeviceCopyWithPortableBitSemantics::from(cuda_repr);

        // Safety: The device value is not safely exposed until either
        // - the passed-in [`Stream`] is synchronised
        // - the kernel is launched on the passed-in [`Stream`]
        unsafe {
            self.device_box
                .async_copy_from(&*self.locked_cuda_repr, stream)
        }?;

        Async::pending(
            ExchangeWrapperOnDevice {
                value: self.value,
                device_box: self.device_box,
                locked_cuda_repr: self.locked_cuda_repr,
            },
            stream,
            NoCompletion,
        )
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

impl<T: RustToCuda<CudaAllocation: EmptyCudaAlloc>> ExchangeWrapperOnDevice<T> {
    /// Moves the data synchronously back to the host CPU device.
    ///
    /// # Errors
    /// Returns a [`rustacuda::error::CudaError`] iff an error occurs inside
    /// CUDA
    pub fn move_to_host(mut self) -> CudaResult<ExchangeWrapperOnHost<T>> {
        let null_alloc = NoCudaAlloc.into();

        // Reflect deep changes back to the CPU
        let _null_alloc: NoCudaAlloc = unsafe { self.value.restore(null_alloc) }?;

        // Note: Shallow changes are not reflected back to the CPU

        Ok(ExchangeWrapperOnHost {
            value: self.value,
            device_box: self.device_box,
            locked_cuda_repr: self.locked_cuda_repr,
        })
    }

    #[must_use]
    pub fn as_ref(
        &self,
    ) -> HostAndDeviceConstRef<DeviceAccessible<<T as RustToCuda>::CudaRepresentation>> {
        // Safety: `device_box` contains exactly the device copy of `locked_cuda_repr`
        unsafe {
            HostAndDeviceConstRef::new_unchecked(
                &self.device_box,
                (**self.locked_cuda_repr).into_ref(),
            )
        }
    }
}

impl<T: RustToCudaAsync<CudaAllocationAsync: EmptyCudaAlloc, CudaAllocation: EmptyCudaAlloc>>
    ExchangeWrapperOnDevice<T>
{
    #[allow(clippy::needless_lifetimes)] // keep 'stream explicit
    /// Moves the data asynchronously back to the host CPU device.
    ///
    /// # Errors
    /// Returns a [`rustacuda::error::CudaError`] iff an error occurs inside
    /// CUDA
    pub fn move_to_host_async<'stream>(
        self,
        stream: &'stream Stream,
    ) -> CudaResult<
        Async<
            'static,
            'stream,
            ExchangeWrapperOnHost<T>,
            CompletionFnMut<'static, ExchangeWrapperOnHost<T>>,
        >,
    > {
        let null_alloc = NoCudaAlloc.into();

        let value = owning_ref::BoxRefMut::new(self.value);

        // Reflect deep changes back to the CPU
        let (r#async, _null_alloc): (_, NoCudaAlloc) =
            unsafe { RustToCudaAsync::restore_async(value, null_alloc, stream) }?;
        let (value, on_complete) = unsafe { r#async.unwrap_unchecked()? };

        let value = value.into_owner();

        // Note: Shallow changes are not reflected back to the CPU

        if let Some(on_complete) = on_complete {
            Async::<_, CompletionFnMut<ExchangeWrapperOnHost<T>>>::pending(
                ExchangeWrapperOnHost {
                    value,
                    device_box: self.device_box,
                    locked_cuda_repr: self.locked_cuda_repr,
                },
                stream,
                Box::new(|on_host: &mut ExchangeWrapperOnHost<T>| on_complete(&mut on_host.value)),
            )
        } else {
            Ok(Async::ready(
                ExchangeWrapperOnHost {
                    value,
                    device_box: self.device_box,
                    locked_cuda_repr: self.locked_cuda_repr,
                },
                stream,
            ))
        }
    }
}

impl<
        'a,
        'stream,
        T: RustToCudaAsync<CudaAllocationAsync: EmptyCudaAlloc, CudaAllocation: EmptyCudaAlloc>,
    > Async<'a, 'stream, ExchangeWrapperOnDevice<T>, NoCompletion>
{
    /// Moves the data asynchronously back to the host CPU device.
    ///
    /// # Errors
    /// Returns a [`rustacuda::error::CudaError`] iff an error occurs inside
    /// CUDA
    pub fn move_to_host_async(
        self,
        stream: &'stream Stream,
    ) -> CudaResult<
        Async<
            'static,
            'stream,
            ExchangeWrapperOnHost<T>,
            CompletionFnMut<'static, ExchangeWrapperOnHost<T>>,
        >,
    > {
        let (this, completion): (_, Option<NoCompletion>) = unsafe { self.unwrap_unchecked()? };

        let null_alloc = NoCudaAlloc.into();

        let value = owning_ref::BoxRefMut::new(this.value);

        // Reflect deep changes back to the CPU
        let (r#async, _null_alloc): (_, NoCudaAlloc) =
            unsafe { RustToCudaAsync::restore_async(value, null_alloc, stream) }?;
        let (value, on_complete) = unsafe { r#async.unwrap_unchecked()? };

        let value = value.into_owner();

        // Note: Shallow changes are not reflected back to the CPU

        let on_host = ExchangeWrapperOnHost {
            value,
            device_box: this.device_box,
            locked_cuda_repr: this.locked_cuda_repr,
        };

        if let Some(on_complete) = on_complete {
            Async::<_, CompletionFnMut<ExchangeWrapperOnHost<T>>>::pending(
                on_host,
                stream,
                Box::new(|on_host: &mut ExchangeWrapperOnHost<T>| on_complete(&mut on_host.value)),
            )
        } else if matches!(completion, Some(NoCompletion)) {
            Async::<_, CompletionFnMut<ExchangeWrapperOnHost<T>>>::pending(
                on_host,
                stream,
                Box::new(|_on_host: &mut ExchangeWrapperOnHost<T>| Ok(())),
            )
        } else {
            Ok(Async::ready(on_host, stream))
        }
    }

    #[must_use]
    pub fn as_ref_async(
        &self,
    ) -> AsyncProj<
        '_,
        'stream,
        HostAndDeviceConstRef<DeviceAccessible<<T as RustToCuda>::CudaRepresentation>>,
    > {
        let this = unsafe { self.as_ref().unwrap_unchecked() };

        // Safety: this projection captures this async
        unsafe {
            AsyncProj::new(
                HostAndDeviceConstRef::new_unchecked(
                    &*(this.device_box),
                    (**(this.locked_cuda_repr)).into_ref(),
                ),
                None,
            )
        }
    }

    #[must_use]
    pub fn as_mut_async(
        &mut self,
    ) -> AsyncProj<
        '_,
        'stream,
        HostAndDeviceMutRef<DeviceAccessible<<T as RustToCuda>::CudaRepresentation>>,
    >
    where
        T: SafeMutableAliasing,
    {
        let (this, use_callback) = unsafe { self.as_mut().unwrap_unchecked_with_use() };

        // Safety: this projection captures this async
        unsafe {
            AsyncProj::new(
                HostAndDeviceMutRef::new_unchecked(
                    &mut *(this.device_box),
                    (**(this.locked_cuda_repr)).into_mut(),
                ),
                use_callback,
            )
        }
    }
}
