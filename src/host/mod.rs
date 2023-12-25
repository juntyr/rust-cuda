use std::{
    marker::PhantomData,
    mem::ManuallyDrop,
    ops::{Deref, DerefMut},
};

use rustacuda::{
    context::Context,
    error::{CudaError, CudaResult},
    event::Event,
    memory::{DeviceBox, DeviceBuffer, LockedBox, LockedBuffer},
    module::Module,
    stream::Stream,
};
use rustacuda_core::{DeviceCopy, DevicePointer};

use crate::{
    safety::SafeDeviceCopy,
    utils::ffi::{DeviceConstRef, DeviceMutRef, DeviceOwnedRef},
};

pub trait CudaDroppable: Sized {
    #[allow(clippy::missing_errors_doc)]
    fn drop(val: Self) -> Result<(), (rustacuda::error::CudaError, Self)>;
}

#[repr(transparent)]
pub struct CudaDropWrapper<C: CudaDroppable>(ManuallyDrop<C>);
impl<C: CudaDroppable> crate::alloc::CudaAlloc for CudaDropWrapper<C> {}
impl<C: CudaDroppable> crate::alloc::sealed::alloc::Sealed for CudaDropWrapper<C> {}
impl<C: CudaDroppable> From<C> for CudaDropWrapper<C> {
    fn from(val: C) -> Self {
        Self(ManuallyDrop::new(val))
    }
}
impl<C: CudaDroppable> Drop for CudaDropWrapper<C> {
    fn drop(&mut self) {
        // Safety: drop is only ever called once
        let val = unsafe { ManuallyDrop::take(&mut self.0) };

        if let Err((_err, val)) = C::drop(val) {
            core::mem::forget(val);
        }
    }
}
impl<C: CudaDroppable> Deref for CudaDropWrapper<C> {
    type Target = C;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl<C: CudaDroppable> DerefMut for CudaDropWrapper<C> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

macro_rules! impl_sealed_drop_collection {
    ($type:ident) => {
        impl<C: DeviceCopy> CudaDroppable for $type<C> {
            fn drop(val: Self) -> Result<(), (CudaError, Self)> {
                Self::drop(val)
            }
        }
    };
}

impl_sealed_drop_collection!(DeviceBuffer);
impl_sealed_drop_collection!(DeviceBox);
impl_sealed_drop_collection!(LockedBuffer);
impl_sealed_drop_collection!(LockedBox);

macro_rules! impl_sealed_drop_value {
    ($type:ident) => {
        impl CudaDroppable for $type {
            fn drop(val: Self) -> Result<(), (CudaError, Self)> {
                Self::drop(val)
            }
        }
    };
}

impl_sealed_drop_value!(Module);
impl_sealed_drop_value!(Stream);
impl_sealed_drop_value!(Context);
impl_sealed_drop_value!(Event);

#[repr(transparent)]
#[allow(clippy::module_name_repetitions)]
pub struct HostLockedBox<T: DeviceCopy>(*mut T);

impl<T: DeviceCopy> HostLockedBox<T> {
    /// # Errors
    /// Returns a [`CudaError`] iff an error occurs inside CUDA
    pub fn new(value: T) -> CudaResult<Self> {
        // Safety: uninitialised memory is immediately written to without reading it
        let locked_ptr = unsafe {
            let locked_ptr: *mut T = LockedBox::into_raw(LockedBox::uninitialized()?);
            locked_ptr.write(value);
            locked_ptr
        };

        Ok(Self(locked_ptr))
    }
}

impl<T: DeviceCopy> Deref for HostLockedBox<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        unsafe { &*self.0 }
    }
}

impl<T: DeviceCopy> DerefMut for HostLockedBox<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { &mut *self.0 }
    }
}

impl<T: DeviceCopy> From<LockedBox<T>> for HostLockedBox<T> {
    fn from(locked_box: LockedBox<T>) -> Self {
        Self(LockedBox::into_raw(locked_box))
    }
}

impl<T: DeviceCopy> From<HostLockedBox<T>> for LockedBox<T> {
    fn from(host_locked_box: HostLockedBox<T>) -> Self {
        // Safety: pointer comes from [`LockedBox::into_raw`]
        //         i.e. this function completes the roundtrip
        unsafe { Self::from_raw(host_locked_box.0) }
    }
}

impl<T: DeviceCopy> Drop for HostLockedBox<T> {
    fn drop(&mut self) {
        // Safety: pointer comes from [`LockedBox::into_raw`]
        //         i.e. this function completes the roundtrip
        let locked_box = unsafe { LockedBox::from_raw(self.0) };

        core::mem::drop(CudaDropWrapper::from(locked_box));
    }
}

#[repr(transparent)]
#[allow(clippy::module_name_repetitions)]
pub struct HostDeviceBox<T: DeviceCopy>(DevicePointer<T>);

impl<T: DeviceCopy> crate::alloc::CudaAlloc for HostDeviceBox<T> {}
impl<T: DeviceCopy> crate::alloc::sealed::alloc::Sealed for HostDeviceBox<T> {}

impl<T: DeviceCopy> HostDeviceBox<T> {
    /// # Errors
    ///
    /// Returns a [`CudaError`] iff copying from `value` into `self` failed.
    pub fn copy_from(&mut self, value: &T) -> CudaResult<()> {
        // Safety: pointer comes from [`DeviceBox::into_device`]
        //         i.e. this function completes the roundtrip
        let mut device_box = unsafe { ManuallyDrop::new(DeviceBox::from_device(self.0)) };

        rustacuda::memory::CopyDestination::copy_from(&mut *device_box, value)
    }

    /// # Errors
    ///
    /// Returns a [`CudaError`] iff copying from `self` into `value` failed.
    pub fn copy_to(&self, value: &mut T) -> CudaResult<()> {
        // Safety: pointer comes from [`DeviceBox::into_device`]
        //         i.e. this function completes the roundtrip
        let device_box = unsafe { ManuallyDrop::new(DeviceBox::from_device(self.0)) };

        rustacuda::memory::CopyDestination::copy_to(&*device_box, value)
    }

    /// # Errors
    ///
    /// Returns a [`CudaError`] iff copying from `value` into `self` failed.
    ///
    /// # Safety
    ///
    /// To use the data inside the device box, either
    /// - the passed-in [`Stream`] must be synchronised
    /// - the kernel must be launched on the passed-in [`Stream`]
    pub unsafe fn async_copy_from(
        &mut self,
        value: &HostLockedBox<T>,
        stream: &Stream,
    ) -> CudaResult<()> {
        // Safety: pointer comes from [`DeviceBox::into_device`]
        //         i.e. this function completes the roundtrip
        let mut device_box = unsafe { ManuallyDrop::new(DeviceBox::from_device(self.0)) };
        // Safety: pointer comes from [`LockedBox::into_raw`]
        //         i.e. this function completes the roundtrip
        let locked_box = unsafe { ManuallyDrop::new(LockedBox::from_raw(value.0)) };

        unsafe {
            rustacuda::memory::AsyncCopyDestination::async_copy_from(
                &mut *device_box,
                &*locked_box,
                stream,
            )
        }
    }

    /// # Errors
    ///
    /// Returns a [`CudaError`] iff copying from `self` into `value` failed.
    ///
    /// # Safety
    ///
    /// To use the data inside `value`, the passed-in [`Stream`] must be
    /// synchronised.
    pub unsafe fn async_copy_to(
        &self,
        value: &mut HostLockedBox<T>,
        stream: &Stream,
    ) -> CudaResult<()> {
        // Safety: pointer comes from [`DeviceBox::into_device`]
        //         i.e. this function completes the roundtrip
        let device_box = unsafe { ManuallyDrop::new(DeviceBox::from_device(self.0)) };
        // Safety: pointer comes from [`LockedBox::into_raw`]
        //         i.e. this function completes the roundtrip
        let mut locked_box = unsafe { ManuallyDrop::new(LockedBox::from_raw(value.0)) };

        unsafe {
            rustacuda::memory::AsyncCopyDestination::async_copy_to(
                &*device_box,
                &mut *locked_box,
                stream,
            )
        }
    }
}

impl<T: DeviceCopy> From<DeviceBox<T>> for HostDeviceBox<T> {
    fn from(device_box: DeviceBox<T>) -> Self {
        Self(DeviceBox::into_device(device_box))
    }
}

impl<T: DeviceCopy> From<HostDeviceBox<T>> for DeviceBox<T> {
    fn from(host_device_box: HostDeviceBox<T>) -> Self {
        // Safety: pointer comes from [`DeviceBox::into_device`]
        //         i.e. this function completes the roundtrip
        unsafe { Self::from_device(host_device_box.0) }
    }
}

impl<T: DeviceCopy> Drop for HostDeviceBox<T> {
    fn drop(&mut self) {
        // Safety: pointer comes from [`DeviceBox::into_device`]
        //         i.e. this function completes the roundtrip
        let device_box = unsafe { DeviceBox::from_device(self.0) };

        core::mem::drop(CudaDropWrapper::from(device_box));
    }
}

#[allow(clippy::module_name_repetitions)]
pub struct HostAndDeviceMutRef<'a, T: DeviceCopy> {
    device_box: &'a mut HostDeviceBox<T>,
    host_ref: &'a mut T,
}

impl<'a, T: DeviceCopy> HostAndDeviceMutRef<'a, T> {
    /// # Safety
    ///
    /// `device_box` must contain EXACTLY the device copy of `host_ref`
    pub unsafe fn new(device_box: &'a mut HostDeviceBox<T>, host_ref: &'a mut T) -> Self {
        Self {
            device_box,
            host_ref,
        }
    }

    /// # Errors
    ///
    /// Returns a [`CudaError`] iff `value` cannot be moved
    ///  to CUDA or an error occurs inside `inner`.
    pub fn with_new<
        O,
        E: From<CudaError>,
        F: for<'b> FnOnce(HostAndDeviceMutRef<'b, T>) -> Result<O, E>,
    >(
        host_ref: &mut T,
        inner: F,
    ) -> Result<O, E> {
        let mut device_box: HostDeviceBox<_> = DeviceBox::new(host_ref)?.into();

        // Safety: `device_box` contains exactly the device copy of `host_ref`
        let result = inner(HostAndDeviceMutRef {
            device_box: &mut device_box,
            host_ref,
        });

        // Copy back any changes made
        device_box.copy_to(host_ref)?;

        core::mem::drop(device_box);

        result
    }

    #[must_use]
    pub fn for_device<'b>(&'b mut self) -> DeviceMutRef<'a, T>
    where
        'a: 'b,
    {
        DeviceMutRef {
            pointer: self.device_box.0.as_raw_mut(),
            reference: PhantomData,
        }
    }

    #[must_use]
    pub fn for_host<'b: 'a>(&'b self) -> &'a T {
        self.host_ref
    }

    #[must_use]
    pub fn as_ref<'b>(&'b self) -> HostAndDeviceConstRef<'b, T>
    where
        'a: 'b,
    {
        HostAndDeviceConstRef {
            device_box: self.device_box,
            host_ref: self.host_ref,
        }
    }

    #[must_use]
    pub fn as_mut<'b>(&'b mut self) -> HostAndDeviceMutRef<'b, T>
    where
        'a: 'b,
    {
        HostAndDeviceMutRef {
            device_box: self.device_box,
            host_ref: self.host_ref,
        }
    }

    #[must_use]
    pub fn as_async<'stream, 'b>(&'b mut self) -> HostAndDeviceMutRefAsync<'stream, 'b, T>
    where
        'a: 'b,
    {
        HostAndDeviceMutRefAsync {
            device_box: self.device_box,
            host_ref: self.host_ref,
            stream: PhantomData::<&'stream Stream>,
        }
    }
}

#[allow(clippy::module_name_repetitions)]
pub struct HostAndDeviceConstRef<'a, T: DeviceCopy> {
    device_box: &'a HostDeviceBox<T>,
    host_ref: &'a T,
}

impl<'a, T: DeviceCopy> Clone for HostAndDeviceConstRef<'a, T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<'a, T: DeviceCopy> Copy for HostAndDeviceConstRef<'a, T> {}

impl<'a, T: DeviceCopy> HostAndDeviceConstRef<'a, T> {
    /// # Safety
    ///
    /// `device_box` must contain EXACTLY the device copy of `host_ref`
    pub const unsafe fn new(device_box: &'a HostDeviceBox<T>, host_ref: &'a T) -> Self {
        Self {
            device_box,
            host_ref,
        }
    }

    /// # Errors
    ///
    /// Returns a [`CudaError`] iff `value` cannot be moved
    ///  to CUDA or an error occurs inside `inner`.
    pub fn with_new<
        O,
        E: From<CudaError>,
        F: for<'b> FnOnce(HostAndDeviceConstRef<'b, T>) -> Result<O, E>,
    >(
        host_ref: &T,
        inner: F,
    ) -> Result<O, E> {
        let device_box: HostDeviceBox<_> = DeviceBox::new(host_ref)?.into();

        // Safety: `device_box` contains exactly the device copy of `host_ref`
        let result = inner(HostAndDeviceConstRef {
            device_box: &device_box,
            host_ref,
        });

        core::mem::drop(device_box);

        result
    }

    #[must_use]
    pub fn for_device<'b>(&'b self) -> DeviceConstRef<'a, T>
    where
        'a: 'b,
    {
        DeviceConstRef {
            pointer: self.device_box.0.as_raw(),
            reference: PhantomData,
        }
    }

    #[must_use]
    pub const fn for_host(&'a self) -> &'a T {
        self.host_ref
    }

    #[must_use]
    pub const fn as_ref<'b>(&'b self) -> HostAndDeviceConstRef<'b, T>
    where
        'a: 'b,
    {
        *self
    }

    #[must_use]
    pub const fn as_async<'stream, 'b>(&'b self) -> HostAndDeviceConstRefAsync<'stream, 'b, T>
    where
        'a: 'b,
    {
        HostAndDeviceConstRefAsync {
            device_box: self.device_box,
            host_ref: self.host_ref,
            stream: PhantomData::<&'stream Stream>,
        }
    }
}

#[allow(clippy::module_name_repetitions)]
pub struct HostAndDeviceOwned<'a, T: SafeDeviceCopy + DeviceCopy> {
    device_box: &'a mut HostDeviceBox<T>,
    host_val: &'a mut T,
}

impl<'a, T: SafeDeviceCopy + DeviceCopy> HostAndDeviceOwned<'a, T> {
    /// # Errors
    ///
    /// Returns a [`CudaError`] iff `value` cannot be moved
    ///  to CUDA or an error occurs inside `inner`.
    pub fn with_new<O, E: From<CudaError>, F: FnOnce(HostAndDeviceOwned<T>) -> Result<O, E>>(
        mut value: T,
        inner: F,
    ) -> Result<O, E> {
        let mut device_box: HostDeviceBox<_> = DeviceBox::new(&value)?.into();

        // Safety: `device_box` contains exactly the device copy of `value`
        inner(HostAndDeviceOwned {
            device_box: &mut device_box,
            host_val: &mut value,
        })
    }

    #[must_use]
    pub fn for_device(self) -> DeviceOwnedRef<'a, T> {
        DeviceOwnedRef {
            pointer: self.device_box.0.as_raw_mut(),
            marker: PhantomData::<T>,
            reference: PhantomData::<&'a mut ()>,
        }
    }

    #[must_use]
    pub fn for_host(&self) -> &T {
        self.host_val
    }

    #[must_use]
    pub fn into_async<'stream>(self) -> HostAndDeviceOwnedAsync<'stream, 'a, T> {
        HostAndDeviceOwnedAsync {
            device_box: self.device_box,
            host_val: self.host_val,
            stream: PhantomData::<&'stream Stream>,
        }
    }
}

#[allow(clippy::module_name_repetitions)]
pub struct HostAndDeviceMutRefAsync<'stream, 'a, T: DeviceCopy> {
    device_box: &'a mut HostDeviceBox<T>,
    host_ref: &'a mut T,
    stream: PhantomData<&'stream Stream>,
}

impl<'stream, 'a, T: DeviceCopy> HostAndDeviceMutRefAsync<'stream, 'a, T> {
    /// # Safety
    ///
    /// `device_box` must contain EXACTLY the device copy of `host_ref`
    pub unsafe fn new(
        device_box: &'a mut HostDeviceBox<T>,
        host_ref: &'a mut T,
        stream: &'stream Stream,
    ) -> Self {
        let _ = stream;

        Self {
            device_box,
            host_ref,
            stream: PhantomData::<&'stream Stream>,
        }
    }

    #[must_use]
    /// # Safety
    ///
    /// The returned [`DeviceMutRef`] must only be used on the constructed-with
    /// [`Stream`]
    pub unsafe fn for_device_async<'b>(&'b mut self) -> DeviceMutRef<'a, T>
    where
        'a: 'b,
    {
        DeviceMutRef {
            pointer: self.device_box.0.as_raw_mut(),
            reference: PhantomData,
        }
    }

    #[must_use]
    pub fn for_host<'b: 'a>(&'b self) -> &'a T {
        self.host_ref
    }

    #[must_use]
    pub fn as_ref<'b>(&'b self) -> HostAndDeviceConstRefAsync<'stream, 'b, T>
    where
        'a: 'b,
    {
        HostAndDeviceConstRefAsync {
            device_box: self.device_box,
            host_ref: self.host_ref,
            stream: self.stream,
        }
    }

    #[must_use]
    pub fn as_mut<'b>(&'b mut self) -> HostAndDeviceMutRefAsync<'stream, 'b, T>
    where
        'a: 'b,
    {
        HostAndDeviceMutRefAsync {
            device_box: self.device_box,
            host_ref: self.host_ref,
            stream: self.stream,
        }
    }
}

#[allow(clippy::module_name_repetitions)]
pub struct HostAndDeviceConstRefAsync<'stream, 'a, T: DeviceCopy> {
    device_box: &'a HostDeviceBox<T>,
    host_ref: &'a T,
    stream: PhantomData<&'stream Stream>,
}

impl<'stream, 'a, T: DeviceCopy> Clone for HostAndDeviceConstRefAsync<'stream, 'a, T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<'stream, 'a, T: DeviceCopy> Copy for HostAndDeviceConstRefAsync<'stream, 'a, T> {}

impl<'stream, 'a, T: DeviceCopy> HostAndDeviceConstRefAsync<'stream, 'a, T> {
    /// # Safety
    ///
    /// `device_box` must contain EXACTLY the device copy of `host_ref`
    #[must_use]
    pub const unsafe fn new(
        device_box: &'a HostDeviceBox<T>,
        host_ref: &'a T,
        stream: &'stream Stream,
    ) -> Self {
        let _ = stream;

        Self {
            device_box,
            host_ref,
            stream: PhantomData::<&'stream Stream>,
        }
    }

    #[must_use]
    /// # Safety
    ///
    /// The returned [`DeviceConstRef`] must only be used on the
    /// constructed-with [`Stream`]
    pub unsafe fn for_device_async<'b>(&'b self) -> DeviceConstRef<'a, T>
    where
        'a: 'b,
    {
        DeviceConstRef {
            pointer: self.device_box.0.as_raw(),
            reference: PhantomData,
        }
    }

    #[must_use]
    pub const fn for_host(&'a self) -> &'a T {
        self.host_ref
    }

    #[must_use]
    pub const fn as_ref<'b>(&'b self) -> HostAndDeviceConstRefAsync<'stream, 'b, T>
    where
        'a: 'b,
    {
        *self
    }
}

#[allow(clippy::module_name_repetitions)]
pub struct HostAndDeviceOwnedAsync<'stream, 'a, T: SafeDeviceCopy + DeviceCopy> {
    device_box: &'a mut HostDeviceBox<T>,
    host_val: &'a mut T,
    stream: PhantomData<&'stream Stream>,
}

impl<'stream, 'a, T: SafeDeviceCopy + DeviceCopy> HostAndDeviceOwnedAsync<'stream, 'a, T> {
    #[must_use]
    /// # Safety
    ///
    /// The returned [`DeviceOwnedRef`] must only be used on the
    /// constructed-with [`Stream`]
    pub unsafe fn for_device_async(self) -> DeviceOwnedRef<'a, T> {
        DeviceOwnedRef {
            pointer: self.device_box.0.as_raw_mut(),
            marker: PhantomData::<T>,
            reference: PhantomData::<&'a mut ()>,
        }
    }

    #[must_use]
    pub fn for_host(&self) -> &T {
        self.host_val
    }
}
