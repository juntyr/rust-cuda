use std::{
    marker::PhantomData,
    mem::ManuallyDrop,
    ops::{Deref, DerefMut},
};

use const_type_layout::TypeGraphLayout;
use rustacuda::{
    context::Context,
    error::CudaError,
    event::Event,
    memory::{CopyDestination, DeviceBox, DeviceBuffer, LockedBox, LockedBuffer},
    module::Module,
    stream::Stream,
};

use crate::{
    safety::PortableBitSemantics,
    utils::{
        adapter::DeviceCopyWithPortableBitSemantics,
        ffi::{
            DeviceConstPointer, DeviceConstRef, DeviceMutPointer, DeviceMutRef, DeviceOwnedPointer,
            DeviceOwnedRef,
        },
    },
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

impl<T> CudaDroppable for DeviceBox<T> {
    fn drop(val: Self) -> Result<(), (CudaError, Self)> {
        Self::drop(val)
    }
}

impl<T: rustacuda_core::DeviceCopy> CudaDroppable for DeviceBuffer<T> {
    fn drop(val: Self) -> Result<(), (CudaError, Self)> {
        Self::drop(val)
    }
}

impl<T> CudaDroppable for LockedBox<T> {
    fn drop(val: Self) -> Result<(), (CudaError, Self)> {
        Self::drop(val)
    }
}

impl<T: rustacuda_core::DeviceCopy> CudaDroppable for LockedBuffer<T> {
    fn drop(val: Self) -> Result<(), (CudaError, Self)> {
        Self::drop(val)
    }
}

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

#[allow(clippy::module_name_repetitions)]
pub struct HostAndDeviceMutRef<'a, T: PortableBitSemantics + TypeGraphLayout> {
    device_box: &'a mut DeviceBox<DeviceCopyWithPortableBitSemantics<T>>,
    host_ref: &'a mut T,
}

impl<'a, T: PortableBitSemantics + TypeGraphLayout> HostAndDeviceMutRef<'a, T> {
    /// # Safety
    ///
    /// `device_box` must contain EXACTLY the device copy of `host_ref`
    pub unsafe fn new(
        device_box: &'a mut DeviceBox<DeviceCopyWithPortableBitSemantics<T>>,
        host_ref: &'a mut T,
    ) -> Self {
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
        let mut device_box = CudaDropWrapper::from(DeviceBox::new(
            DeviceCopyWithPortableBitSemantics::from_ref(host_ref),
        )?);

        // Safety: `device_box` contains exactly the device copy of `host_ref`
        let result = inner(HostAndDeviceMutRef {
            device_box: &mut device_box,
            host_ref,
        });

        // Copy back any changes made
        device_box.copy_to(DeviceCopyWithPortableBitSemantics::from_mut(host_ref))?;

        core::mem::drop(device_box);

        result
    }

    #[must_use]
    pub fn for_device<'b>(&'b mut self) -> DeviceMutRef<'a, T>
    where
        'a: 'b,
    {
        DeviceMutRef {
            pointer: DeviceMutPointer(self.device_box.as_device_ptr().as_raw_mut().cast()),
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
pub struct HostAndDeviceConstRef<'a, T: PortableBitSemantics + TypeGraphLayout> {
    device_box: &'a DeviceBox<DeviceCopyWithPortableBitSemantics<T>>,
    host_ref: &'a T,
}

impl<'a, T: PortableBitSemantics + TypeGraphLayout> Clone for HostAndDeviceConstRef<'a, T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<'a, T: PortableBitSemantics + TypeGraphLayout> Copy for HostAndDeviceConstRef<'a, T> {}

impl<'a, T: PortableBitSemantics + TypeGraphLayout> HostAndDeviceConstRef<'a, T> {
    /// # Safety
    ///
    /// `device_box` must contain EXACTLY the device copy of `host_ref`
    pub const unsafe fn new(
        device_box: &'a DeviceBox<DeviceCopyWithPortableBitSemantics<T>>,
        host_ref: &'a T,
    ) -> Self {
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
        let device_box = CudaDropWrapper::from(DeviceBox::new(
            DeviceCopyWithPortableBitSemantics::from_ref(host_ref),
        )?);

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
        let mut hack = ManuallyDrop::new(unsafe { std::ptr::read(self.device_box) });

        DeviceConstRef {
            pointer: DeviceConstPointer(hack.as_device_ptr().as_raw().cast()),
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
pub struct HostAndDeviceOwned<'a, T: PortableBitSemantics + TypeGraphLayout> {
    device_box: &'a mut DeviceBox<DeviceCopyWithPortableBitSemantics<T>>,
    host_val: &'a mut T,
}

impl<'a, T: PortableBitSemantics + TypeGraphLayout> HostAndDeviceOwned<'a, T> {
    /// # Errors
    ///
    /// Returns a [`CudaError`] iff `value` cannot be moved
    ///  to CUDA or an error occurs inside `inner`.
    pub fn with_new<O, E: From<CudaError>, F: FnOnce(HostAndDeviceOwned<T>) -> Result<O, E>>(
        mut value: T,
        inner: F,
    ) -> Result<O, E> {
        let mut device_box = CudaDropWrapper::from(DeviceBox::new(
            DeviceCopyWithPortableBitSemantics::from_ref(&value),
        )?);

        // Safety: `device_box` contains exactly the device copy of `value`
        inner(HostAndDeviceOwned {
            device_box: &mut device_box,
            host_val: &mut value,
        })
    }

    #[must_use]
    pub fn for_device(self) -> DeviceOwnedRef<'a, T> {
        DeviceOwnedRef {
            pointer: DeviceOwnedPointer(self.device_box.as_device_ptr().as_raw_mut().cast()),
            marker: PhantomData::<T>,
            reference: PhantomData::<&'a mut ()>,
        }
    }

    #[must_use]
    pub fn for_host(&self) -> &T {
        self.host_val
    }

    #[must_use]
    pub(crate) fn for_async_completion(&mut self) -> &mut T {
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
pub struct HostAndDeviceMutRefAsync<'stream, 'a, T: PortableBitSemantics + TypeGraphLayout> {
    device_box: &'a mut DeviceBox<DeviceCopyWithPortableBitSemantics<T>>,
    host_ref: &'a mut T,
    stream: PhantomData<&'stream Stream>,
}

impl<'stream, 'a, T: PortableBitSemantics + TypeGraphLayout>
    HostAndDeviceMutRefAsync<'stream, 'a, T>
{
    /// # Safety
    ///
    /// `device_box` must contain EXACTLY the device copy of `host_ref`
    pub unsafe fn new(
        device_box: &'a mut DeviceBox<DeviceCopyWithPortableBitSemantics<T>>,
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
            pointer: DeviceMutPointer(self.device_box.as_device_ptr().as_raw_mut().cast()),
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
pub struct HostAndDeviceConstRefAsync<'stream, 'a, T: PortableBitSemantics + TypeGraphLayout> {
    device_box: &'a DeviceBox<DeviceCopyWithPortableBitSemantics<T>>,
    host_ref: &'a T,
    stream: PhantomData<&'stream Stream>,
}

impl<'stream, 'a, T: PortableBitSemantics + TypeGraphLayout> Clone
    for HostAndDeviceConstRefAsync<'stream, 'a, T>
{
    fn clone(&self) -> Self {
        *self
    }
}

impl<'stream, 'a, T: PortableBitSemantics + TypeGraphLayout> Copy
    for HostAndDeviceConstRefAsync<'stream, 'a, T>
{
}

impl<'stream, 'a, T: PortableBitSemantics + TypeGraphLayout>
    HostAndDeviceConstRefAsync<'stream, 'a, T>
{
    /// # Safety
    ///
    /// `device_box` must contain EXACTLY the device copy of `host_ref`
    #[must_use]
    pub const unsafe fn new(
        device_box: &'a DeviceBox<DeviceCopyWithPortableBitSemantics<T>>,
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
        let mut hack = ManuallyDrop::new(unsafe { std::ptr::read(self.device_box) });

        DeviceConstRef {
            pointer: DeviceConstPointer(hack.as_device_ptr().as_raw().cast()),
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
pub struct HostAndDeviceOwnedAsync<'stream, 'a, T: PortableBitSemantics + TypeGraphLayout> {
    device_box: &'a mut DeviceBox<DeviceCopyWithPortableBitSemantics<T>>,
    host_val: &'a mut T,
    stream: PhantomData<&'stream Stream>,
}

impl<'stream, 'a, T: PortableBitSemantics + TypeGraphLayout>
    HostAndDeviceOwnedAsync<'stream, 'a, T>
{
    #[must_use]
    /// # Safety
    ///
    /// The returned [`DeviceOwnedRef`] must only be used on the
    /// constructed-with [`Stream`]
    pub unsafe fn for_device_async(self) -> DeviceOwnedRef<'a, T> {
        DeviceOwnedRef {
            pointer: DeviceOwnedPointer(self.device_box.as_device_ptr().as_raw_mut().cast()),
            marker: PhantomData::<T>,
            reference: PhantomData::<&'a mut ()>,
        }
    }

    #[must_use]
    pub fn for_host(&self) -> &T {
        self.host_val
    }
}
