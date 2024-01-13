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
};

use crate::{
    safety::PortableBitSemantics,
    utils::{
        adapter::DeviceCopyWithPortableBitSemantics,
        ffi::{
            DeviceConstPointer, DeviceConstRef, DeviceMutPointer, DeviceMutRef, DeviceOwnedPointer,
            DeviceOwnedRef,
        },
        r#async::{Async, NoCompletion},
    },
};

#[repr(transparent)]
pub struct Stream {
    stream: rustacuda::stream::Stream,
}

impl Deref for Stream {
    type Target = rustacuda::stream::Stream;

    fn deref(&self) -> &Self::Target {
        &self.stream
    }
}

impl Stream {
    pub fn with<O>(
        stream: &mut rustacuda::stream::Stream,
        inner: impl for<'stream> FnOnce(&'stream Self) -> O,
    ) -> O {
        // Safety:
        //  - Stream is a newtype wrapper around rustacuda::stream::Stream
        //  - we forge a unique lifetime for a unique reference
        let stream = unsafe { &*std::ptr::from_ref(stream).cast() };

        inner(stream)
    }
}

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
    ($type:ty) => {
        impl CudaDroppable for $type {
            fn drop(val: Self) -> Result<(), (CudaError, Self)> {
                Self::drop(val)
            }
        }
    };
}

impl_sealed_drop_value!(Module);
impl_sealed_drop_value!(rustacuda::stream::Stream);
impl_sealed_drop_value!(Context);
impl_sealed_drop_value!(Event);

#[allow(clippy::module_name_repetitions)]
pub struct HostAndDeviceMutRef<'a, T: PortableBitSemantics + TypeGraphLayout> {
    device_box: &'a mut DeviceBox<DeviceCopyWithPortableBitSemantics<T>>,
    host_ref: &'a mut T,
}

impl<'a, T: PortableBitSemantics + TypeGraphLayout> HostAndDeviceMutRef<'a, T> {
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

    /// # Safety
    ///
    /// `device_box` must contain EXACTLY the device copy of `host_ref`
    pub(crate) unsafe fn new_unchecked(
        device_box: &'a mut DeviceBox<DeviceCopyWithPortableBitSemantics<T>>,
        host_ref: &'a mut T,
    ) -> Self {
        Self {
            device_box,
            host_ref,
        }
    }

    #[must_use]
    pub(crate) fn for_device<'b>(&'b mut self) -> DeviceMutRef<'a, T>
    where
        'a: 'b,
    {
        DeviceMutRef {
            pointer: DeviceMutPointer(self.device_box.as_device_ptr().as_raw_mut().cast()),
            reference: PhantomData,
        }
    }

    #[must_use]
    pub(crate) fn for_host<'b: 'a>(&'b self) -> &'a T {
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
    pub fn into_mut<'b>(self) -> HostAndDeviceMutRef<'b, T>
    where
        'a: 'b,
    {
        HostAndDeviceMutRef {
            device_box: self.device_box,
            host_ref: self.host_ref,
        }
    }

    #[must_use]
    pub fn into_async<'b, 'stream>(
        self,
        stream: &'stream Stream,
    ) -> Async<'b, 'stream, HostAndDeviceMutRef<'b, T>, NoCompletion>
    where
        'a: 'b,
    {
        Async::ready(self.into_mut(), stream)
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

    /// # Safety
    ///
    /// `device_box` must contain EXACTLY the device copy of `host_ref`
    pub(crate) const unsafe fn new_unchecked(
        device_box: &'a DeviceBox<DeviceCopyWithPortableBitSemantics<T>>,
        host_ref: &'a T,
    ) -> Self {
        Self {
            device_box,
            host_ref,
        }
    }

    #[must_use]
    pub(crate) fn for_device<'b>(&'b self) -> DeviceConstRef<'a, T>
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
    pub(crate) const fn for_host(&'a self) -> &'a T {
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
    pub const fn as_async<'b, 'stream>(
        &'b self,
        stream: &'stream Stream,
    ) -> Async<'b, 'stream, HostAndDeviceConstRef<'b, T>, NoCompletion>
    where
        'a: 'b,
    {
        Async::ready(
            HostAndDeviceConstRef {
                device_box: self.device_box,
                host_ref: self.host_ref,
            },
            stream,
        )
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
    pub(crate) fn for_device(self) -> DeviceOwnedRef<'a, T> {
        DeviceOwnedRef {
            pointer: DeviceOwnedPointer(self.device_box.as_device_ptr().as_raw_mut().cast()),
            marker: PhantomData::<T>,
            reference: PhantomData::<&'a mut ()>,
        }
    }

    #[must_use]
    pub(crate) fn for_host(&self) -> &T {
        self.host_val
    }

    #[must_use]
    pub const fn into_async<'stream>(
        self,
        stream: &'stream Stream,
    ) -> Async<'a, 'stream, Self, NoCompletion> {
        Async::ready(self, stream)
    }
}
