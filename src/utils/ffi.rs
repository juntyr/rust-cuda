use core::marker::PhantomData;
#[cfg(feature = "device")]
use core::{
    convert::{AsMut, AsRef},
    ops::{Deref, DerefMut},
};
#[cfg(feature = "host")]
use std::fmt;

use const_type_layout::{TypeGraphLayout, TypeLayout};

use crate::safety::PortableBitSemantics;
#[cfg(feature = "host")]
use crate::{lend::CudaAsRust, utils::adapter::RustToCudaWithPortableBitCopySemantics};

#[cfg_attr(any(feature = "device", doc), derive(Debug))]
#[derive(TypeLayout)]
#[repr(transparent)]
pub struct DeviceAccessible<T: PortableBitSemantics + TypeGraphLayout>(T);

#[cfg(feature = "host")]
impl<T: CudaAsRust> From<T> for DeviceAccessible<T> {
    fn from(value: T) -> Self {
        Self(value)
    }
}

#[cfg(feature = "host")]
impl<T: Copy + PortableBitSemantics + TypeGraphLayout> From<&T>
    for DeviceAccessible<RustToCudaWithPortableBitCopySemantics<T>>
{
    fn from(value: &T) -> Self {
        Self(RustToCudaWithPortableBitCopySemantics::from_copy(value))
    }
}

#[cfg(all(feature = "host", not(doc)))]
impl<T: PortableBitSemantics + TypeGraphLayout + fmt::Debug> fmt::Debug for DeviceAccessible<T> {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        fmt.debug_struct(stringify!(DeviceAccessible))
            .finish_non_exhaustive()
    }
}

#[cfg(feature = "device")]
impl<T: PortableBitSemantics + TypeGraphLayout> Deref for DeviceAccessible<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[cfg(feature = "device")]
impl<T: PortableBitSemantics + TypeGraphLayout> DerefMut for DeviceAccessible<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

#[derive(TypeLayout)]
#[repr(transparent)]
pub struct DeviceConstRef<'r, T: PortableBitSemantics + 'r> {
    #[cfg_attr(feature = "host", allow(dead_code))]
    pub(crate) pointer: DeviceConstPointer<T>,
    pub(crate) reference: PhantomData<&'r T>,
}

impl<'r, T: PortableBitSemantics> Copy for DeviceConstRef<'r, T> {}

impl<'r, T: PortableBitSemantics> Clone for DeviceConstRef<'r, T> {
    fn clone(&self) -> Self {
        *self
    }
}

#[cfg(feature = "device")]
impl<'r, T: PortableBitSemantics> AsRef<T> for DeviceConstRef<'r, T> {
    fn as_ref(&self) -> &T {
        unsafe { &*self.pointer.0 }
    }
}

#[derive(TypeLayout)]
#[repr(transparent)]
pub struct DeviceMutRef<'r, T: PortableBitSemantics + 'r> {
    #[cfg_attr(feature = "host", allow(dead_code))]
    pub(crate) pointer: DeviceMutPointer<T>,
    pub(crate) reference: PhantomData<&'r mut T>,
}

#[cfg(feature = "device")]
impl<'r, T: PortableBitSemantics> AsRef<T> for DeviceMutRef<'r, T> {
    fn as_ref(&self) -> &T {
        unsafe { &*self.pointer.0 }
    }
}

#[cfg(feature = "device")]
impl<'r, T: PortableBitSemantics> AsMut<T> for DeviceMutRef<'r, T> {
    fn as_mut(&mut self) -> &mut T {
        unsafe { &mut *self.pointer.0 }
    }
}

#[derive(TypeLayout)]
#[repr(transparent)]
pub struct DeviceOwnedRef<'r, T: PortableBitSemantics> {
    #[cfg_attr(feature = "host", allow(dead_code))]
    pub(crate) pointer: DeviceOwnedPointer<T>,
    pub(crate) reference: PhantomData<&'r mut ()>,
    pub(crate) marker: PhantomData<T>,
}

#[cfg(feature = "device")]
impl<'r, T: PortableBitSemantics> AsRef<T> for DeviceOwnedRef<'r, T> {
    fn as_ref(&self) -> &T {
        unsafe { &*self.pointer.0 }
    }
}

#[cfg(feature = "device")]
impl<'r, T: PortableBitSemantics> AsMut<T> for DeviceOwnedRef<'r, T> {
    fn as_mut(&mut self) -> &mut T {
        unsafe { &mut *self.pointer.0 }
    }
}

#[derive(TypeLayout)]
#[repr(transparent)]
pub struct DeviceConstPointer<T: ?Sized>(pub(crate) *const T);

impl<T: ?Sized> Copy for DeviceConstPointer<T> {}

impl<T: ?Sized> Clone for DeviceConstPointer<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> DeviceConstPointer<[T]> {
    #[must_use]
    pub fn into_raw_parts(self) -> (DeviceConstPointer<T>, usize) {
        let (data, len) = self.0.to_raw_parts();
        (DeviceConstPointer(data.cast()), len)
    }
}

#[derive(TypeLayout)]
#[repr(transparent)]
pub struct DeviceMutPointer<T: ?Sized>(pub(crate) *mut T);

impl<T: ?Sized> Copy for DeviceMutPointer<T> {}

impl<T: ?Sized> Clone for DeviceMutPointer<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T: ?Sized> DeviceMutPointer<T> {
    #[must_use]
    pub const fn as_const(self) -> DeviceConstPointer<T> {
        DeviceConstPointer(self.0.cast_const())
    }
}

impl<T> DeviceMutPointer<[T]> {
    #[must_use]
    pub fn into_raw_parts(self) -> (DeviceMutPointer<T>, usize) {
        let (data, len) = self.0.to_raw_parts();
        (DeviceMutPointer(data.cast()), len)
    }
}

#[derive(TypeLayout)]
#[repr(transparent)]
pub struct DeviceOwnedPointer<T: ?Sized>(pub(crate) *mut T);

impl<T: ?Sized> Copy for DeviceOwnedPointer<T> {}

impl<T: ?Sized> Clone for DeviceOwnedPointer<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T: ?Sized> DeviceOwnedPointer<T> {
    #[must_use]
    pub const fn as_const(self) -> DeviceConstPointer<T> {
        DeviceConstPointer(self.0.cast_const())
    }

    #[must_use]
    pub const fn as_mut(self) -> DeviceMutPointer<T> {
        DeviceMutPointer(self.0)
    }
}

impl<T> DeviceOwnedPointer<[T]> {
    #[must_use]
    pub fn into_raw_parts(self) -> (DeviceOwnedPointer<T>, usize) {
        let (data, len) = self.0.to_raw_parts();
        (DeviceOwnedPointer(data.cast()), len)
    }
}
