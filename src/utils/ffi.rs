use core::marker::PhantomData;
#[cfg(feature = "device")]
use core::{
    convert::{AsMut, AsRef},
    ops::{Deref, DerefMut},
};
#[cfg(feature = "host")]
use std::{fmt, mem::MaybeUninit, ptr::copy_nonoverlapping};

#[cfg(feature = "host")]
use const_type_layout::TypeGraphLayout;
use const_type_layout::TypeLayout;
use rustacuda_core::DeviceCopy;

#[cfg(feature = "host")]
use crate::{lend::CudaAsRust, safety::SafeDeviceCopy, utils::device_copy::SafeDeviceCopyWrapper};

#[repr(transparent)]
#[cfg_attr(any(feature = "device", doc), derive(Debug))]
#[derive(TypeLayout)]
pub struct DeviceAccessible<T: ?Sized + DeviceCopy>(T);

unsafe impl<T: ?Sized + DeviceCopy> DeviceCopy for DeviceAccessible<T> {}

#[cfg(feature = "host")]
impl<T: CudaAsRust> From<T> for DeviceAccessible<T> {
    fn from(value: T) -> Self {
        Self(value)
    }
}

#[cfg(feature = "host")]
impl<T: SafeDeviceCopy + TypeGraphLayout> From<&T> for DeviceAccessible<SafeDeviceCopyWrapper<T>> {
    fn from(value: &T) -> Self {
        let value = unsafe {
            let mut uninit = MaybeUninit::uninit();
            copy_nonoverlapping(value, uninit.as_mut_ptr(), 1);
            uninit.assume_init()
        };

        Self(SafeDeviceCopyWrapper::from(value))
    }
}

#[cfg(all(feature = "host", not(doc)))]
impl<T: ?Sized + DeviceCopy + fmt::Debug> fmt::Debug for DeviceAccessible<T> {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        fmt.debug_struct(stringify!(DeviceAccessible))
            .finish_non_exhaustive()
    }
}

#[cfg(feature = "device")]
impl<T: ?Sized + DeviceCopy> Deref for DeviceAccessible<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[cfg(feature = "device")]
impl<T: ?Sized + DeviceCopy> DerefMut for DeviceAccessible<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

#[repr(transparent)]
#[derive(Clone, Copy, TypeLayout)]
pub struct DeviceConstRef<'r, T: DeviceCopy + 'r> {
    #[cfg_attr(feature = "host", allow(dead_code))]
    pub(crate) pointer: *const T,
    pub(crate) reference: PhantomData<&'r T>,
}

unsafe impl<'r, T: DeviceCopy> DeviceCopy for DeviceConstRef<'r, T> {}

#[cfg(feature = "device")]
impl<'r, T: DeviceCopy> AsRef<T> for DeviceConstRef<'r, T> {
    fn as_ref(&self) -> &T {
        unsafe { &*self.pointer }
    }
}

#[repr(transparent)]
#[derive(TypeLayout)]
pub struct DeviceMutRef<'r, T: DeviceCopy + 'r> {
    #[cfg_attr(feature = "host", allow(dead_code))]
    pub(crate) pointer: *mut T,
    pub(crate) reference: PhantomData<&'r mut T>,
}

unsafe impl<'r, T: DeviceCopy> DeviceCopy for DeviceMutRef<'r, T> {}

#[cfg(feature = "device")]
impl<'r, T: DeviceCopy> AsRef<T> for DeviceMutRef<'r, T> {
    fn as_ref(&self) -> &T {
        unsafe { &*self.pointer }
    }
}

#[cfg(feature = "device")]
impl<'r, T: DeviceCopy> AsMut<T> for DeviceMutRef<'r, T> {
    fn as_mut(&mut self) -> &mut T {
        unsafe { &mut *self.pointer }
    }
}

#[repr(transparent)]
#[derive(TypeLayout)]
pub struct DeviceOwnedRef<'r, T: DeviceCopy> {
    #[cfg_attr(feature = "host", allow(dead_code))]
    pub(crate) pointer: *mut T,
    pub(crate) reference: PhantomData<&'r mut ()>,
    pub(crate) marker: PhantomData<T>,
}

unsafe impl<'r, T: DeviceCopy> DeviceCopy for DeviceOwnedRef<'r, T> {}

#[cfg(feature = "device")]
impl<'r, T: DeviceCopy> AsRef<T> for DeviceOwnedRef<'r, T> {
    fn as_ref(&self) -> &T {
        unsafe { &*self.pointer }
    }
}

#[cfg(feature = "device")]
impl<'r, T: DeviceCopy> AsMut<T> for DeviceOwnedRef<'r, T> {
    fn as_mut(&mut self) -> &mut T {
        unsafe { &mut *self.pointer }
    }
}
