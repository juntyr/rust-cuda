#[cfg(any(not(feature = "host"), doc))]
use core::convert::{AsMut, AsRef};

#[cfg(feature = "host")]
use alloc::fmt;
#[cfg(all(not(feature = "host")))]
use core::ops::{Deref, DerefMut};
#[cfg(feature = "host")]
use core::{mem::MaybeUninit, ptr::copy_nonoverlapping};

use rustacuda_core::DeviceCopy;

#[cfg(feature = "derive")]
#[doc(cfg(feature = "derive"))]
pub use rust_cuda_derive::RustToCudaAsRust;

#[cfg(feature = "derive")]
#[doc(cfg(feature = "derive"))]
pub use rust_cuda_derive::kernel;

#[cfg(feature = "host")]
use crate::utils::stack::{StackOnly, StackOnlyWrapper};

#[doc(hidden)]
pub mod r#impl;

mod sealed;
mod specs;
mod wrapper;

pub use wrapper::{CudaAsRust, RustToCuda};

#[repr(transparent)]
#[cfg_attr(not(feature = "host"), derive(Debug))]
pub struct DeviceAccessible<T: ?Sized + DeviceCopy>(T);

unsafe impl<T: ?Sized + DeviceCopy> DeviceCopy for DeviceAccessible<T> {}

#[cfg(feature = "host")]
impl<T: DeviceCopy> From<T> for DeviceAccessible<T> {
    fn from(value: T) -> Self {
        Self(value)
    }
}

#[cfg(feature = "host")]
impl<T: StackOnly> From<&T> for DeviceAccessible<StackOnlyWrapper<T>> {
    fn from(value: &T) -> Self {
        let value = unsafe {
            let mut uninit = MaybeUninit::uninit();
            copy_nonoverlapping(value, uninit.as_mut_ptr(), 1);
            uninit.assume_init()
        };

        Self(StackOnlyWrapper::from(value))
    }
}

#[cfg(feature = "host")]
impl<T: ?Sized + DeviceCopy + fmt::Debug> fmt::Debug for DeviceAccessible<T> {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        fmt.debug_struct(stringify!(DeviceAccessible))
            .finish_non_exhaustive()
    }
}

#[cfg(not(feature = "host"))]
impl<T: ?Sized + DeviceCopy> Deref for DeviceAccessible<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[cfg(not(feature = "host"))]
impl<T: ?Sized + DeviceCopy> DerefMut for DeviceAccessible<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

pub trait RustToCudaProxy<T>: r#impl::RustToCudaImpl {
    fn from_ref(val: &T) -> &Self;
    fn from_mut(val: &mut T) -> &mut Self;

    fn into(self) -> T;
}

#[repr(transparent)]
#[derive(Clone, Copy)]
pub struct DevicePointerConst<T: Sized + DeviceCopy>(pub(super) *const T);

unsafe impl<T: Sized + DeviceCopy> DeviceCopy for DevicePointerConst<T> {}

#[cfg(feature = "host")]
#[doc(cfg(feature = "host"))]
impl<T: Sized + DeviceCopy> DevicePointerConst<T> {
    #[must_use]
    pub fn from(device_pointer: &rustacuda::memory::DevicePointer<T>) -> Self {
        Self(device_pointer.as_raw())
    }
}

#[cfg(any(not(feature = "host"), doc))]
#[doc(cfg(not(feature = "host")))]
impl<T: Sized + DeviceCopy> AsRef<T> for DevicePointerConst<T> {
    fn as_ref(&self) -> &T {
        unsafe { &*self.0 }
    }
}

#[repr(transparent)]
pub struct DevicePointerMut<T: Sized + DeviceCopy>(pub(super) *mut T);

unsafe impl<T: Sized + DeviceCopy> DeviceCopy for DevicePointerMut<T> {}

#[cfg(feature = "host")]
#[doc(cfg(feature = "host"))]
impl<T: Sized + DeviceCopy> DevicePointerMut<T> {
    #[must_use]
    pub fn from(device_pointer: &mut rustacuda::memory::DevicePointer<T>) -> Self {
        Self(device_pointer.as_raw_mut())
    }
}

#[cfg(any(not(feature = "host"), doc))]
#[doc(cfg(not(feature = "host")))]
impl<T: Sized + DeviceCopy> AsRef<T> for DevicePointerMut<T> {
    fn as_ref(&self) -> &T {
        unsafe { &*self.0 }
    }
}

#[cfg(any(not(feature = "host"), doc))]
#[doc(cfg(not(feature = "host")))]
impl<T: Sized + DeviceCopy> AsMut<T> for DevicePointerMut<T> {
    fn as_mut(&mut self) -> &mut T {
        unsafe { &mut *self.0 }
    }
}
