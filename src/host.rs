use core::ops::{Deref, DerefMut};

use rustacuda::{
    context::Context,
    error::{CudaError, CudaResult},
    memory::{DeviceBox, DeviceBuffer, LockedBuffer},
    module::Module,
    stream::Stream,
};
use rustacuda_core::DeviceCopy;

#[cfg(feature = "derive")]
pub use rust_cuda_derive::LendToCuda;

#[cfg(feature = "derive")]
pub use rust_cuda_derive::{link_kernel, specialise_kernel_call};

use crate::common::{DeviceBoxConst, DeviceBoxMut, RustToCuda};

/// # Safety
/// This trait should ONLY be derived automatically using
/// `#[derive(LendToCuda)]`
pub unsafe trait LendToCuda: RustToCuda {
    /// Lends an immutable copy of `&self` to CUDA:
    /// - code in the CUDA kernel can only access `&self` through the
    ///   `DeviceBoxConst` inside the closure
    /// - after the closure, `&self` will not have changed
    ///
    /// # Errors
    /// Returns a `rustacuda::errors::CudaError` iff an error occurs inside CUDA
    fn lend_to_cuda<
        O,
        F: FnOnce(HostDeviceBoxConst<<Self as RustToCuda>::CudaRepresentation>) -> CudaResult<O>,
    >(
        &self,
        inner: F,
    ) -> CudaResult<O>;

    /// Lends a mutable copy of `&mut self` to CUDA:
    /// - code in the CUDA kernel can only access `&mut self` through the
    ///   `DeviceBoxMut` inside the closure
    /// - after the closure, `&mut self` might have changed in the following
    ///   ways:
    ///   - to avoid aliasing, each CUDA thread gets its own shallow copy of
    ///     `&mut self`, i.e. any shallow changes will NOT be reflected after
    ///     the closure
    ///   - each CUDA thread can access the same heap allocated storage, i.e.
    ///     any deep changes will be reflected after the closure
    ///
    /// # Errors
    /// Returns a `rustacuda::errors::CudaError` iff an error occurs inside CUDA
    fn lend_to_cuda_mut<
        O,
        F: FnOnce(HostDeviceBoxMut<<Self as RustToCuda>::CudaRepresentation>) -> CudaResult<O>,
    >(
        &mut self,
        inner: F,
    ) -> CudaResult<O>;
}

pub(crate) mod private {
    pub mod alloc {
        pub trait Sealed {}
    }

    pub mod drop {
        pub trait Sealed: Sized {
            fn drop(val: Self) -> Result<(), (rustacuda::error::CudaError, Self)>;
        }
    }

    pub mod empty {
        pub trait Sealed {}
    }
}

pub trait EmptyCudaAlloc: private::empty::Sealed {}
impl<T: private::empty::Sealed> EmptyCudaAlloc for T {}

pub trait CudaAlloc: private::alloc::Sealed {}
impl<T: private::alloc::Sealed> CudaAlloc for T {}

pub struct NullCudaAlloc;
impl private::alloc::Sealed for NullCudaAlloc {}
impl private::empty::Sealed for NullCudaAlloc {}

pub struct CombinedCudaAlloc<A: CudaAlloc, B: CudaAlloc>(A, B);
impl<A: CudaAlloc, B: CudaAlloc> private::alloc::Sealed for CombinedCudaAlloc<A, B> {}
impl<A: CudaAlloc + EmptyCudaAlloc, B: CudaAlloc + EmptyCudaAlloc> private::empty::Sealed
    for CombinedCudaAlloc<A, B>
{
}
impl<A: CudaAlloc, B: CudaAlloc> CombinedCudaAlloc<A, B> {
    pub fn new(front: A, tail: B) -> Self {
        Self(front, tail)
    }

    pub fn split(self) -> (A, B) {
        (self.0, self.1)
    }
}

pub struct CudaDropWrapper<C: private::drop::Sealed>(Option<C>);
impl<C: private::drop::Sealed> private::alloc::Sealed for CudaDropWrapper<C> {}
impl<C: private::drop::Sealed> From<C> for CudaDropWrapper<C> {
    fn from(val: C) -> Self {
        Self(Some(val))
    }
}
impl<C: private::drop::Sealed> Drop for CudaDropWrapper<C> {
    fn drop(&mut self) {
        if let Some(val) = self.0.take() {
            if let Err((_err, val)) = C::drop(val) {
                core::mem::forget(val);
            }
        }
    }
}
impl<C: private::drop::Sealed> Deref for CudaDropWrapper<C> {
    type Target = C;

    fn deref(&self) -> &Self::Target {
        self.0.as_ref().unwrap()
    }
}
impl<C: private::drop::Sealed> DerefMut for CudaDropWrapper<C> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.0.as_mut().unwrap()
    }
}

macro_rules! impl_sealed_drop_collection {
    ($type:ident) => {
        impl<C: DeviceCopy> private::drop::Sealed for $type<C> {
            fn drop(val: Self) -> Result<(), (CudaError, Self)> {
                Self::drop(val)
            }
        }
    };
}

impl_sealed_drop_collection!(DeviceBuffer);
impl_sealed_drop_collection!(DeviceBox);
impl_sealed_drop_collection!(LockedBuffer);

macro_rules! impl_sealed_drop_value {
    ($type:ident) => {
        impl private::drop::Sealed for $type {
            fn drop(val: Self) -> Result<(), (CudaError, Self)> {
                Self::drop(val)
            }
        }
    };
}

impl_sealed_drop_value!(Module);
impl_sealed_drop_value!(Stream);
impl_sealed_drop_value!(Context);

#[allow(clippy::module_name_repetitions)]
pub struct HostDeviceBoxMut<'a, T: Sized + DeviceCopy> {
    device_box: &'a mut DeviceBox<T>,
    host_ref: &'a mut T,
}

impl<'a, T: Sized + DeviceCopy> HostDeviceBoxMut<'a, T> {
    pub fn new(device_box: &'a mut DeviceBox<T>, host_ref: &'a mut T) -> Self {
        Self {
            device_box,
            host_ref,
        }
    }

    #[must_use]
    pub fn for_device(&mut self) -> DeviceBoxMut<T> {
        DeviceBoxMut::from(&mut self.device_box)
    }

    #[must_use]
    pub fn for_host(&mut self) -> &T {
        self.host_ref
    }
}

#[allow(clippy::module_name_repetitions)]
pub struct HostDeviceBoxConst<'a, T: Sized + DeviceCopy> {
    device_box: &'a DeviceBox<T>,
    host_ref: &'a T,
}

impl<'a, T: Sized + DeviceCopy> HostDeviceBoxConst<'a, T> {
    pub fn new(device_box: &'a DeviceBox<T>, host_ref: &'a T) -> Self {
        Self {
            device_box,
            host_ref,
        }
    }

    #[must_use]
    pub fn for_device(&self) -> DeviceBoxConst<T> {
        DeviceBoxConst::from(self.device_box)
    }

    #[must_use]
    pub fn for_host(&self) -> &T {
        self.host_ref
    }
}
