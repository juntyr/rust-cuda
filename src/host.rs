use core::ops::{Deref, DerefMut};

use rustacuda::{
    context::Context,
    error::{CudaError, CudaResult},
    memory::{DeviceBox, DeviceBuffer, LockedBuffer},
    module::Module,
    stream::Stream,
};
use rustacuda_core::{DeviceCopy, DevicePointer};

use crate::common::RustToCuda;

/// # Safety
/// This trait should ONLY be derived automatically using
/// `#[derive(LendToCuda)]`
pub unsafe trait LendToCuda: RustToCuda {
    /// # Errors
    /// Returns a `rustacuda::errors::CudaError` iff an error occurs inside CUDA
    fn lend_to_cuda<
        O,
        F: FnOnce(DevicePointer<<Self as RustToCuda>::CudaRepresentation>) -> CudaResult<O>,
    >(
        &self,
        inner: F,
    ) -> CudaResult<O>;

    /// # Errors
    /// Returns a `rustacuda::errors::CudaError` iff an error occurs inside CUDA
    fn lend_to_cuda_mut<
        O,
        F: FnOnce(DevicePointer<<Self as RustToCuda>::CudaRepresentation>) -> CudaResult<O>,
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
}

pub trait CudaAlloc: private::alloc::Sealed {}
impl<T: private::alloc::Sealed> CudaAlloc for T {}

pub struct NullCudaAlloc;
impl private::alloc::Sealed for NullCudaAlloc {}

pub struct CombinedCudaAlloc<A: CudaAlloc, B: CudaAlloc>(A, B);
impl<A: CudaAlloc, B: CudaAlloc> private::alloc::Sealed for CombinedCudaAlloc<A, B> {}
impl<A: CudaAlloc, B: CudaAlloc> CombinedCudaAlloc<A, B> {
    pub fn new(front: A, tail: B) -> Self {
        Self(front, tail)
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
                core::mem::forget(val)
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
