#[cfg(any(not(feature = "host"), doc))]
use core::convert::{AsMut, AsRef};
use core::marker::PhantomData;

#[cfg(feature = "host")]
use alloc::fmt;
#[cfg(not(feature = "host"))]
use core::ops::{Deref, DerefMut};
#[cfg(feature = "host")]
use core::{mem::MaybeUninit, ptr::copy_nonoverlapping};

use const_type_layout::TypeGraphLayout;
use rustacuda_core::DeviceCopy;

#[cfg(feature = "derive")]
#[doc(cfg(feature = "derive"))]
pub use rust_cuda_derive::LendRustToCuda;

#[cfg(feature = "derive")]
#[doc(cfg(feature = "derive"))]
pub use rust_cuda_derive::kernel;

#[cfg(feature = "host")]
use crate::{safety::SafeDeviceCopy, utils::device_copy::SafeDeviceCopyWrapper};

#[repr(transparent)]
#[cfg_attr(not(feature = "host"), derive(Debug))]
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

/// # Safety
///
/// This is an internal trait and should ONLY be derived automatically using
/// `#[derive(LendRustToCuda)]`
pub unsafe trait RustToCuda {
    type CudaAllocation: CudaAlloc;
    type CudaRepresentation: CudaAsRust<RustRepresentation = Self> + TypeGraphLayout;

    #[cfg(feature = "host")]
    #[doc(cfg(feature = "host"))]
    /// # Errors
    ///
    /// Returns a [`rustacuda::error::CudaError`] iff an error occurs inside
    /// CUDA
    ///
    /// # Safety
    ///
    /// This is an internal function and should NEVER be called manually
    /// The returned [`Self::CudaRepresentation`] must NEVER be accessed on the
    ///  CPU  as it contains a GPU-resident copy of `self`.
    #[allow(clippy::type_complexity)]
    unsafe fn borrow<A: CudaAlloc>(
        &self,
        alloc: A,
    ) -> rustacuda::error::CudaResult<(
        DeviceAccessible<Self::CudaRepresentation>,
        CombinedCudaAlloc<Self::CudaAllocation, A>,
    )>;

    #[cfg(feature = "host")]
    #[doc(cfg(feature = "host"))]
    /// # Errors
    ///
    /// Returns a [`rustacuda::error::CudaError`] iff an error occurs inside
    /// CUDA
    ///
    /// # Safety
    ///
    /// This is an internal function and should NEVER be called manually
    #[allow(clippy::type_complexity)]
    unsafe fn restore<A: CudaAlloc>(
        &mut self,
        alloc: CombinedCudaAlloc<Self::CudaAllocation, A>,
    ) -> rustacuda::error::CudaResult<A>;
}

/// # Safety
///
/// This is an internal trait and should ONLY be derived automatically using
/// `#[derive(LendRustToCuda)]`
pub unsafe trait RustToCudaAsync: RustToCuda {
    #[cfg(feature = "host")]
    #[doc(cfg(feature = "host"))]
    /// # Errors
    ///
    /// Returns a [`rustacuda::error::CudaError`] iff an error occurs inside
    /// CUDA
    ///
    /// # Safety
    ///
    /// This is an internal function and should NEVER be called manually
    /// The returned
    /// [`Self::CudaRepresentation`](RustToCuda::CudaRepresentation) must NEVER
    /// be accessed on the  CPU  as it contains a GPU-resident copy of
    /// `self`.
    #[allow(clippy::type_complexity)]
    unsafe fn borrow_async<A: CudaAlloc>(
        &self,
        alloc: A,
        stream: &rustacuda::stream::Stream,
    ) -> rustacuda::error::CudaResult<(
        DeviceAccessible<Self::CudaRepresentation>,
        CombinedCudaAlloc<Self::CudaAllocation, A>,
    )>;

    #[cfg(feature = "host")]
    #[doc(cfg(feature = "host"))]
    /// # Errors
    ///
    /// Returns a [`rustacuda::error::CudaError`] iff an error occurs inside
    /// CUDA
    ///
    /// # Safety
    ///
    /// This is an internal function and should NEVER be called manually
    #[allow(clippy::type_complexity)]
    unsafe fn restore_async<A: CudaAlloc>(
        &mut self,
        alloc: CombinedCudaAlloc<Self::CudaAllocation, A>,
        stream: &rustacuda::stream::Stream,
    ) -> rustacuda::error::CudaResult<A>;
}

/// # Safety
///
/// This is an internal trait and should NEVER be implemented manually
pub unsafe trait CudaAsRust: DeviceCopy + TypeGraphLayout {
    type RustRepresentation: RustToCuda<CudaRepresentation = Self>;

    #[cfg(any(not(feature = "host"), doc))]
    #[doc(cfg(not(feature = "host")))]
    /// # Safety
    ///
    /// This is an internal function and should NEVER be called manually
    unsafe fn as_rust(this: &DeviceAccessible<Self>) -> Self::RustRepresentation;
}

pub trait RustToCudaProxy<T>: RustToCuda {
    fn from_ref(val: &T) -> &Self;
    fn from_mut(val: &mut T) -> &mut Self;

    fn into(self) -> T;
}

pub trait RustToCudaAsyncProxy<T>: RustToCudaAsync {
    fn from_ref(val: &T) -> &Self;
    fn from_mut(val: &mut T) -> &mut Self;

    fn into(self) -> T;
}

#[repr(transparent)]
#[derive(Clone, Copy, TypeLayout)]
pub struct DeviceConstRef<'r, T: DeviceCopy + 'r> {
    #[cfg_attr(feature = "host", allow(dead_code))]
    pub(super) pointer: *const T,
    pub(super) reference: PhantomData<&'r T>,
}

unsafe impl<'r, T: DeviceCopy> DeviceCopy for DeviceConstRef<'r, T> {}

#[cfg(any(not(feature = "host"), doc))]
#[doc(cfg(not(feature = "host")))]
impl<'r, T: DeviceCopy> AsRef<T> for DeviceConstRef<'r, T> {
    fn as_ref(&self) -> &T {
        unsafe { &*self.pointer }
    }
}

#[repr(transparent)]
#[derive(TypeLayout)]
pub struct DeviceMutRef<'r, T: DeviceCopy + 'r> {
    #[cfg_attr(feature = "host", allow(dead_code))]
    pub(super) pointer: *mut T,
    pub(super) reference: PhantomData<&'r mut T>,
}

unsafe impl<'r, T: DeviceCopy> DeviceCopy for DeviceMutRef<'r, T> {}

#[cfg(any(not(feature = "host"), doc))]
#[doc(cfg(not(feature = "host")))]
impl<'r, T: DeviceCopy> AsRef<T> for DeviceMutRef<'r, T> {
    fn as_ref(&self) -> &T {
        unsafe { &*self.pointer }
    }
}

#[cfg(any(not(feature = "host"), doc))]
#[doc(cfg(not(feature = "host")))]
impl<'r, T: DeviceCopy> AsMut<T> for DeviceMutRef<'r, T> {
    fn as_mut(&mut self) -> &mut T {
        unsafe { &mut *self.pointer }
    }
}

pub(crate) mod crate_private {
    pub mod alloc {
        pub trait Sealed {}
    }
}

mod private {
    pub mod empty {
        pub trait Sealed {}
    }
}

pub trait EmptyCudaAlloc: private::empty::Sealed {}
impl<T: private::empty::Sealed> EmptyCudaAlloc for T {}

pub trait CudaAlloc: crate_private::alloc::Sealed {}
impl<T: crate_private::alloc::Sealed> CudaAlloc for T {}

impl<T: CudaAlloc> crate_private::alloc::Sealed for Option<T> {}

pub struct NoCudaAlloc;
impl crate_private::alloc::Sealed for NoCudaAlloc {}
impl private::empty::Sealed for NoCudaAlloc {}

pub struct SomeCudaAlloc(());
impl crate_private::alloc::Sealed for SomeCudaAlloc {}
impl !private::empty::Sealed for SomeCudaAlloc {}

pub struct CombinedCudaAlloc<A: CudaAlloc, B: CudaAlloc>(A, B);
impl<A: CudaAlloc, B: CudaAlloc> crate_private::alloc::Sealed for CombinedCudaAlloc<A, B> {}
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
