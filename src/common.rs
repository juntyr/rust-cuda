#[cfg(any(not(feature = "host"), doc))]
use core::convert::{AsMut, AsRef};
use core::mem::size_of;

#[cfg(feature = "host")]
use alloc::fmt;
#[cfg(all(not(feature = "host")))]
use core::ops::{Deref, DerefMut};
#[cfg(feature = "host")]
use core::{mem::MaybeUninit, ptr::copy_nonoverlapping};

use rustacuda_core::DeviceCopy;

#[cfg(feature = "derive")]
#[doc(cfg(feature = "derive"))]
pub use rust_cuda_derive::{LendRustBorrowToCuda, RustToCudaAsRust};

#[cfg(feature = "derive")]
#[doc(cfg(feature = "derive"))]
pub use rust_cuda_derive::kernel;

use crate::host::{CombinedCudaAlloc, EmptyCudaAlloc, NullCudaAlloc};
#[cfg(feature = "host")]
use crate::utils::stack::{StackOnly, StackOnlyDeviceCopy};

#[repr(transparent)]
#[cfg_attr(not(feature = "host"), derive(Debug))]
pub struct DeviceAccessible<T: DeviceCopy>(T);

unsafe impl<T: DeviceCopy> DeviceCopy for DeviceAccessible<T> {}

#[cfg(feature = "host")]
impl<T: DeviceCopy> From<T> for DeviceAccessible<T> {
    fn from(value: T) -> Self {
        Self(value)
    }
}

#[cfg(feature = "host")]
impl<T: StackOnly> From<&T> for DeviceAccessible<StackOnlyDeviceCopy<T>> {
    fn from(value: &T) -> Self {
        let value = unsafe {
            let mut uninit = MaybeUninit::uninit();
            copy_nonoverlapping(value, uninit.as_mut_ptr(), 1);
            uninit.assume_init()
        };

        Self(StackOnlyDeviceCopy::from(value))
    }
}

#[cfg(feature = "host")]
impl<T: DeviceCopy + fmt::Debug> fmt::Debug for DeviceAccessible<T> {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        fmt.debug_struct(stringify!(DeviceAccessible))
            .finish_non_exhaustive()
    }
}

#[cfg(not(feature = "host"))]
impl<T: DeviceCopy> Deref for DeviceAccessible<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[cfg(not(feature = "host"))]
impl<T: DeviceCopy> DerefMut for DeviceAccessible<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

#[marker]
pub trait StackOnlyBottom {}
impl<T: StackOnly> StackOnlyBottom for T {}
pub trait RustToCudaTop: StackOnlyBottom {}
impl<T: RustToCuda> StackOnlyBottom for T {}
impl<T: RustToCuda> RustToCudaTop for T {}

pub trait RustToCudaWrapperCore {
    type CudaRepresentation: DeviceCopy;//: CudaAsRustCore<RustRepresentation = Self>;
    type CudaAllocation: crate::host::CudaAlloc;
}

pub trait RustToCudaWrapper: RustToCudaWrapperCore {
    //type CudaRepresentation: DeviceCopy;//: CudaAsRustCore<RustRepresentation = Self>;
    //type CudaAllocation: crate::host::CudaAlloc;

    unsafe fn borrow<A: crate::host::CudaAlloc>(
        &self,
        alloc: A,
    ) -> rustacuda::error::CudaResult<(
        DeviceAccessible<Self::CudaRepresentation>,
        crate::host::CombinedCudaAlloc<Self::CudaAllocation, A>,
    )>;

    unsafe fn restore<A: crate::host::CudaAlloc>(
        &mut self,
        alloc: crate::host::CombinedCudaAlloc<Self::CudaAllocation, A>,
    ) -> rustacuda::error::CudaResult<A>;
}

#[repr(transparent)]
pub struct UnsafeDeviceCopy<T>(T);

unsafe impl<T> DeviceCopy for UnsafeDeviceCopy<T> {}

impl<T: StackOnlyBottom> RustToCudaWrapperCore for T {
    default type CudaRepresentation = UnsafeDeviceCopy<T>;
    default type CudaAllocation = NullCudaAlloc;
}

/// ```rust
/// let x = Box::new(true);
/// unsafe {
///    rust_cuda::common::RustToCudaWrapper::borrow(&x, rust_cuda::host::NullCudaAlloc);
/// }
/// ```
impl<T: StackOnlyBottom> RustToCudaWrapper for T {
    //default type CudaRepresentation = UnsafeDeviceCopy<T>;
    //default type CudaAllocation = NullCudaAlloc;

    default unsafe fn borrow<A: crate::host::CudaAlloc>(&self, alloc: A) -> rustacuda::error::CudaResult<(
        DeviceAccessible<Self::CudaRepresentation>,
        crate::host::CombinedCudaAlloc<Self::CudaAllocation, A>,
    )> {
        trait IsNullCudaAlloc {
            const IMPLS: bool = false;
        }
        impl<T: ?Sized> IsNullCudaAlloc for T {}
        struct Wrapper<T: ?Sized>(core::marker::PhantomData<T>);
        #[allow(dead_code)]
        impl<T: RustToCudaWrapperCore<
            CudaRepresentation = UnsafeDeviceCopy<T>,
            CudaAllocation = NullCudaAlloc,
        >> Wrapper<T> {
            const IMPLS: bool = true;
        }

        if !<Wrapper<Self>>::IMPLS {
            extern "C" {
                fn linker_error();
            }

            linker_error();
        }

        let cuda_repr = {
            let mut uninit = MaybeUninit::uninit();
            copy_nonoverlapping(self as *const Self as *const _, uninit.as_mut_ptr(), 1);
            uninit.assume_init()
        };

        let null_alloc = MaybeUninit::uninit().assume_init();

        Ok((cuda_repr, CombinedCudaAlloc::new(null_alloc, alloc)))
    }

    default unsafe fn restore<A: crate::host::CudaAlloc>(&mut self, alloc: crate::host::CombinedCudaAlloc<Self::CudaAllocation, A>) -> rustacuda::error::CudaResult<A> {
        Ok(alloc.split().1)
    }
}

impl<T: RustToCudaTop + RustToCuda> RustToCudaWrapperCore for T {
    type CudaRepresentation = <T as RustToCudaCore>::CudaRepresentation;
    type CudaAllocation = <T as RustToCudaAlloc>::CudaAllocation;
}

impl<T: RustToCudaTop + RustToCuda> RustToCudaWrapper for T {
    //type CudaRepresentation = <T as RustToCudaCore>::CudaRepresentation;
    //type CudaAllocation = <T as RustToCudaAlloc>::CudaAllocation;

    unsafe fn borrow<A: crate::host::CudaAlloc>(&self, alloc: A) -> rustacuda::error::CudaResult<(
        DeviceAccessible<Self::CudaRepresentation>,
        crate::host::CombinedCudaAlloc<Self::CudaAllocation, A>,
    )> {
        RustToCuda::borrow(self, alloc)
    }

    unsafe fn restore<A: crate::host::CudaAlloc>(
        &mut self,
        alloc: crate::host::CombinedCudaAlloc<Self::CudaAllocation, A>,
    ) -> rustacuda::error::CudaResult<A> {
        RustToCuda::restore(self, alloc)
    }
}

/*impl<T: crate::utils::stack::StackOnly> RustToCudaCore for T {
    default type CudaRepresentation = crate::utils::stack::StackOnlyDeviceCopy<T>;
}

impl<T: crate::utils::stack::StackOnly> CudaAsRustCore for crate::utils::stack::StackOnlyDeviceCopy<T> {
    default type RustRepresentation = T;
}*/

/*unsafe impl<T: /*crate::utils::stack::StackOnly
                + */RustToCudaCore<CudaRepresentation = crate::utils::stack::StackOnlyDeviceCopy<T>>
                + RustToCudaAlloc<CudaAllocation = crate::host::NullCudaAlloc>
> RustToCuda for T {
    #[cfg(feature = "host")]
    #[doc(cfg(feature = "host"))]
    #[allow(clippy::type_complexity)]
    default unsafe fn borrow<A: crate::host::CudaAlloc>(
        &self,
        alloc: A,
    ) -> rustacuda::error::CudaResult<(
        DeviceAccessible<Self::CudaRepresentation>,
        crate::host::CombinedCudaAlloc<Self::CudaAllocation, A>,
    )> {
        let alloc = crate::host::CombinedCudaAlloc::new(crate::host::NullCudaAlloc, alloc);

        Ok((DeviceAccessible::from(self), alloc))
    }

    #[cfg(feature = "host")]
    #[doc(cfg(feature = "host"))]
    default unsafe fn restore<A: crate::host::CudaAlloc>(
        &mut self,
        alloc: crate::host::CombinedCudaAlloc<Self::CudaAllocation, A>,
    ) -> rustacuda::error::CudaResult<A> {
        let (_alloc_front, alloc_tail) = alloc.split();

        Ok(alloc_tail)
    }
}*/

/*unsafe impl<T: crate::utils::stack::StackOnly + RustToCuda<CudaRepresentation = crate::utils::stack::StackOnlyDeviceCopy<T>>> CudaAsRust for crate::utils::stack::StackOnlyDeviceCopy<T> {
    type RustRepresentation = T;

    #[cfg(any(not(feature = "host"), doc))]
    #[doc(cfg(not(feature = "host")))]
    unsafe fn as_rust(this: &DeviceAccessible<Self>) -> Self::RustRepresentation {
        let mut uninit = core::mem::MaybeUninit::uninit();
        core::ptr::copy_nonoverlapping(&***this, uninit.as_mut_ptr(), 1);
        uninit.assume_init()
    }
}*/

pub trait RustToCudaCore: Sized {
    type CudaRepresentation: CudaAsRustCore<RustRepresentation = Self>;
}

#[cfg(feature = "host")]
#[doc(cfg(feature = "host"))]
pub trait RustToCudaAlloc: Sized {
    type CudaAllocation: crate::host::CudaAlloc;
}

pub trait CudaAsRustCore: Sized + DeviceCopy {
    type RustRepresentation: RustToCudaCore<CudaRepresentation = Self>;
}

/// # Safety
/// This is an internal trait and should ONLY be derived automatically using
/// `#[derive(RustToCuda)]`
pub unsafe trait RustToCuda: RustToCudaCore + RustToCudaAlloc {
    //#[cfg(feature = "host")]
    //#[doc(cfg(feature = "host"))]
    //type CudaAllocation: crate::host::CudaAlloc;

    #[cfg(feature = "host")]
    #[doc(cfg(feature = "host"))]
    /// # Errors
    ///
    /// Returns a `rustacuda::errors::CudaError` iff an error occurs inside CUDA
    ///
    /// # Safety
    ///
    /// This is an internal function and should NEVER be called manually
    /// The returned `Self::CudaRepresentation` must NEVER be accessed on the
    ///  CPU  as it contains a GPU-resident copy of `self`.
    #[allow(clippy::type_complexity)]
    unsafe fn borrow<A: crate::host::CudaAlloc>(
        &self,
        alloc: A,
    ) -> rustacuda::error::CudaResult<(
        DeviceAccessible<Self::CudaRepresentation>,
        crate::host::CombinedCudaAlloc<Self::CudaAllocation, A>,
    )>;

    #[cfg(feature = "host")]
    #[doc(cfg(feature = "host"))]
    /// # Errors
    ///
    /// Returns a `rustacuda::errors::CudaError` iff an error occurs inside CUDA
    ///
    /// # Safety
    ///
    /// This is an internal function and should NEVER be called manually
    #[allow(clippy::type_complexity)]
    unsafe fn restore<A: crate::host::CudaAlloc>(
        &mut self,
        alloc: crate::host::CombinedCudaAlloc<Self::CudaAllocation, A>,
    ) -> rustacuda::error::CudaResult<A>;
}

/// # Safety
///
/// This is an internal trait and should NEVER be implemented manually
pub unsafe trait CudaAsRust: CudaAsRustCore {
    #[cfg(any(not(feature = "host"), doc))]
    #[doc(cfg(not(feature = "host")))]
    /// # Safety
    ///
    /// This is an internal function and should NEVER be called manually
    unsafe fn as_rust(this: &DeviceAccessible<Self>) -> Self::RustRepresentation;
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
