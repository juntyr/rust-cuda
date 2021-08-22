use core::ops::{Deref, DerefMut};
#[cfg(any(not(feature = "host"), doc))]
use core::{mem::MaybeUninit, ptr::copy_nonoverlapping};

use rustacuda_core::DeviceCopy;

use crate::common::{CudaAsRust, DeviceAccessible, RustToCuda};

/// ```rust
/// # use rust_cuda::utils::stack::StackOnly;
/// fn assert_stackonly(_x: impl StackOnly) {}
/// ```
/// ```rust
/// # use rust_cuda::utils::stack::StackOnly;
/// # fn assert_stackonly(_x: impl StackOnly) {}
/// assert_stackonly(42);
/// ```
/// ```rust
/// # use rust_cuda::utils::stack::StackOnly;
/// # fn assert_stackonly(_x: impl StackOnly) {}
/// assert_stackonly([42; 42]);
/// ```
/// ```rust,compile_fail
/// # use alloc::vec;
/// # use rust_cuda::utils::stack::StackOnly;
/// # fn assert_stackonly(_x: impl StackOnly) {}
/// assert_stackonly(vec![42]);
/// ```
/// ```rust,compile_fail
/// # use alloc::vec;
/// # use rust_cuda::utils::stack::StackOnly;
/// # fn assert_stackonly(_x: impl StackOnly) {}
/// assert_stackonly(&42);
/// ```
#[allow(clippy::module_name_repetitions)]
pub trait StackOnly: private::StackOnly {}
impl<T: private::StackOnly> StackOnly for T {}

#[repr(transparent)]
#[derive(Debug)]
#[allow(clippy::module_name_repetitions)]
pub struct StackOnlyDeviceCopy<T: StackOnly>(T);

// Safety: Any type that is fully on the stack without any references
//         to the heap can be safely copied to the GPU
unsafe impl<T: StackOnly> DeviceCopy for StackOnlyDeviceCopy<T> {}

impl<T: StackOnly> From<T> for StackOnlyDeviceCopy<T> {
    fn from(inner: T) -> Self {
        Self(inner)
    }
}

impl<T: StackOnly> StackOnlyDeviceCopy<T> {
    pub fn into_inner(self) -> T {
        self.0
    }
}

impl<T: StackOnly> Deref for StackOnlyDeviceCopy<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T: StackOnly> DerefMut for StackOnlyDeviceCopy<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

#[repr(transparent)]
#[allow(clippy::module_name_repetitions)]
pub struct StackOnlyRustToCuda<T: StackOnly>(T);

unsafe impl<T: StackOnly> DeviceCopy for StackOnlyRustToCuda<T> {}

impl<T: StackOnly> From<T> for StackOnlyRustToCuda<T> {
    fn from(inner: T) -> Self {
        Self(inner)
    }
}

impl<T: StackOnly> StackOnlyRustToCuda<T> {
    pub fn into_inner(self) -> T {
        self.0
    }
}

impl<T: StackOnly> Deref for StackOnlyRustToCuda<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T: StackOnly> DerefMut for StackOnlyRustToCuda<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

/*unsafe impl<T: StackOnly> RustToCuda for StackOnlyRustToCuda<T> {
    #[cfg(feature = "host")]
    #[doc(cfg(feature = "host"))]
    type CudaAllocation = crate::host::NullCudaAlloc;
    type CudaRepresentation = StackOnlyDeviceCopy<T>;

    #[cfg(feature = "host")]
    #[doc(cfg(feature = "host"))]
    #[allow(clippy::type_complexity)]
    unsafe fn borrow<A: crate::host::CudaAlloc>(
        &self,
        alloc: A,
    ) -> rustacuda::error::CudaResult<(
        DeviceAccessible<Self::CudaRepresentation>,
        crate::host::CombinedCudaAlloc<Self::CudaAllocation, A>,
    )> {
        let alloc = crate::host::CombinedCudaAlloc::new(crate::host::NullCudaAlloc, alloc);

        Ok((DeviceAccessible::from(&self.0), alloc))
    }

    #[cfg(feature = "host")]
    #[doc(cfg(feature = "host"))]
    unsafe fn restore<A: crate::host::CudaAlloc>(
        &mut self,
        alloc: crate::host::CombinedCudaAlloc<Self::CudaAllocation, A>,
    ) -> rustacuda::error::CudaResult<A> {
        let (_alloc_front, alloc_tail) = alloc.split();

        Ok(alloc_tail)
    }
}

unsafe impl<T: StackOnly> CudaAsRust for StackOnlyDeviceCopy<T> {
    type RustRepresentation = StackOnlyRustToCuda<T>;

    #[cfg(any(not(feature = "host"), doc))]
    #[doc(cfg(not(feature = "host")))]
    unsafe fn as_rust(this: &DeviceAccessible<Self>) -> Self::RustRepresentation {
        let mut uninit = MaybeUninit::uninit();
        copy_nonoverlapping(&***this, uninit.as_mut_ptr(), 1);
        StackOnlyRustToCuda(uninit.assume_init())
    }
}*/

mod private {
    pub auto trait StackOnly {}

    impl<T> !StackOnly for *const T {}
    impl<T> !StackOnly for *mut T {}
    impl<T> !StackOnly for &T {}
    impl<T> !StackOnly for &mut T {}

    //impl<T> !StackOnly for alloc::boxed::Box<T> {}

    impl<T> StackOnly for core::marker::PhantomData<T> {}
}

#[marker]
pub trait DeviceCopy2 {}

impl<T: private::StackOnly> DeviceCopy2 for T {}
impl<T: DeviceCopy> DeviceCopy2 for T {}
