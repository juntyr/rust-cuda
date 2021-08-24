use core::ops::{Deref, DerefMut};

use rustacuda_core::DeviceCopy;

use crate::common::{
    r#impl::{CudaAsRustImpl, RustToCudaImpl},
    DeviceAccessible,
};

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
pub struct StackOnlyWrapper<T: StackOnly>(T);

// Safety: Any type that is fully on the stack without any references
//         to the heap can be safely copied to the GPU
unsafe impl<T: StackOnly> DeviceCopy for StackOnlyWrapper<T> {}

impl<T: StackOnly> From<T> for StackOnlyWrapper<T> {
    fn from(inner: T) -> Self {
        Self(inner)
    }
}

impl<T: StackOnly> StackOnlyWrapper<T> {
    pub fn into_inner(self) -> T {
        self.0
    }
}

impl<T: StackOnly> Deref for StackOnlyWrapper<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T: StackOnly> DerefMut for StackOnlyWrapper<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

unsafe impl<T: StackOnly> RustToCudaImpl for StackOnlyWrapper<T> {
    #[cfg(feature = "host")]
    type CudaAllocationImpl = crate::host::NullCudaAlloc;
    type CudaRepresentationImpl = Self;

    #[cfg(feature = "host")]
    #[allow(clippy::type_complexity)]
    unsafe fn borrow_impl<A: crate::host::CudaAlloc>(
        &self,
        alloc: A,
    ) -> rustacuda::error::CudaResult<(
        DeviceAccessible<Self::CudaRepresentationImpl>,
        crate::host::CombinedCudaAlloc<Self::CudaAllocationImpl, A>,
    )> {
        let alloc = crate::host::CombinedCudaAlloc::new(crate::host::NullCudaAlloc, alloc);
        Ok((DeviceAccessible::from(&self.0), alloc))
    }

    #[cfg(feature = "host")]
    #[doc(cfg(feature = "host"))]
    unsafe fn restore_impl<A: crate::host::CudaAlloc>(
        &mut self,
        alloc: crate::host::CombinedCudaAlloc<Self::CudaAllocationImpl, A>,
    ) -> rustacuda::error::CudaResult<A> {
        let (_alloc_front, alloc_tail): (crate::host::NullCudaAlloc, A) = alloc.split();

        Ok(alloc_tail)
    }
}
unsafe impl<T: StackOnly> CudaAsRustImpl for StackOnlyWrapper<T> {
    type RustRepresentationImpl = Self;

    #[cfg(any(not(feature = "host"), doc))]
    unsafe fn as_rust_impl(this: &DeviceAccessible<Self>) -> Self::RustRepresentationImpl {
        let mut uninit = core::mem::MaybeUninit::uninit();
        core::ptr::copy_nonoverlapping(&**this, uninit.as_mut_ptr(), 1);
        uninit.assume_init()
    }
}

mod private {
    pub auto trait StackOnly {}

    impl<T> !StackOnly for *const T {}
    impl<T> !StackOnly for *mut T {}
    impl<T> !StackOnly for &T {}
    impl<T> !StackOnly for &mut T {}

    impl<T> StackOnly for core::marker::PhantomData<T> {}
}