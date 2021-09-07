use core::ops::{Deref, DerefMut};

use rustacuda_core::DeviceCopy;

use crate::common::{CudaAsRust, DeviceAccessible, RustToCuda};

/// ```rust
/// # use rust_cuda::utils::alloc::unified::StackOrUnified;
/// fn assert_stack_or_unified(_x: impl StackOrUnified) {}
/// ```
/// ```rust
/// # use rust_cuda::utils::alloc::unified::StackOrUnified;
/// # fn assert_stack_or_unified(_x: impl StackOrUnified) {}
/// assert_stack_or_unified(42);
/// ```
/// ```rust
/// # use rust_cuda::utils::alloc::unified::StackOrUnified;
/// # fn assert_stack_or_unified(_x: impl StackOrUnified) {}
/// assert_stack_or_unified([42; 42]);
/// ```
/// ```rust,compile_fail
/// # use alloc::vec;
/// # use rust_cuda::utils::alloc::unified::StackOrUnified;
/// # fn assert_stack_or_unified(_x: impl StackOrUnified) {}
/// assert_stack_or_unified(vec![42]);
/// ```
/// ```rust,compile_fail
/// # use rust_cuda::utils::alloc::unified::StackOrUnified;
/// # fn assert_stack_or_unified(_x: impl StackOrUnified) {}
/// assert_stack_or_unified(&42);
/// ```
#[allow(clippy::module_name_repetitions)]
pub trait StackOrUnified: sealed::StackOrUnified {}
impl<T: sealed::StackOrUnified> StackOrUnified for T {}

#[repr(transparent)]
#[derive(Debug)]
#[allow(clippy::module_name_repetitions)]
pub struct StackOrUnifiedWrapper<T: StackOrUnified>(T);

// Safety: Any type that is fully on the stack or that contains just
//          references to the unified heap can be safely copied to the GPU
unsafe impl<T: StackOrUnified> DeviceCopy for StackOrUnifiedWrapper<T> {}

impl<T: StackOrUnified> From<T> for StackOrUnifiedWrapper<T> {
    fn from(inner: T) -> Self {
        Self(inner)
    }
}

impl<T: StackOrUnified> StackOrUnifiedWrapper<T> {
    pub fn into_inner(self) -> T {
        self.0
    }
}

impl<T: StackOrUnified> Deref for StackOrUnifiedWrapper<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T: StackOrUnified> DerefMut for StackOrUnifiedWrapper<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

unsafe impl<T: StackOrUnified> RustToCuda for StackOrUnifiedWrapper<T> {
    #[cfg(feature = "host")]
    type CudaAllocation = crate::host::NullCudaAlloc;
    type CudaRepresentation = Self;

    #[cfg(feature = "host")]
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
        let (_alloc_front, alloc_tail): (crate::host::NullCudaAlloc, A) = alloc.split();

        Ok(alloc_tail)
    }
}

unsafe impl<T: StackOrUnified> CudaAsRust for StackOrUnifiedWrapper<T> {
    type RustRepresentation = Self;

    #[cfg(any(not(feature = "host"), doc))]
    unsafe fn as_rust(this: &DeviceAccessible<Self>) -> Self::RustRepresentation {
        let mut uninit = core::mem::MaybeUninit::uninit();
        core::ptr::copy_nonoverlapping(&**this, uninit.as_mut_ptr(), 1);
        uninit.assume_init()
    }
}

mod sealed {
    use crate::utils::alloc::allocator::UnifiedAllocator;

    pub auto trait StackOrUnified {}

    impl<T> !StackOrUnified for *const T {}
    impl<T> !StackOrUnified for *mut T {}
    impl<T> !StackOrUnified for &T {}
    impl<T> !StackOrUnified for &mut T {}

    impl<T> StackOrUnified for core::marker::PhantomData<T> {}

    impl<T> StackOrUnified for alloc::boxed::Box<T, UnifiedAllocator> {}
    impl<T> StackOrUnified for alloc::vec::Vec<T, UnifiedAllocator> {}
    impl<T> StackOrUnified for hashbrown::HashMap<T, UnifiedAllocator> {}
}
