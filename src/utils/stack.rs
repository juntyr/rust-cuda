use core::ops::{Deref, DerefMut};

use rustacuda_core::DeviceCopy;

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

mod private {
    pub auto trait StackOnly {}

    impl<T> !StackOnly for *const T {}
    impl<T> !StackOnly for *mut T {}
    impl<T> !StackOnly for &T {}
    impl<T> !StackOnly for &mut T {}

    impl<T> StackOnly for core::marker::PhantomData<T> {}
}
