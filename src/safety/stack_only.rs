/// ```rust
/// # use rust_cuda::safety::StackOnly;
/// fn assert_stackonly(_x: impl StackOnly) {}
/// ```
/// ```rust
/// # use rust_cuda::safety::StackOnly;
/// # fn assert_stackonly(_x: impl StackOnly) {}
/// assert_stackonly(42);
/// ```
/// ```rust
/// # use rust_cuda::safety::StackOnly;
/// # fn assert_stackonly(_x: impl StackOnly) {}
/// assert_stackonly([42; 42]);
/// ```
/// ```rust,compile_fail
/// # use alloc::vec;
/// # use rust_cuda::safety::StackOnly;
/// # fn assert_stackonly(_x: impl StackOnly) {}
/// assert_stackonly(vec![42]);
/// ```
/// ```rust,compile_fail
/// # use alloc::vec;
/// # use rust_cuda::safety::StackOnly;
/// # fn assert_stackonly(_x: impl StackOnly) {}
/// assert_stackonly(&42);
/// ```
#[allow(clippy::module_name_repetitions)]
pub trait StackOnly: sealed::StackOnly {}
impl<T: sealed::StackOnly> StackOnly for T {}

mod sealed {
    pub auto trait StackOnly {}

    impl<T> !StackOnly for *const T {}
    impl<T> !StackOnly for *mut T {}
    impl<T> !StackOnly for &T {}
    impl<T> !StackOnly for &mut T {}

    // Thread-block-shared data contains data not on the stack
    impl<T: 'static> !StackOnly for crate::utils::shared::r#static::ThreadBlockShared<T> {}
    impl<T: 'static + const_type_layout::TypeGraphLayout> !StackOnly
        for crate::utils::shared::slice::ThreadBlockSharedSlice<T>
    {
    }

    impl<T> StackOnly for core::marker::PhantomData<T> {}
}
