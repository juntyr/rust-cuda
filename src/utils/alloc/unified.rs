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
