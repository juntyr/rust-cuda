/// ```rust
/// # use rust_cuda::utils::alloc::unified::StackOrUnifiedHeap;
/// fn assert_stack_or_unified_heap(_x: impl StackOrUnifiedHeap) {}
/// ```
/// ```rust
/// # use rust_cuda::utils::alloc::unified::StackOrUnifiedHeap;
/// # fn assert_stack_or_unified_heap(_x: impl StackOrUnifiedHeap) {}
/// assert_stack_or_unified_heap(42);
/// ```
/// ```rust
/// # use rust_cuda::utils::alloc::unified::StackOrUnifiedHeap;
/// # fn assert_stack_or_unified_heap(_x: impl StackOrUnifiedHeap) {}
/// assert_stack_or_unified_heap([42; 42]);
/// ```
/// ```rust,compile_fail
/// # use alloc::vec;
/// # use rust_cuda::utils::alloc::unified::StackOrUnifiedHeap;
/// # fn assert_stack_or_unified_heap(_x: impl StackOrUnifiedHeap) {}
/// assert_stack_or_unified_heap(vec![42]);
/// ```
/// ```rust,compile_fail
/// # use rust_cuda::utils::alloc::unified::StackOrUnifiedHeap;
/// # fn assert_stack_or_unified_heap(_x: impl StackOrUnifiedHeap) {}
/// assert_stack_or_unified_heap(&42);
/// ```
#[allow(clippy::module_name_repetitions)]
pub trait StackOrUnifiedHeap: sealed::StackOrUnifiedHeap {}
impl<T: sealed::StackOrUnifiedHeap> StackOrUnifiedHeap for T {}

mod sealed {
    use crate::utils::alloc::allocator::UnifiedAllocator;

    pub auto trait StackOrUnifiedHeap {}

    impl<T> !StackOrUnifiedHeap for *const T {}
    impl<T> !StackOrUnifiedHeap for *mut T {}
    impl<T> !StackOrUnifiedHeap for &T {}
    impl<T> !StackOrUnifiedHeap for &mut T {}

    impl<T> StackOrUnifiedHeap for core::marker::PhantomData<T> {}

    impl<T> StackOrUnifiedHeap for alloc::boxed::Box<T, UnifiedAllocator> {}
    impl<T> StackOrUnifiedHeap for alloc::vec::Vec<T, UnifiedAllocator> {}
    impl<T> StackOrUnifiedHeap for hashbrown::HashMap<T, UnifiedAllocator> {}
}
