#[doc(cfg(feature = "alloc"))]
/// ```rust
/// # use rust_cuda::safety::UnifiedHeapOnly;
/// fn assert_unified_heap_only(_x: impl UnifiedHeapOnly) {}
/// ```
/// ```rust
/// # use rust_cuda::safety::UnifiedHeapOnly;
/// # fn assert_unified_heap_only(_x: impl UnifiedHeapOnly) {}
/// assert_unified_heap_only(42);
/// ```
/// ```rust
/// # use rust_cuda::safety::UnifiedHeapOnly;
/// # fn assert_unified_heap_only(_x: impl UnifiedHeapOnly) {}
/// assert_unified_heap_only([42; 42]);
/// ```
/// ```rust,compile_fail
/// # use alloc::vec;
/// # use rust_cuda::safety::UnifiedHeapOnly;
/// # fn assert_unified_heap_only(_x: impl UnifiedHeapOnly) {}
/// assert_unified_heap_only(vec![42]);
/// ```
/// ```rust,compile_fail
/// # use rust_cuda::safety::UnifiedHeapOnly;
/// # fn assert_unified_heap_only(_x: impl UnifiedHeapOnly) {}
/// assert_unified_heap_only(&42);
/// ```
#[allow(clippy::module_name_repetitions)]
pub trait UnifiedHeapOnly: sealed::UnifiedHeapOnly {}
impl<T: sealed::UnifiedHeapOnly> UnifiedHeapOnly for T {}

mod sealed {
    use crate::utils::alloc::UnifiedAllocator;

    pub auto trait UnifiedHeapOnly {}

    impl<T> !UnifiedHeapOnly for *const T {}
    impl<T> !UnifiedHeapOnly for *mut T {}
    impl<T> !UnifiedHeapOnly for &T {}
    impl<T> !UnifiedHeapOnly for &mut T {}

    // Thread-block-shared data contains CUDA-only data
    impl<T: 'static> !UnifiedHeapOnly for crate::utils::shared::r#static::ThreadBlockShared<T> {}
    impl<T: 'static + const_type_layout::TypeGraphLayout> !UnifiedHeapOnly
        for crate::utils::shared::slice::ThreadBlockSharedSlice<T>
    {
    }

    impl<T> UnifiedHeapOnly for core::marker::PhantomData<T> {}

    impl<T> UnifiedHeapOnly for alloc::boxed::Box<T, UnifiedAllocator> {}
    impl<T> UnifiedHeapOnly for alloc::vec::Vec<T, UnifiedAllocator> {}
    impl<T> UnifiedHeapOnly for hashbrown::HashMap<T, UnifiedAllocator> {}
}
