#[allow(clippy::module_name_repetitions)]
pub trait NoAliasing: private::NoAliasing {}
impl<T: private::NoAliasing> NoAliasing for T {}

mod private {
    pub auto trait NoAliasing {}

    impl<T> !NoAliasing for *const T {}
    impl<T> !NoAliasing for *mut T {}
    impl<T> !NoAliasing for &mut T {}

    impl<T> NoAliasing for core::marker::PhantomData<T> {}

    impl<T> NoAliasing for r#final::Final<T> {}
    impl<T: crate::common::CudaAsRust> NoAliasing
        for crate::utils::aliasing::FinalCudaRepresentation<T>
    {
    }

    impl<T, const STRIDE: usize> NoAliasing
        for crate::utils::aliasing::SplitSliceOverCudaThreadsConstStride<T, STRIDE>
    {
    }
    impl<T> NoAliasing for crate::utils::aliasing::SplitSliceOverCudaThreadsDynamicStride<T> {}

    // Thread-block-shared data only allows unsafe aliasing since only raw pointers
    //  are exposed
    impl<T: 'static> NoAliasing for crate::utils::shared::r#static::ThreadBlockShared<T> {}
    impl<T: 'static + const_type_layout::TypeGraphLayout> NoAliasing
        for crate::utils::shared::slice::ThreadBlockSharedSlice<T>
    {
    }
}
