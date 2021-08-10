pub mod r#const;
pub mod dynamic;
pub mod r#final;

#[allow(clippy::module_name_repetitions)]
pub trait NoAliasing: private::NoAliasing {}
impl<T: private::NoAliasing> NoAliasing for T {}

mod private {
    use super::{
        dynamic::SplitSliceOverCudaThreadsDynamicStride,
        r#const::SplitSliceOverCudaThreadsConstStride, r#final::Final,
    };

    pub auto trait NoAliasing {}

    impl<T> !NoAliasing for *const T {}
    impl<T> !NoAliasing for *mut T {}
    impl<T> !NoAliasing for &T {}
    impl<T> !NoAliasing for &mut T {}

    impl<T> NoAliasing for Final<T> {}
    impl<T, const STRIDE: usize> NoAliasing for SplitSliceOverCudaThreadsConstStride<T, STRIDE> {}
    impl<T> NoAliasing for SplitSliceOverCudaThreadsDynamicStride<T> {}
}

// TODO: conditionally impl RustToCuda etc for the three wrappers
