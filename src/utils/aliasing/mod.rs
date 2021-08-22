pub mod r#const;
pub mod dynamic;
pub mod r#final;

#[allow(clippy::module_name_repetitions)]
pub trait NoAliasing: private::NoAliasing {}
impl<T: private::NoAliasing> NoAliasing for T {}

mod private {
    use crate::common::CudaAsRust;

    use super::{
        dynamic::SplitSliceOverCudaThreadsDynamicStride,
        r#const::SplitSliceOverCudaThreadsConstStride,
        r#final::{Final, FinalCudaRepresentation},
    };

    pub auto trait NoAliasing {}

    impl<T> !NoAliasing for *const T {}
    impl<T> !NoAliasing for *mut T {}
    impl<T> !NoAliasing for &T {}
    impl<T> !NoAliasing for &mut T {}

    impl<T> NoAliasing for core::marker::PhantomData<T> {}

    impl<T> NoAliasing for Final<T> {}
    impl<T: CudaAsRust> NoAliasing for FinalCudaRepresentation<T> {}

    impl<T, const STRIDE: usize> NoAliasing for SplitSliceOverCudaThreadsConstStride<T, STRIDE> {}
    impl<T> NoAliasing for SplitSliceOverCudaThreadsDynamicStride<T> {}
}
