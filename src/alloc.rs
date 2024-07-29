#![expect(clippy::module_name_repetitions)]

pub trait EmptyCudaAlloc: From<NoCudaAlloc> + Into<NoCudaAlloc> + sealed::empty::Sealed {}

pub trait CudaAlloc: sealed::alloc::Sealed {}

impl<T: CudaAlloc> CudaAlloc for Option<T> {}
impl<T: CudaAlloc> sealed::alloc::Sealed for Option<T> {}

pub struct NoCudaAlloc;
impl CudaAlloc for NoCudaAlloc {}
impl sealed::alloc::Sealed for NoCudaAlloc {}
impl EmptyCudaAlloc for NoCudaAlloc {}
impl sealed::empty::Sealed for NoCudaAlloc {}

pub struct SomeCudaAlloc(());
impl CudaAlloc for SomeCudaAlloc {}
impl sealed::alloc::Sealed for SomeCudaAlloc {}
impl !EmptyCudaAlloc for SomeCudaAlloc {}
impl !sealed::empty::Sealed for SomeCudaAlloc {}

pub struct CombinedCudaAlloc<A: CudaAlloc, B: CudaAlloc>(A, B);
impl<A: CudaAlloc, B: CudaAlloc> CudaAlloc for CombinedCudaAlloc<A, B> {}
impl<A: CudaAlloc, B: CudaAlloc> sealed::alloc::Sealed for CombinedCudaAlloc<A, B> {}
impl<A: CudaAlloc + EmptyCudaAlloc, B: CudaAlloc + EmptyCudaAlloc> EmptyCudaAlloc
    for CombinedCudaAlloc<A, B>
{
}
impl<A: CudaAlloc + EmptyCudaAlloc, B: CudaAlloc + EmptyCudaAlloc> sealed::empty::Sealed
    for CombinedCudaAlloc<A, B>
{
}
impl<A: CudaAlloc + EmptyCudaAlloc, B: CudaAlloc + EmptyCudaAlloc> From<NoCudaAlloc>
    for CombinedCudaAlloc<A, B>
{
    fn from(_: NoCudaAlloc) -> Self {
        Self(A::from(NoCudaAlloc), B::from(NoCudaAlloc))
    }
}
impl<A: CudaAlloc + EmptyCudaAlloc, B: CudaAlloc + EmptyCudaAlloc> From<CombinedCudaAlloc<A, B>>
    for NoCudaAlloc
{
    fn from(val: CombinedCudaAlloc<A, B>) -> Self {
        let _: (Self, Self) = (val.0.into(), val.1.into());
        Self
    }
}
impl<A: CudaAlloc, B: CudaAlloc> CombinedCudaAlloc<A, B> {
    #[must_use]
    pub const fn new(front: A, tail: B) -> Self {
        Self(front, tail)
    }

    pub fn split(self) -> (A, B) {
        (self.0, self.1)
    }
}

pub(crate) mod sealed {
    pub(super) mod empty {
        pub trait Sealed {}
    }

    pub mod alloc {
        pub trait Sealed {}
    }
}
