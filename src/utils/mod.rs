pub mod aliasing;
#[cfg(any(feature = "alloc", doc))]
#[doc(cfg(feature = "alloc"))]
pub mod alloc;
pub mod exchange;
pub mod stack;

mod r#box;
mod boxed_slice;
mod option;

pub trait SafeDeviceCopy: sealed::SafeDeviceCopy {}

impl<T: sealed::SafeDeviceCopy> SafeDeviceCopy for T {}

mod sealed {
    #[marker]
    pub trait SafeDeviceCopy {}

    impl<T: super::stack::StackOnly> SafeDeviceCopy for T {}
    #[cfg(any(feature = "alloc", doc))]
    impl<T: super::alloc::unified::StackOrUnified> SafeDeviceCopy for T {}
}
