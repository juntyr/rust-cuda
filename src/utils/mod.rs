pub mod aliasing;
#[cfg(any(feature = "alloc", doc))]
#[doc(cfg(feature = "alloc"))]
pub mod alloc;
pub mod exchange;
pub mod stack;

mod r#box;
mod boxed_slice;
mod option;
