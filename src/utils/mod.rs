pub mod aliasing;
#[cfg(any(feature = "alloc", doc))]
#[doc(cfg(feature = "alloc"))]
pub mod alloc;
pub mod device_copy;
pub mod exchange;
pub mod shared;

mod r#box;
mod boxed_slice;
mod option;
