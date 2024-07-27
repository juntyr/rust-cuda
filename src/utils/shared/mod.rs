mod slice;
mod r#static;

pub use slice::ThreadBlockSharedSlice;

#[expect(clippy::module_name_repetitions)]
pub use r#static::ThreadBlockShared;

#[doc(hidden)]
#[cfg(feature = "device")]
pub use slice::init;

#[cfg(feature = "host")]
pub(crate) use slice::SharedMemorySize;
