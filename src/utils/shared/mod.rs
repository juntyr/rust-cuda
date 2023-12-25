mod slice;
mod r#static;

pub use slice::ThreadBlockSharedSlice;

#[allow(clippy::module_name_repetitions)]
pub use r#static::ThreadBlockShared;

#[doc(hidden)]
#[cfg(feature = "device")]
pub use slice::init;
