mod device_copy;
mod no_aliasing;
mod register_fit;
mod stack_only;
#[cfg(any(feature = "alloc", doc))]
mod unified_heap;

pub use device_copy::SafeDeviceCopy;
pub use no_aliasing::NoAliasing;
pub use register_fit::FitsIntoDeviceRegister;
pub use stack_only::StackOnly;
#[cfg(any(feature = "alloc", doc))]
pub use unified_heap::UnifiedHeapOnly;
