use rustacuda_core::DeviceCopy;

use crate::common::CudaAsRust;

use super::CudaExchangeBuffer;

#[allow(clippy::module_name_repetitions)]
#[doc(hidden)]
#[repr(C)]
pub struct CudaExchangeBufferCudaRepresentation<T: DeviceCopy>(pub(super) *mut T, pub(super) usize);

// Safety: `CudaExchangeBufferCudaRepresentation<T>` is also `DeviceCopy`
//         iff `T` is `DeviceCopy`
unsafe impl<T: DeviceCopy> DeviceCopy for CudaExchangeBufferCudaRepresentation<T> {}

unsafe impl<T: DeviceCopy> CudaAsRust for CudaExchangeBufferCudaRepresentation<T> {
    type RustRepresentation = CudaExchangeBuffer<T>;

    #[cfg(any(not(feature = "host"), doc))]
    #[doc(cfg(not(feature = "host")))]
    unsafe fn as_rust(this: &crate::common::DeviceAccessible<Self>) -> Self::RustRepresentation {
        CudaExchangeBuffer(core::mem::ManuallyDrop::new(alloc::boxed::Box::from_raw(
            core::slice::from_raw_parts_mut(this.0, this.1),
        )))
    }
}
