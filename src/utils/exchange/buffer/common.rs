use rustacuda_core::DeviceCopy;

use crate::common::{CudaAsRust, DeviceOwnedSlice};

use super::CudaExchangeBuffer;

#[allow(clippy::module_name_repetitions)]
#[doc(hidden)]
#[repr(transparent)]
pub struct CudaExchangeBufferCudaRepresentation<T: DeviceCopy>(pub(super) DeviceOwnedSlice<T>);

// Safety: `CudaExchangeBufferCudaRepresentation<T>` is also `DeviceCopy`
//         iff `T` is `DeviceCopy`
unsafe impl<T: DeviceCopy> DeviceCopy for CudaExchangeBufferCudaRepresentation<T> {}

unsafe impl<T: DeviceCopy> CudaAsRust for CudaExchangeBufferCudaRepresentation<T> {
    type RustRepresentation = CudaExchangeBuffer<T>;

    #[cfg(any(not(feature = "host"), doc))]
    #[doc(cfg(not(feature = "host")))]
    unsafe fn as_rust(&mut self) -> Self::RustRepresentation {
        CudaExchangeBuffer(core::mem::ManuallyDrop::new(alloc::boxed::Box::from_raw(
            self.0.as_mut(),
        )))
    }
}
