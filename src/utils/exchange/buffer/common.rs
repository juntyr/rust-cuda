use rustacuda_core::DeviceCopy;

use crate::common::{CudaAsRust, DeviceOwnedSlice};

use super::CudaExchangeBuffer;

#[allow(clippy::module_name_repetitions)]
#[doc(hidden)]
pub struct CudaExchangeBufferCudaRepresentation<T: Clone + DeviceCopy>(
    pub(super) DeviceOwnedSlice<T>,
);

// Safety: `CudaExchangeBufferCudaRepresentation<T>` is also `DeviceCopy`
//         iff `T` is `DeviceCopy`
unsafe impl<T: Clone + DeviceCopy> DeviceCopy for CudaExchangeBufferCudaRepresentation<T> {}

unsafe impl<T: Clone + DeviceCopy> CudaAsRust for CudaExchangeBufferCudaRepresentation<T> {
    type RustRepresentation = CudaExchangeBuffer<T>;

    #[cfg(any(not(feature = "host"), doc))]
    unsafe fn as_rust(&mut self) -> Self::RustRepresentation {
        CudaExchangeBuffer(core::mem::ManuallyDrop::new(alloc::boxed::Box::from_raw(
            self.0.as_mut(),
        )))
    }
}
