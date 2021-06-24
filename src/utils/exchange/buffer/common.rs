use rustacuda_core::DeviceCopy;

use crate::common::{CudaAsRust, DeviceOwnedSlice};

use super::CudaExchangeBuffer;

#[allow(clippy::module_name_repetitions)]
#[doc(hidden)]
#[derive(DeviceCopy)]
pub struct CudaExchangeBufferCudaRepresentation<T: Clone + DeviceCopy>(
    pub(super) DeviceOwnedSlice<T>,
);

unsafe impl<T: Clone + DeviceCopy> CudaAsRust for CudaExchangeBufferCudaRepresentation<T> {
    type RustRepresentation = CudaExchangeBuffer<T>;

    #[cfg(target_os = "cuda")]
    unsafe fn as_rust(&mut self) -> Self::RustRepresentation {
        CudaExchangeBuffer(core::mem::ManuallyDrop::new(alloc::boxed::Box::from_raw(
            self.0.as_mut(),
        )))
    }
}
