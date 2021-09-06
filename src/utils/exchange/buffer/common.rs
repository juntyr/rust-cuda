use rustacuda_core::DeviceCopy;

use crate::common::CudaAsRust;

use super::{CudaExchangeBuffer, CudaExchangeItem};

#[allow(clippy::module_name_repetitions)]
#[doc(hidden)]
#[repr(C)]
pub struct CudaExchangeBufferCudaRepresentation<T: DeviceCopy, const M2D: bool, const M2H: bool>(
    pub(super) *mut CudaExchangeItem<T, M2D, M2H>,
    pub(super) usize,
);

// Safety: `CudaExchangeBufferCudaRepresentation<T>` is also `DeviceCopy`
//         iff `T` is `DeviceCopy`
unsafe impl<T: DeviceCopy, const M2D: bool, const M2H: bool> DeviceCopy
    for CudaExchangeBufferCudaRepresentation<T, M2D, M2H>
{
}

unsafe impl<T: DeviceCopy, const M2D: bool, const M2H: bool> CudaAsRust
    for CudaExchangeBufferCudaRepresentation<T, M2D, M2H>
{
    type RustRepresentation = CudaExchangeBuffer<T, M2D, M2H>;

    #[cfg(any(not(feature = "host"), doc))]
    #[doc(cfg(not(feature = "host")))]
    unsafe fn as_rust(this: &crate::common::DeviceAccessible<Self>) -> Self::RustRepresentation {
        CudaExchangeBuffer(core::mem::ManuallyDrop::new(alloc::boxed::Box::from_raw(
            core::slice::from_raw_parts_mut(this.0, this.1),
        )))
    }
}
