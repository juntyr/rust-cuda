use const_type_layout::TypeGraphLayout;
use rustacuda_core::DeviceCopy;

use crate::{common::CudaAsRust, safety::SafeDeviceCopy};

use super::{CudaExchangeBuffer, CudaExchangeItem};

#[allow(clippy::module_name_repetitions)]
#[doc(hidden)]
#[derive(TypeLayout)]
#[repr(C)]
pub struct CudaExchangeBufferCudaRepresentation<T, const M2D: bool, const M2H: bool>(
    pub(super) *mut CudaExchangeItem<T, M2D, M2H>,
    pub(super) usize,
)
where
    T: SafeDeviceCopy + TypeGraphLayout;

// Safety: [`CudaExchangeBufferCudaRepresentation<T>`] is [`DeviceCopy`]
//         iff [`T`] is [`SafeDeviceCopy`]
unsafe impl<T: SafeDeviceCopy + TypeGraphLayout, const M2D: bool, const M2H: bool> DeviceCopy
    for CudaExchangeBufferCudaRepresentation<T, M2D, M2H>
{
}

unsafe impl<T: SafeDeviceCopy + TypeGraphLayout, const M2D: bool, const M2H: bool> CudaAsRust
    for CudaExchangeBufferCudaRepresentation<T, M2D, M2H>
{
    type RustRepresentation = CudaExchangeBuffer<T, M2D, M2H>;

    #[cfg(not(feature = "host"))]
    unsafe fn as_rust(this: &crate::common::DeviceAccessible<Self>) -> Self::RustRepresentation {
        CudaExchangeBuffer(core::mem::ManuallyDrop::new(alloc::boxed::Box::from_raw(
            core::slice::from_raw_parts_mut(this.0, this.1),
        )))
    }
}
