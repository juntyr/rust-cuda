use const_type_layout::{TypeGraphLayout, TypeLayout};

use crate::{
    lend::CudaAsRust,
    safety::{PortableBitSemantics, StackOnly},
    utils::ffi::DeviceMutPointer,
};

use super::{CudaExchangeBuffer, CudaExchangeItem};

#[doc(hidden)]
#[derive(TypeLayout)]
#[repr(C)]
pub struct CudaExchangeBufferCudaRepresentation<
    T: StackOnly + PortableBitSemantics + TypeGraphLayout,
    const M2D: bool,
    const M2H: bool,
>(
    pub(super) DeviceMutPointer<CudaExchangeItem<T, M2D, M2H>>,
    pub(super) usize,
);

unsafe impl<T: StackOnly + PortableBitSemantics + TypeGraphLayout, const M2D: bool, const M2H: bool>
    CudaAsRust for CudaExchangeBufferCudaRepresentation<T, M2D, M2H>
{
    type RustRepresentation = CudaExchangeBuffer<T, M2D, M2H>;

    #[cfg(feature = "device")]
    unsafe fn as_rust(
        this: &crate::utils::ffi::DeviceAccessible<Self>,
    ) -> Self::RustRepresentation {
        CudaExchangeBuffer {
            inner: super::device::CudaExchangeBufferDevice(core::mem::ManuallyDrop::new(
                crate::deps::alloc::boxed::Box::from_raw(core::slice::from_raw_parts_mut(
                    (**this).0.0, this.1,
                )),
            )),
        }
    }
}
