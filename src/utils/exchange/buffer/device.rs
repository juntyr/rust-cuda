use core::ops::{Deref, DerefMut};

use const_type_layout::TypeGraphLayout;

use crate::{
    common::{NoCudaAlloc, RustToCuda, RustToCudaAsync},
    safety::SafeDeviceCopy,
};

use super::{common::CudaExchangeBufferCudaRepresentation, CudaExchangeItem};

#[allow(clippy::module_name_repetitions)]
#[doc(cfg(not(feature = "host")))]
/// When the `host` feature is set,
/// [`CudaExchangeBuffer`](super::CudaExchangeBuffer)
/// refers to
/// [`CudaExchangeBufferHost`](super::CudaExchangeBufferHost)
/// instead.
/// [`CudaExchangeBufferDevice`](Self) is never exposed directly.
pub struct CudaExchangeBufferDevice<T: SafeDeviceCopy, const M2D: bool, const M2H: bool>(
    pub(super) core::mem::ManuallyDrop<alloc::boxed::Box<[CudaExchangeItem<T, M2D, M2H>]>>,
);

impl<T: SafeDeviceCopy, const M2D: bool, const M2H: bool> Deref
    for CudaExchangeBufferDevice<T, M2D, M2H>
{
    type Target = [CudaExchangeItem<T, M2D, M2H>];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T: SafeDeviceCopy, const M2D: bool, const M2H: bool> DerefMut
    for CudaExchangeBufferDevice<T, M2D, M2H>
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

#[cfg(not(all(doc, feature = "host")))]
unsafe impl<T: SafeDeviceCopy + TypeGraphLayout, const M2D: bool, const M2H: bool> RustToCuda
    for CudaExchangeBufferDevice<T, M2D, M2H>
{
    type CudaAllocation = NoCudaAlloc;
    type CudaRepresentation = CudaExchangeBufferCudaRepresentation<T, M2D, M2H>;
}

#[cfg(not(all(doc, feature = "host")))]
unsafe impl<T: SafeDeviceCopy + TypeGraphLayout, const M2D: bool, const M2H: bool> RustToCudaAsync
    for CudaExchangeBufferDevice<T, M2D, M2H>
{
}
