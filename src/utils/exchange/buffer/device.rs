use core::ops::{Deref, DerefMut};

use const_type_layout::TypeGraphLayout;

use crate::{deps::alloc::boxed::Box, safety::PortableBitSemantics};

use super::CudaExchangeItem;

#[allow(clippy::module_name_repetitions)]
pub struct CudaExchangeBufferDevice<
    T: PortableBitSemantics + TypeGraphLayout,
    const M2D: bool,
    const M2H: bool,
>(pub(super) core::mem::ManuallyDrop<Box<[CudaExchangeItem<T, M2D, M2H>]>>);

impl<T: PortableBitSemantics + TypeGraphLayout, const M2D: bool, const M2H: bool> Deref
    for CudaExchangeBufferDevice<T, M2D, M2H>
{
    type Target = [CudaExchangeItem<T, M2D, M2H>];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T: PortableBitSemantics + TypeGraphLayout, const M2D: bool, const M2H: bool> DerefMut
    for CudaExchangeBufferDevice<T, M2D, M2H>
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
