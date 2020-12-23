use core::ops::{Deref, DerefMut};

use rustacuda_core::DeviceCopy;

use crate::common::RustToCuda;

use super::CudaExchangeBufferCudaRepresentation;

#[allow(clippy::module_name_repetitions)]
pub struct CudaExchangeBufferDevice<T: Clone + DeviceCopy>(
    pub(super) core::mem::ManuallyDrop<alloc::boxed::Box<[T]>>,
);

impl<T: Clone + DeviceCopy> Deref for CudaExchangeBufferDevice<T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T: Clone + DeviceCopy> DerefMut for CudaExchangeBufferDevice<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

unsafe impl<T: Clone + DeviceCopy> RustToCuda for CudaExchangeBufferDevice<T> {
    type CudaRepresentation = CudaExchangeBufferCudaRepresentation<T>;
}
