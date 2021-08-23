use core::ops::{Deref, DerefMut};

use rustacuda_core::DeviceCopy;

use crate::common::r#impl::RustToCudaImpl;

use super::CudaExchangeBufferCudaRepresentation;

#[allow(clippy::module_name_repetitions)]
pub struct CudaExchangeBufferDevice<T: DeviceCopy>(
    pub(super) core::mem::ManuallyDrop<alloc::boxed::Box<[T]>>,
);

impl<T: DeviceCopy> Deref for CudaExchangeBufferDevice<T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T: DeviceCopy> DerefMut for CudaExchangeBufferDevice<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

#[cfg(not(all(doc, feature = "host")))]
unsafe impl<T: DeviceCopy> RustToCudaImpl for CudaExchangeBufferDevice<T> {
    type CudaRepresentationImpl = CudaExchangeBufferCudaRepresentation<T>;
}
