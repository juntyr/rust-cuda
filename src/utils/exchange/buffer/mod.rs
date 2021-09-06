mod common;
#[cfg(any(not(feature = "host"), doc))]
#[doc(cfg(not(feature = "host")))]
mod device;
#[cfg(feature = "host")]
#[doc(cfg(feature = "host"))]
mod host;

#[cfg(not(feature = "host"))]
#[doc(cfg(not(feature = "host")))]
#[allow(clippy::module_name_repetitions)]
pub use device::CudaExchangeBufferDevice as CudaExchangeBuffer;
#[cfg(feature = "host")]
#[doc(cfg(feature = "host"))]
#[allow(clippy::module_name_repetitions)]
pub use host::CudaExchangeBufferHost as CudaExchangeBuffer;

#[cfg(all(doc, feature = "host"))]
/// If the `host` feature is enabled, `CudaExchangeBufferHost` is exported as
///  `CudaExchangeBuffer`.
pub use host::CudaExchangeBufferHost;

#[cfg(doc)]
/// If the `host` feature is NOT enabled, `CudaExchangeBufferDevice` is
///  exported as `CudaExchangeBuffer`.
pub use device::CudaExchangeBufferDevice;

pub use common::CudaExchangeBufferCudaRepresentation;
use rustacuda_core::DeviceCopy;

#[repr(transparent)]
#[derive(Clone)]
pub struct CudaExchangeItem<T: DeviceCopy, const M2D: bool, const M2H: bool>(T);

// Safety: Transparent newtype wrapper around `DeviceCopy`
//          is still `DeviceCopy`
unsafe impl<T: DeviceCopy, const M2D: bool, const M2H: bool> DeviceCopy
    for CudaExchangeItem<T, M2D, M2H>
{
}

impl<T: DeviceCopy> CudaExchangeItem<T, false, false> {}

impl<T: DeviceCopy> CudaExchangeItem<T, false, true> {
    #[cfg(any(feature = "host", doc))]
    #[doc(cfg(feature = "host"))]
    pub fn read(&self) -> &T {
        &self.0
    }

    #[cfg(any(not(feature = "host"), doc))]
    #[doc(cfg(not(feature = "host")))]
    pub fn write(&mut self, value: T) {
        self.0 = value;
    }
}

impl<T: DeviceCopy> CudaExchangeItem<T, true, false> {
    #[cfg(any(not(feature = "host"), doc))]
    #[doc(cfg(not(feature = "host")))]
    pub fn read(&self) -> &T {
        &self.0
    }

    #[cfg(any(feature = "host", doc))]
    #[doc(cfg(feature = "host"))]
    pub fn write(&mut self, value: T) {
        self.0 = value;
    }
}

impl<T: DeviceCopy> CudaExchangeItem<T, true, true> {
    pub fn read(&self) -> &T {
        &self.0
    }

    pub fn write(&mut self, value: T) {
        self.0 = value;
    }
}
