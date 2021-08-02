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
