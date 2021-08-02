mod common;
#[cfg(any(not(feature = "host"), doc))]
mod device;
#[cfg(any(feature = "host", doc))]
mod host;

#[cfg(any(not(feature = "host"), doc))]
#[allow(clippy::module_name_repetitions)]
pub use device::CudaExchangeBufferDevice as CudaExchangeBuffer;
#[cfg(any(feature = "host", doc))]
#[allow(clippy::module_name_repetitions)]
pub use host::CudaExchangeBufferHost as CudaExchangeBuffer;

pub use common::CudaExchangeBufferCudaRepresentation;
