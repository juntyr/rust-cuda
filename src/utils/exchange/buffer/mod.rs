mod common;
#[cfg(not(feature = "host"))]
mod device;
#[cfg(feature = "host")]
mod host;

#[cfg(not(feature = "host"))]
pub use device::CudaExchangeBufferDevice as CudaExchangeBuffer;
#[cfg(feature = "host")]
pub use host::CudaExchangeBufferHost as CudaExchangeBuffer;

pub use common::CudaExchangeBufferCudaRepresentation;
