use core::mem::MaybeUninit;

mod common;
#[cfg(any(not(feature = "host"), doc))]
mod device;
#[cfg(feature = "host")]
mod host;

#[cfg(not(feature = "host"))]
#[allow(clippy::module_name_repetitions)]
pub use device::CudaExchangeBufferDevice as CudaExchangeBuffer;
#[cfg(feature = "host")]
#[allow(clippy::module_name_repetitions)]
pub use host::CudaExchangeBufferHost as CudaExchangeBuffer;

#[cfg(doc)]
pub use self::{device::CudaExchangeBufferDevice, host::CudaExchangeBufferHost};

use crate::safety::SafeDeviceCopy;

#[repr(transparent)]
#[derive(Clone, Copy, TypeLayout)]
pub struct CudaExchangeItem<T: SafeDeviceCopy, const M2D: bool, const M2H: bool>(T);

// Safety: Transparent newtype wrapper around [`SafeDeviceCopy`]
//          is [`DeviceCopy`]
unsafe impl<T: SafeDeviceCopy, const M2D: bool, const M2H: bool> rustacuda_core::DeviceCopy
    for CudaExchangeItem<T, M2D, M2H>
{
}

impl<T: SafeDeviceCopy, const M2D: bool> CudaExchangeItem<T, M2D, true> {
    #[cfg(any(feature = "host", doc))]
    #[doc(cfg(feature = "host"))]
    pub const fn read(&self) -> &T {
        &self.0
    }

    #[cfg(any(not(feature = "host"), doc))]
    #[doc(cfg(not(feature = "host")))]
    pub fn write(&mut self, value: T) {
        self.0 = value;
    }
}

impl<T: SafeDeviceCopy, const M2H: bool> CudaExchangeItem<T, true, M2H> {
    #[cfg(any(not(feature = "host"), doc))]
    #[doc(cfg(not(feature = "host")))]
    pub const fn read(&self) -> &T {
        &self.0
    }

    #[cfg(any(feature = "host", doc))]
    #[doc(cfg(feature = "host"))]
    pub fn write(&mut self, value: T) {
        self.0 = value;
    }
}

impl<T: SafeDeviceCopy> AsMut<T> for CudaExchangeItem<T, true, true> {
    fn as_mut(&mut self) -> &mut T {
        &mut self.0
    }
}

impl<T: SafeDeviceCopy> CudaExchangeItem<T, false, true> {
    #[cfg(any(feature = "host", doc))]
    #[doc(cfg(feature = "host"))]
    pub const fn as_scratch(&self) -> &T {
        &self.0
    }

    #[cfg(any(feature = "host", doc))]
    #[doc(cfg(feature = "host"))]
    pub fn as_scratch_mut(&mut self) -> &mut T {
        &mut self.0
    }
}

impl<T: SafeDeviceCopy> CudaExchangeItem<T, true, false> {
    #[cfg(any(not(feature = "host"), doc))]
    #[doc(cfg(not(feature = "host")))]
    pub const fn as_scratch(&self) -> &T {
        &self.0
    }

    #[cfg(any(not(feature = "host"), doc))]
    #[doc(cfg(not(feature = "host")))]
    pub fn as_scratch_mut(&mut self) -> &mut T {
        &mut self.0
    }
}

impl<T: SafeDeviceCopy> CudaExchangeItem<T, true, false> {
    #[cfg(any(feature = "host", doc))]
    #[doc(cfg(feature = "host"))]
    pub const fn as_uninit(&self) -> &MaybeUninit<T> {
        // Safety:
        // - MaybeUninit is a transparent newtype union
        // - CudaExchangeItem is a transparent newtype
        unsafe { &*(self as *const Self).cast() }
    }

    #[cfg(any(feature = "host", doc))]
    #[doc(cfg(feature = "host"))]
    pub fn as_uninit_mut(&mut self) -> &mut MaybeUninit<T> {
        // Safety:
        // - MaybeUninit is a transparent newtype union
        // - CudaExchangeItem is a transparent newtype
        unsafe { &mut *(self as *mut Self).cast() }
    }
}

impl<T: SafeDeviceCopy> CudaExchangeItem<T, false, true> {
    #[cfg(any(not(feature = "host"), doc))]
    #[doc(cfg(not(feature = "host")))]
    pub const fn as_uninit(&self) -> &MaybeUninit<T> {
        // Safety:
        // - MaybeUninit is a transparent newtype union
        // - CudaExchangeItem is a transparent newtype
        unsafe { &*(self as *const Self).cast() }
    }

    #[cfg(any(not(feature = "host"), doc))]
    #[doc(cfg(not(feature = "host")))]
    pub fn as_uninit_mut(&mut self) -> &mut MaybeUninit<T> {
        // Safety:
        // - MaybeUninit is a transparent newtype union
        // - CudaExchangeItem is a transparent newtype
        unsafe { &mut *(self as *mut Self).cast() }
    }
}
