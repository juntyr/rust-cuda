#[cfg(not(feature = "host"))]
use core::convert::{AsMut, AsRef};

use rustacuda_core::DeviceCopy;

#[cfg(feature = "derive")]
pub use rust_cuda_derive::RustToCuda;

#[cfg(feature = "derive")]
pub use rust_cuda_derive::kernel;

/// # Safety
/// This is an internal trait and should ONLY be derived automatically using
/// `#[derive(RustToCuda)]`
pub unsafe trait RustToCuda {
    type CudaRepresentation: DeviceCopy + CudaAsRust<RustRepresentation = Self>;

    #[cfg(feature = "host")]
    type CudaAllocation: crate::host::CudaAlloc;

    #[cfg(feature = "host")]
    /// # Errors
    /// Returns a `rustacuda::errors::CudaError` iff an error occurs inside CUDA
    /// # Safety
    /// This is an internal function and should NEVER be called manually
    /// The returned `Self::CudaRepresentation` must NEVER be accessed mutably
    /// (guaranteed by user-facing `LendToCuda`)
    #[allow(clippy::type_complexity)]
    unsafe fn borrow<A: crate::host::CudaAlloc>(
        &self,
        alloc: A,
    ) -> rustacuda::error::CudaResult<(
        Self::CudaRepresentation,
        crate::host::CombinedCudaAlloc<Self::CudaAllocation, A>,
    )> {
        // This change in mutability is only safe iff the mutability is not exposed in
        // `Self::CudaRepresentation`
        #[allow(clippy::cast_ref_to_mut)]
        Self::borrow_mut(&mut *(self as *const Self as *mut Self), alloc)
    }

    #[cfg(feature = "host")]
    /// # Errors
    /// Returns a `rustacuda::errors::CudaError` iff an error occurs inside CUDA
    /// # Safety
    /// This is an internal function and should NEVER be called manually
    #[allow(clippy::type_complexity)]
    unsafe fn borrow_mut<A: crate::host::CudaAlloc>(
        &mut self,
        alloc: A,
    ) -> rustacuda::error::CudaResult<(
        Self::CudaRepresentation,
        crate::host::CombinedCudaAlloc<Self::CudaAllocation, A>,
    )>;

    #[cfg(feature = "host")]
    /// # Errors
    /// Returns a `rustacuda::errors::CudaError` iff an error occurs inside CUDA
    /// # Safety
    /// This is an internal function and should NEVER be called manually
    #[allow(clippy::type_complexity)]
    unsafe fn un_borrow_mut<A: crate::host::CudaAlloc>(
        &mut self,
        cuda_repr: Self::CudaRepresentation,
        alloc: crate::host::CombinedCudaAlloc<Self::CudaAllocation, A>,
    ) -> rustacuda::error::CudaResult<A>;
}

/// # Safety
/// This is an internal trait and should NEVER be implemented manually
pub unsafe trait CudaAsRust {
    type RustRepresentation: RustToCuda<CudaRepresentation = Self>;

    #[cfg(not(feature = "host"))]
    /// # Safety
    /// This is an internal function and should NEVER be called manually
    unsafe fn as_rust(&mut self) -> Self::RustRepresentation;
}

#[derive(Debug)]
pub struct DeviceOwnedSlice<T: Sized + DeviceCopy>(*mut T, usize);

unsafe impl<T: Sized + DeviceCopy> DeviceCopy for DeviceOwnedSlice<T> {}

#[cfg(feature = "host")]
impl<T: Sized + DeviceCopy> DeviceOwnedSlice<T> {
    pub fn from(owned_slice: &mut rustacuda::memory::DeviceBuffer<T>) -> Self {
        Self(owned_slice.as_mut_ptr(), owned_slice.len())
    }
}

#[cfg(not(feature = "host"))]
impl<T: Sized + DeviceCopy> AsRef<[T]> for DeviceOwnedSlice<T> {
    fn as_ref(&self) -> &[T] {
        unsafe { core::slice::from_raw_parts(self.0, self.1) }
    }
}

#[cfg(not(feature = "host"))]
impl<T: Sized + DeviceCopy> AsMut<[T]> for DeviceOwnedSlice<T> {
    fn as_mut(&mut self) -> &mut [T] {
        unsafe { core::slice::from_raw_parts_mut(self.0, self.1) }
    }
}

#[repr(transparent)]
pub struct DeviceBoxConst<T: Sized + DeviceCopy>(pub(super) *const T);

impl<T: Sized + DeviceCopy> Clone for DeviceBoxConst<T> {
    fn clone(&self) -> Self {
        Self(self.0)
    }
}
impl<T: Sized + DeviceCopy> Copy for DeviceBoxConst<T> {}
unsafe impl<T: Sized + DeviceCopy> DeviceCopy for DeviceBoxConst<T> {}

#[cfg(feature = "host")]
impl<T: Sized + DeviceCopy> DeviceBoxConst<T> {
    #[must_use]
    pub fn from(device_box: &rustacuda::memory::DeviceBox<T>) -> Self {
        // This is only safe because we only expose the immutable *const T
        #[allow(clippy::cast_ref_to_mut)]
        Self(
            unsafe {
                &mut *(device_box as *const rustacuda::memory::DeviceBox<T>
                    as *mut rustacuda::memory::DeviceBox<T>)
            }
            .as_device_ptr()
            .as_raw(),
        )
    }
}

#[cfg(not(feature = "host"))]
impl<T: Sized + DeviceCopy> AsRef<T> for DeviceBoxConst<T> {
    fn as_ref(&self) -> &T {
        unsafe { &*self.0 }
    }
}

#[repr(transparent)]
pub struct DeviceBoxMut<T: Sized + DeviceCopy>(pub(super) *mut T);

unsafe impl<T: Sized + DeviceCopy> DeviceCopy for DeviceBoxMut<T> {}

#[cfg(feature = "host")]
impl<T: Sized + DeviceCopy> DeviceBoxMut<T> {
    #[must_use]
    pub fn from(device_box_mut: &mut rustacuda::memory::DeviceBox<T>) -> Self {
        Self(device_box_mut.as_device_ptr().as_raw_mut())
    }
}

#[cfg(not(feature = "host"))]
impl<T: Sized + DeviceCopy> AsRef<T> for DeviceBoxMut<T> {
    fn as_ref(&self) -> &T {
        unsafe { &*self.0 }
    }
}

#[cfg(not(feature = "host"))]
impl<T: Sized + DeviceCopy> AsMut<T> for DeviceBoxMut<T> {
    fn as_mut(&mut self) -> &mut T {
        unsafe { &mut *self.0 }
    }
}
