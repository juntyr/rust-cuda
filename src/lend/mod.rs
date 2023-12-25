use const_type_layout::TypeGraphLayout;
#[cfg(feature = "host")]
use rustacuda::error::CudaError;
use rustacuda_core::DeviceCopy;

#[cfg(feature = "derive")]
#[allow(clippy::module_name_repetitions)]
pub use rust_cuda_derive::LendRustToCuda;

use crate::{alloc::CudaAlloc, utils::ffi::DeviceAccessible};

#[cfg(feature = "device")]
use crate::utils::ffi::{DeviceConstRef, DeviceOwnedRef};
#[cfg(feature = "host")]
use crate::{
    alloc::{CombinedCudaAlloc, EmptyCudaAlloc, NoCudaAlloc},
    host::{HostAndDeviceConstRef, HostAndDeviceOwned},
};

#[cfg(any(feature = "host", feature = "device"))]
use crate::safety::{NoSafeAliasing, SafeDeviceCopy};

mod impls;

/// # Safety
///
/// This is an internal trait and should ONLY be derived automatically using
/// `#[derive(LendRustToCuda)]`
pub unsafe trait RustToCuda {
    type CudaAllocation: CudaAlloc;
    type CudaRepresentation: CudaAsRust<RustRepresentation = Self>;

    #[doc(hidden)]
    #[cfg(feature = "host")]
    /// # Errors
    ///
    /// Returns a [`rustacuda::error::CudaError`] iff an error occurs inside
    /// CUDA
    ///
    /// # Safety
    ///
    /// This is an internal function and should NEVER be called manually
    /// The returned [`Self::CudaRepresentation`] must NEVER be accessed on the
    ///  CPU  as it contains a GPU-resident copy of `self`.
    #[allow(clippy::type_complexity)]
    unsafe fn borrow<A: CudaAlloc>(
        &self,
        alloc: A,
    ) -> rustacuda::error::CudaResult<(
        DeviceAccessible<Self::CudaRepresentation>,
        CombinedCudaAlloc<Self::CudaAllocation, A>,
    )>;

    #[doc(hidden)]
    #[cfg(feature = "host")]
    /// # Errors
    ///
    /// Returns a [`rustacuda::error::CudaError`] iff an error occurs inside
    /// CUDA
    ///
    /// # Safety
    ///
    /// This is an internal function and should NEVER be called manually
    #[allow(clippy::type_complexity)]
    unsafe fn restore<A: CudaAlloc>(
        &mut self,
        alloc: CombinedCudaAlloc<Self::CudaAllocation, A>,
    ) -> rustacuda::error::CudaResult<A>;
}

/// # Safety
///
/// This is an internal trait and should ONLY be derived automatically using
/// `#[derive(LendRustToCuda)]`
pub unsafe trait RustToCudaAsync: RustToCuda {
    #[doc(hidden)]
    #[cfg(feature = "host")]
    /// # Errors
    ///
    /// Returns a [`rustacuda::error::CudaError`] iff an error occurs inside
    /// CUDA
    ///
    /// # Safety
    ///
    /// This is an internal function and should NEVER be called manually
    /// The returned
    /// [`Self::CudaRepresentation`](RustToCuda::CudaRepresentation) must NEVER
    /// be accessed on the  CPU  as it contains a GPU-resident copy of
    /// `self`.
    #[allow(clippy::type_complexity)]
    unsafe fn borrow_async<A: CudaAlloc>(
        &self,
        alloc: A,
        stream: &rustacuda::stream::Stream,
    ) -> rustacuda::error::CudaResult<(
        DeviceAccessible<Self::CudaRepresentation>,
        CombinedCudaAlloc<Self::CudaAllocation, A>,
    )>;

    #[doc(hidden)]
    #[cfg(feature = "host")]
    /// # Errors
    ///
    /// Returns a [`rustacuda::error::CudaError`] iff an error occurs inside
    /// CUDA
    ///
    /// # Safety
    ///
    /// This is an internal function and should NEVER be called manually
    #[allow(clippy::type_complexity)]
    unsafe fn restore_async<A: CudaAlloc>(
        &mut self,
        alloc: CombinedCudaAlloc<Self::CudaAllocation, A>,
        stream: &rustacuda::stream::Stream,
    ) -> rustacuda::error::CudaResult<A>;
}

/// # Safety
///
/// This is an internal trait and should NEVER be implemented manually
pub unsafe trait CudaAsRust: DeviceCopy + TypeGraphLayout {
    type RustRepresentation: RustToCuda<CudaRepresentation = Self>;

    #[doc(hidden)]
    #[cfg(feature = "device")]
    /// # Safety
    ///
    /// This is an internal function and should NEVER be called manually
    unsafe fn as_rust(this: &DeviceAccessible<Self>) -> Self::RustRepresentation;
}

pub trait RustToCudaProxy<T>: RustToCuda {
    fn from_ref(val: &T) -> &Self;
    fn from_mut(val: &mut T) -> &mut Self;

    fn into(self) -> T;
}

pub trait RustToCudaAsyncProxy<T>: RustToCudaAsync {
    fn from_ref(val: &T) -> &Self;
    fn from_mut(val: &mut T) -> &mut Self;

    fn into(self) -> T;
}

#[cfg(feature = "host")]
#[allow(clippy::module_name_repetitions)]
pub trait LendToCuda: RustToCuda + NoSafeAliasing {
    /// Lends an immutable copy of `&self` to CUDA:
    /// - code in the CUDA kernel can only access `&self` through the
    ///   [`DeviceConstRef`] inside the closure
    /// - after the closure, `&self` will not have changed
    ///
    /// # Errors
    ///
    /// Returns a [`CudaError`] iff an error occurs inside CUDA
    fn lend_to_cuda<
        O,
        E: From<CudaError>,
        F: FnOnce(
            HostAndDeviceConstRef<DeviceAccessible<<Self as RustToCuda>::CudaRepresentation>>,
        ) -> Result<O, E>,
    >(
        &self,
        inner: F,
    ) -> Result<O, E>;

    /// Moves `self` to CUDA iff `self` is [`SafeDeviceCopy`]
    ///
    /// # Errors
    ///
    /// Returns a [`CudaError`] iff an error occurs inside CUDA
    fn move_to_cuda<
        O,
        E: From<CudaError>,
        F: FnOnce(
            HostAndDeviceOwned<DeviceAccessible<<Self as RustToCuda>::CudaRepresentation>>,
        ) -> Result<O, E>,
    >(
        self,
        inner: F,
    ) -> Result<O, E>
    where
        Self: RustToCuda<CudaRepresentation: SafeDeviceCopy, CudaAllocation: EmptyCudaAlloc>;
}

#[cfg(feature = "host")]
impl<T: RustToCuda + NoSafeAliasing> LendToCuda for T {
    fn lend_to_cuda<
        O,
        E: From<CudaError>,
        F: FnOnce(
            HostAndDeviceConstRef<DeviceAccessible<<Self as RustToCuda>::CudaRepresentation>>,
        ) -> Result<O, E>,
    >(
        &self,
        inner: F,
    ) -> Result<O, E> {
        let (cuda_repr, alloc) = unsafe { self.borrow(NoCudaAlloc) }?;

        let result = HostAndDeviceConstRef::with_new(&cuda_repr, inner);

        core::mem::drop(cuda_repr);
        core::mem::drop(alloc);

        result
    }

    fn move_to_cuda<
        O,
        E: From<CudaError>,
        F: FnOnce(
            HostAndDeviceOwned<DeviceAccessible<<Self as RustToCuda>::CudaRepresentation>>,
        ) -> Result<O, E>,
    >(
        self,
        inner: F,
    ) -> Result<O, E>
    where
        Self: RustToCuda<CudaRepresentation: SafeDeviceCopy, CudaAllocation: EmptyCudaAlloc>,
    {
        let (cuda_repr, alloc) = unsafe { self.borrow(NoCudaAlloc) }?;

        let result = HostAndDeviceOwned::with_new(cuda_repr, inner);

        core::mem::drop(alloc);

        result
    }
}

#[cfg(feature = "device")]
pub trait BorrowFromRust: RustToCuda + NoSafeAliasing {
    /// # Safety
    ///
    /// This function is only safe to call iff `cuda_repr` is the
    ///  [`DeviceConstRef`] borrowed on the CPU using the corresponding
    ///  [`LendToCuda::lend_to_cuda`].
    unsafe fn with_borrow_from_rust<O, F: FnOnce(&Self) -> O>(
        cuda_repr: DeviceConstRef<DeviceAccessible<<Self as RustToCuda>::CudaRepresentation>>,
        inner: F,
    ) -> O;

    /// # Safety
    ///
    /// This function is only safe to call iff `cuda_repr` is the
    ///  [`DeviceOwnedRef`] borrowed on the CPU using the corresponding
    ///  [`LendToCuda::move_to_cuda`].
    unsafe fn with_moved_from_rust<O, F: FnOnce(Self) -> O>(
        cuda_repr: DeviceOwnedRef<DeviceAccessible<<Self as RustToCuda>::CudaRepresentation>>,
        inner: F,
    ) -> O
    where
        Self: Sized,
        <Self as RustToCuda>::CudaRepresentation: SafeDeviceCopy;
}

#[cfg(feature = "device")]
impl<T: RustToCuda + NoSafeAliasing> BorrowFromRust for T {
    #[inline]
    unsafe fn with_borrow_from_rust<O, F: FnOnce(&Self) -> O>(
        cuda_repr: DeviceConstRef<DeviceAccessible<<Self as RustToCuda>::CudaRepresentation>>,
        inner: F,
    ) -> O {
        // `rust_repr` must never be dropped as we do NOT own any of the
        //  heap memory it might reference
        let rust_repr = core::mem::ManuallyDrop::new(CudaAsRust::as_rust(cuda_repr.as_ref()));

        inner(&rust_repr)
    }

    #[inline]
    unsafe fn with_moved_from_rust<O, F: FnOnce(Self) -> O>(
        mut cuda_repr: DeviceOwnedRef<DeviceAccessible<<Self as RustToCuda>::CudaRepresentation>>,
        inner: F,
    ) -> O
    where
        Self: Sized,
        <Self as RustToCuda>::CudaRepresentation: SafeDeviceCopy,
    {
        inner(CudaAsRust::as_rust(cuda_repr.as_mut()))
    }
}
