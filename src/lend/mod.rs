use const_type_layout::TypeGraphLayout;
#[cfg(feature = "host")]
use rustacuda::error::CudaError;

#[cfg(feature = "derive")]
#[allow(clippy::module_name_repetitions)]
pub use rust_cuda_derive::LendRustToCuda;

#[cfg(any(feature = "host", feature = "device", doc))]
use crate::safety::{SafeMutableAliasing, StackOnly};
#[cfg(feature = "device")]
use crate::utils::ffi::{DeviceConstRef, DeviceMutRef, DeviceOwnedRef};
use crate::{alloc::CudaAlloc, safety::PortableBitSemantics};
#[cfg(any(feature = "host", feature = "device"))]
use crate::{alloc::EmptyCudaAlloc, utils::ffi::DeviceAccessible};
#[cfg(feature = "host")]
use crate::{
    alloc::{CombinedCudaAlloc, NoCudaAlloc},
    host::{HostAndDeviceConstRef, HostAndDeviceMutRef, HostAndDeviceOwned},
    utils::r#async::{Async, CompletionFnMut, NoCompletion},
};

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
    type CudaAllocationAsync: CudaAlloc;

    #[doc(hidden)]
    #[cfg(feature = "host")]
    /// # Errors
    ///
    /// Returns a [`rustacuda::error::CudaError`] iff an error occurs inside
    /// CUDA
    ///
    /// # Safety
    ///
    /// This is an internal function and should NEVER be called manually.
    ///
    /// The returned
    /// [`Self::CudaRepresentation`](RustToCuda::CudaRepresentation) must NEVER
    /// be accessed on the  CPU  as it contains a GPU-resident copy of
    /// `self`.
    ///
    /// Since this method may perform asynchronous computation but returns its
    /// result immediately, this result must only be used to construct compound
    /// asynchronous computations before it has been synchronized on.
    ///
    /// Similarly, `&self` should remain borrowed until synchronisation has
    /// been performed.
    #[allow(clippy::type_complexity)]
    unsafe fn borrow_async<'stream, A: CudaAlloc>(
        &self,
        alloc: A,
        stream: &'stream rustacuda::stream::Stream,
    ) -> rustacuda::error::CudaResult<(
        Async<'_, 'stream, DeviceAccessible<Self::CudaRepresentation>>,
        CombinedCudaAlloc<Self::CudaAllocationAsync, A>,
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
    /// This is an internal function and should NEVER be called manually.
    ///
    /// Since this method may perform asynchronous computation but returns
    /// immediately, `&mut self` not be used until it has been synchronized on.
    ///
    /// Therefore, `&mut self` should remain mutably borrowed until
    /// synchronisation has been performed.
    #[allow(clippy::type_complexity)]
    unsafe fn restore_async<'a, 'stream, A: CudaAlloc, O>(
        this: owning_ref::BoxRefMut<'a, O, Self>,
        alloc: CombinedCudaAlloc<Self::CudaAllocationAsync, A>,
        stream: &'stream rustacuda::stream::Stream,
    ) -> rustacuda::error::CudaResult<(
        Async<'a, 'stream, owning_ref::BoxRefMut<'a, O, Self>, CompletionFnMut<'a, Self>>,
        A,
    )>;
}

/// # Safety
///
/// This is an internal trait and should NEVER be implemented manually
pub unsafe trait CudaAsRust: PortableBitSemantics + TypeGraphLayout {
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

pub trait RustToCudaAsyncProxy<T>: RustToCudaAsync + RustToCudaProxy<T> {}

impl<T, P: RustToCudaAsync + RustToCudaProxy<T>> RustToCudaAsyncProxy<T> for P {}

#[cfg(feature = "host")]
#[allow(clippy::module_name_repetitions)]
pub trait LendToCuda: RustToCuda {
    /// Lends an immutable borrow of `&self` to CUDA:
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
    ) -> Result<O, E>
    where
        Self: Sync;

    /// Lends a mutable borrow of `&mut self` to CUDA iff `Self` is
    /// [`SafeMutableAliasing`]:
    /// - code in the CUDA kernel can only access `&mut self` through the
    ///   `DeviceMutRef` inside the closure
    /// - after the closure, `&mut self` will reflect the changes from the
    ///   kernel execution
    ///
    /// # Errors
    ///
    /// Returns a `rustacuda::errors::CudaError` iff an error occurs inside CUDA
    fn lend_to_cuda_mut<
        O,
        E: From<CudaError>,
        F: FnOnce(
            HostAndDeviceMutRef<DeviceAccessible<<Self as RustToCuda>::CudaRepresentation>>,
        ) -> Result<O, E>,
    >(
        &mut self,
        inner: F,
    ) -> Result<O, E>
    where
        Self: Sync + SafeMutableAliasing;

    /// Moves `self` to CUDA iff `Self` is [`StackOnly`].
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
        Self: Send + RustToCuda<CudaRepresentation: StackOnly, CudaAllocation: EmptyCudaAlloc>;
}

#[cfg(feature = "host")]
impl<T: RustToCuda> LendToCuda for T {
    fn lend_to_cuda<
        O,
        E: From<CudaError>,
        F: FnOnce(
            HostAndDeviceConstRef<DeviceAccessible<<Self as RustToCuda>::CudaRepresentation>>,
        ) -> Result<O, E>,
    >(
        &self,
        inner: F,
    ) -> Result<O, E>
    where
        Self: Sync,
    {
        let (cuda_repr, alloc) = unsafe { self.borrow(NoCudaAlloc) }?;

        let result = HostAndDeviceConstRef::with_new(&cuda_repr, inner);

        core::mem::drop(cuda_repr);
        core::mem::drop(alloc);

        result
    }

    fn lend_to_cuda_mut<
        O,
        E: From<CudaError>,
        F: FnOnce(
            HostAndDeviceMutRef<DeviceAccessible<<Self as RustToCuda>::CudaRepresentation>>,
        ) -> Result<O, E>,
    >(
        &mut self,
        inner: F,
    ) -> Result<O, E>
    where
        Self: Sync + SafeMutableAliasing,
    {
        let (mut cuda_repr, alloc) = unsafe { self.borrow(NoCudaAlloc) }?;

        let result = HostAndDeviceMutRef::with_new(&mut cuda_repr, inner);

        core::mem::drop(cuda_repr);

        let _: NoCudaAlloc = unsafe { self.restore(alloc) }?;

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
        Self: Send + RustToCuda<CudaRepresentation: StackOnly, CudaAllocation: EmptyCudaAlloc>,
    {
        let (cuda_repr, alloc) = unsafe { self.borrow(NoCudaAlloc) }?;

        let result = HostAndDeviceOwned::with_new(cuda_repr, inner);

        core::mem::drop(alloc);

        result
    }
}

#[cfg(feature = "host")]
#[allow(clippy::module_name_repetitions)]
pub trait LendToCudaAsync: RustToCudaAsync {
    /// Lends an immutable copy of `&self` to CUDA:
    /// - code in the CUDA kernel can only access `&self` through the
    ///   [`DeviceConstRef`] inside the closure
    /// - after the closure, `&self` will not have changed, i.e. interior
    ///   mutability is not handled by this method
    ///
    /// Since the [`HostAndDeviceConstRef`] is wrapped in an [`Async`] with
    /// [`NoCompletion`], this [`Async`] can be safely dropped or forgotten
    /// without changing any behaviour. Therefore, this [`Async`] does *not*
    /// need to be returned from the `inner` closure.
    ///
    /// # Errors
    ///
    /// Returns a [`CudaError`] iff an error occurs inside CUDA
    fn lend_to_cuda_async<
        'stream,
        O,
        E: From<CudaError>,
        F: FnOnce(
            Async<
                '_,
                'stream,
                HostAndDeviceConstRef<DeviceAccessible<<Self as RustToCuda>::CudaRepresentation>>,
                NoCompletion,
            >,
        ) -> Result<O, E>,
    >(
        &self,
        stream: &'stream rustacuda::stream::Stream,
        inner: F,
    ) -> Result<O, E>
    where
        Self: Sync;

    #[allow(clippy::type_complexity)]
    /// Lends a mutable borrow of `&mut self` to CUDA iff `Self` is
    /// [`SafeMutableAliasing`]:
    /// - code in the CUDA kernel can only access `&mut self` through the
    ///   `DeviceMutRef` inside the closure
    /// - after the closure, `&mut self` will reflect the changes from the
    ///   kernel execution
    ///
    /// # Errors
    ///
    /// Returns a `rustacuda::errors::CudaError` iff an error occurs inside CUDA
    fn lend_to_cuda_mut_async<
        'a,
        'stream,
        O,
        E: From<CudaError>,
        F: for<'b> FnOnce(
            Async<
                'b,
                'stream,
                HostAndDeviceMutRef<DeviceAccessible<<Self as RustToCuda>::CudaRepresentation>>,
                NoCompletion,
            >,
        ) -> Result<O, E>,
        T: 'a,
    >(
        this: owning_ref::BoxRefMut<'a, T, Self>,
        stream: &'stream rustacuda::stream::Stream,
        inner: F,
    ) -> Result<
        (
            Async<'a, 'stream, owning_ref::BoxRefMut<'a, T, Self>, CompletionFnMut<'a, Self>>,
            O,
        ),
        E,
    >
    where
        Self: Sync + SafeMutableAliasing;

    /// Moves `self` to CUDA iff `self` is [`StackOnly`].
    ///
    /// Since the [`HostAndDeviceOwned`] is wrapped in an [`Async`] with
    /// [`NoCompletion`], this [`Async`] can be safely dropped or forgotten
    /// without changing any behaviour. Therefore, this [`Async`] does *not*
    /// need to be returned from the `inner` closure.
    ///
    /// # Errors
    ///
    /// Returns a [`CudaError`] iff an error occurs inside CUDA
    fn move_to_cuda_async<
        'stream,
        O,
        E: From<CudaError>,
        F: for<'a> FnOnce(
            Async<
                'a,
                'stream,
                HostAndDeviceOwned<DeviceAccessible<<Self as RustToCuda>::CudaRepresentation>>,
                NoCompletion,
            >,
        ) -> Result<O, E>,
    >(
        self,
        stream: &'stream rustacuda::stream::Stream,
        inner: F,
    ) -> Result<O, E>
    where
        Self: Send + RustToCuda<CudaRepresentation: StackOnly, CudaAllocation: EmptyCudaAlloc>;
}

#[cfg(feature = "host")]
impl<T: RustToCudaAsync> LendToCudaAsync for T {
    fn lend_to_cuda_async<
        'stream,
        O,
        E: From<CudaError>,
        F: FnOnce(
            Async<
                '_,
                'stream,
                HostAndDeviceConstRef<DeviceAccessible<<Self as RustToCuda>::CudaRepresentation>>,
                NoCompletion,
            >,
        ) -> Result<O, E>,
    >(
        &self,
        stream: &'stream rustacuda::stream::Stream,
        inner: F,
    ) -> Result<O, E>
    where
        Self: Sync,
    {
        let (cuda_repr, alloc) = unsafe { self.borrow_async(NoCudaAlloc, stream) }?;

        let (cuda_repr, capture_on_completion) = unsafe { cuda_repr.unwrap_unchecked()? };

        let result = HostAndDeviceConstRef::with_new(&cuda_repr, |const_ref| {
            let r#async = if matches!(capture_on_completion, Some(NoCompletion)) {
                Async::pending(const_ref, stream, NoCompletion)?
            } else {
                Async::ready(const_ref, stream)
            };

            inner(r#async)
        });

        core::mem::drop(cuda_repr);
        core::mem::drop(alloc);

        result
    }

    fn lend_to_cuda_mut_async<
        'a,
        'stream,
        O,
        E: From<CudaError>,
        F: for<'b> FnOnce(
            Async<
                'b,
                'stream,
                HostAndDeviceMutRef<DeviceAccessible<<Self as RustToCuda>::CudaRepresentation>>,
                NoCompletion,
            >,
        ) -> Result<O, E>,
        S: 'a,
    >(
        this: owning_ref::BoxRefMut<'a, S, Self>,
        stream: &'stream rustacuda::stream::Stream,
        inner: F,
    ) -> Result<
        (
            Async<'a, 'stream, owning_ref::BoxRefMut<'a, S, Self>, CompletionFnMut<'a, Self>>,
            O,
        ),
        E,
    >
    where
        Self: Sync + SafeMutableAliasing,
    {
        let (cuda_repr, alloc) = unsafe { this.borrow_async(NoCudaAlloc, stream) }?;

        let (mut cuda_repr, capture_on_completion) = unsafe { cuda_repr.unwrap_unchecked()? };

        let result = HostAndDeviceMutRef::with_new(&mut cuda_repr, |mut_ref| {
            let r#async = if matches!(capture_on_completion, Some(NoCompletion)) {
                Async::pending(mut_ref, stream, NoCompletion)?
            } else {
                Async::ready(mut_ref, stream)
            };

            inner(r#async)
        });

        core::mem::drop(cuda_repr);

        let (r#async, _): (_, NoCudaAlloc) = unsafe { Self::restore_async(this, alloc, stream) }?;

        result.map(|ok| (r#async, ok))
    }

    fn move_to_cuda_async<
        'stream,
        O,
        E: From<CudaError>,
        F: for<'a> FnOnce(
            Async<
                'a,
                'stream,
                HostAndDeviceOwned<DeviceAccessible<<Self as RustToCuda>::CudaRepresentation>>,
                NoCompletion,
            >,
        ) -> Result<O, E>,
    >(
        self,
        stream: &'stream rustacuda::stream::Stream,
        inner: F,
    ) -> Result<O, E>
    where
        Self: Send + RustToCuda<CudaRepresentation: StackOnly, CudaAllocation: EmptyCudaAlloc>,
    {
        let (cuda_repr, alloc) = unsafe { self.borrow_async(NoCudaAlloc, stream) }?;

        let (cuda_repr, capture_on_completion) = unsafe { cuda_repr.unwrap_unchecked()? };

        let result = HostAndDeviceOwned::with_new(cuda_repr, |owned_ref| {
            if matches!(capture_on_completion, Some(NoCompletion)) {
                inner(Async::pending(owned_ref, stream, NoCompletion)?)
            } else {
                inner(Async::ready(owned_ref, stream))
            }
        });

        core::mem::drop(alloc);

        result
    }
}

#[cfg(feature = "device")]
pub trait BorrowFromRust: RustToCuda {
    /// # Safety
    ///
    /// This function is only safe to call iff `cuda_repr` is the
    /// [`DeviceConstRef`] borrowed on the CPU using the corresponding
    /// [`LendToCuda::lend_to_cuda`].
    unsafe fn with_borrow_from_rust<O, F: FnOnce(&Self) -> O>(
        cuda_repr: DeviceConstRef<DeviceAccessible<<Self as RustToCuda>::CudaRepresentation>>,
        inner: F,
    ) -> O;

    /// # Safety
    ///
    /// This function is only safe to call iff `cuda_repr_mut` is the
    /// [`DeviceMutRef`] borrowed on the CPU using the corresponding
    /// [`LendToCuda::lend_to_cuda_mut`].
    unsafe fn with_borrow_from_rust_mut<O, F: FnOnce(&mut Self) -> O>(
        cuda_repr_mut: DeviceMutRef<DeviceAccessible<<Self as RustToCuda>::CudaRepresentation>>,
        inner: F,
    ) -> O
    where
        Self: SafeMutableAliasing;

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
        Self: Sized + RustToCuda<CudaRepresentation: StackOnly, CudaAllocation: EmptyCudaAlloc>;
}

#[cfg(feature = "device")]
impl<T: RustToCuda> BorrowFromRust for T {
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
    unsafe fn with_borrow_from_rust_mut<O, F: FnOnce(&mut Self) -> O>(
        mut cuda_repr_mut: DeviceMutRef<DeviceAccessible<<Self as RustToCuda>::CudaRepresentation>>,
        inner: F,
    ) -> O
    where
        Self: SafeMutableAliasing,
    {
        // `rust_repr` must never be dropped as we do NOT own any of the
        //  heap memory it might reference
        let mut rust_repr_mut =
            core::mem::ManuallyDrop::new(CudaAsRust::as_rust(cuda_repr_mut.as_mut()));

        inner(&mut rust_repr_mut)
    }

    #[inline]
    unsafe fn with_moved_from_rust<O, F: FnOnce(Self) -> O>(
        mut cuda_repr: DeviceOwnedRef<DeviceAccessible<<Self as RustToCuda>::CudaRepresentation>>,
        inner: F,
    ) -> O
    where
        Self: RustToCuda<CudaRepresentation: StackOnly, CudaAllocation: EmptyCudaAlloc>,
    {
        inner(CudaAsRust::as_rust(cuda_repr.as_mut()))
    }
}
