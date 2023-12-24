#[cfg(any(not(feature = "host"), doc))]
use core::convert::{AsMut, AsRef};
use core::{
    marker::PhantomData,
    ops::{Deref, DerefMut},
};

#[cfg(feature = "host")]
use alloc::fmt;
#[cfg(feature = "host")]
use core::{
    mem::MaybeUninit,
    ptr::{copy_nonoverlapping, NonNull},
};

use const_type_layout::TypeGraphLayout;
use rustacuda_core::DeviceCopy;

#[cfg(feature = "derive")]
#[doc(cfg(feature = "derive"))]
pub use rust_cuda_derive::LendRustToCuda;

#[cfg(feature = "derive")]
#[doc(cfg(feature = "derive"))]
pub use rust_cuda_derive::kernel;

#[cfg(feature = "host")]
use crate::{safety::SafeDeviceCopy, utils::device_copy::SafeDeviceCopyWrapper};

#[repr(transparent)]
#[cfg_attr(not(feature = "host"), derive(Debug))]
#[derive(TypeLayout)]
pub struct DeviceAccessible<T: ?Sized + DeviceCopy>(T);

unsafe impl<T: ?Sized + DeviceCopy> DeviceCopy for DeviceAccessible<T> {}

#[cfg(feature = "host")]
impl<T: CudaAsRust> From<T> for DeviceAccessible<T> {
    fn from(value: T) -> Self {
        Self(value)
    }
}

#[cfg(feature = "host")]
impl<T: SafeDeviceCopy + TypeGraphLayout> From<&T> for DeviceAccessible<SafeDeviceCopyWrapper<T>> {
    fn from(value: &T) -> Self {
        let value = unsafe {
            let mut uninit = MaybeUninit::uninit();
            copy_nonoverlapping(value, uninit.as_mut_ptr(), 1);
            uninit.assume_init()
        };

        Self(SafeDeviceCopyWrapper::from(value))
    }
}

#[cfg(feature = "host")]
impl<T: ?Sized + DeviceCopy + fmt::Debug> fmt::Debug for DeviceAccessible<T> {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        fmt.debug_struct(stringify!(DeviceAccessible))
            .finish_non_exhaustive()
    }
}

#[cfg(not(feature = "host"))]
impl<T: ?Sized + DeviceCopy> Deref for DeviceAccessible<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[cfg(not(feature = "host"))]
impl<T: ?Sized + DeviceCopy> DerefMut for DeviceAccessible<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

/// # Safety
///
/// This is an internal trait and should ONLY be derived automatically using
/// `#[derive(LendRustToCuda)]`
pub unsafe trait RustToCuda {
    type CudaAllocation: CudaAlloc;
    type CudaRepresentation: CudaAsRust<RustRepresentation = Self> + TypeGraphLayout;

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
    #[cfg(not(feature = "host"))]
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

#[repr(transparent)]
#[derive(Clone, Copy, TypeLayout)]
pub struct DeviceConstRef<'r, T: DeviceCopy + 'r> {
    #[cfg_attr(feature = "host", allow(dead_code))]
    pub(super) pointer: *const T,
    pub(super) reference: PhantomData<&'r T>,
}

unsafe impl<'r, T: DeviceCopy> DeviceCopy for DeviceConstRef<'r, T> {}

#[cfg(any(not(feature = "host"), doc))]
#[doc(cfg(not(feature = "host")))]
impl<'r, T: DeviceCopy> AsRef<T> for DeviceConstRef<'r, T> {
    fn as_ref(&self) -> &T {
        unsafe { &*self.pointer }
    }
}

#[repr(transparent)]
#[derive(TypeLayout)]
pub struct DeviceMutRef<'r, T: DeviceCopy + 'r> {
    #[cfg_attr(feature = "host", allow(dead_code))]
    pub(super) pointer: *mut T,
    pub(super) reference: PhantomData<&'r mut T>,
}

unsafe impl<'r, T: DeviceCopy> DeviceCopy for DeviceMutRef<'r, T> {}

#[cfg(any(not(feature = "host"), doc))]
#[doc(cfg(not(feature = "host")))]
impl<'r, T: DeviceCopy> AsRef<T> for DeviceMutRef<'r, T> {
    fn as_ref(&self) -> &T {
        unsafe { &*self.pointer }
    }
}

#[cfg(any(not(feature = "host"), doc))]
#[doc(cfg(not(feature = "host")))]
impl<'r, T: DeviceCopy> AsMut<T> for DeviceMutRef<'r, T> {
    fn as_mut(&mut self) -> &mut T {
        unsafe { &mut *self.pointer }
    }
}

#[repr(transparent)]
#[derive(TypeLayout)]
pub struct DeviceOwnedRef<'r, T: DeviceCopy> {
    #[cfg_attr(feature = "host", allow(dead_code))]
    pub(super) pointer: *mut T,
    pub(super) reference: PhantomData<&'r mut ()>,
    pub(super) marker: PhantomData<T>,
}

unsafe impl<'r, T: DeviceCopy> DeviceCopy for DeviceOwnedRef<'r, T> {}

#[cfg(any(not(feature = "host"), doc))]
#[doc(cfg(not(feature = "host")))]
impl<'r, T: DeviceCopy> AsRef<T> for DeviceOwnedRef<'r, T> {
    fn as_ref(&self) -> &T {
        unsafe { &*self.pointer }
    }
}

#[cfg(any(not(feature = "host"), doc))]
#[doc(cfg(not(feature = "host")))]
impl<'r, T: DeviceCopy> AsMut<T> for DeviceOwnedRef<'r, T> {
    fn as_mut(&mut self) -> &mut T {
        unsafe { &mut *self.pointer }
    }
}

pub(crate) mod crate_private {
    pub mod alloc {
        pub trait Sealed {}
    }
}

mod private {
    pub mod empty {
        pub trait Sealed {}
    }
}

pub trait EmptyCudaAlloc: private::empty::Sealed {}

pub trait CudaAlloc: crate_private::alloc::Sealed {}

impl<T: CudaAlloc> CudaAlloc for Option<T> {}
impl<T: CudaAlloc> crate_private::alloc::Sealed for Option<T> {}

pub struct NoCudaAlloc;
impl CudaAlloc for NoCudaAlloc {}
impl crate_private::alloc::Sealed for NoCudaAlloc {}
impl EmptyCudaAlloc for NoCudaAlloc {}
impl private::empty::Sealed for NoCudaAlloc {}

pub struct SomeCudaAlloc(());
impl CudaAlloc for SomeCudaAlloc {}
impl crate_private::alloc::Sealed for SomeCudaAlloc {}
impl !EmptyCudaAlloc for SomeCudaAlloc {}
impl !private::empty::Sealed for SomeCudaAlloc {}

pub struct CombinedCudaAlloc<A: CudaAlloc, B: CudaAlloc>(A, B);
impl<A: CudaAlloc, B: CudaAlloc> CudaAlloc for CombinedCudaAlloc<A, B> {}
impl<A: CudaAlloc, B: CudaAlloc> crate_private::alloc::Sealed for CombinedCudaAlloc<A, B> {}
impl<A: CudaAlloc + EmptyCudaAlloc, B: CudaAlloc + EmptyCudaAlloc> EmptyCudaAlloc
    for CombinedCudaAlloc<A, B>
{
}
impl<A: CudaAlloc + EmptyCudaAlloc, B: CudaAlloc + EmptyCudaAlloc> private::empty::Sealed
    for CombinedCudaAlloc<A, B>
{
}
impl<A: CudaAlloc, B: CudaAlloc> CombinedCudaAlloc<A, B> {
    #[must_use]
    pub const fn new(front: A, tail: B) -> Self {
        Self(front, tail)
    }

    pub fn split(self) -> (A, B) {
        (self.0, self.1)
    }
}

mod sealed {
    #[doc(hidden)]
    pub trait Sealed {}
}

// TODO: doc cfg
pub trait CudaKernelParameter: sealed::Sealed {
    #[cfg(feature = "host")]
    type SyncHostType;
    #[cfg(feature = "host")]
    type AsyncHostType<'stream, 'b>;
    #[doc(hidden)]
    type FfiType<'stream, 'b>: rustacuda_core::DeviceCopy + TypeGraphLayout;
    #[cfg(all(not(feature = "host"), target_os = "cuda"))]
    type DeviceType<'b>;

    #[cfg(feature = "host")]
    #[allow(clippy::missing_errors_doc)] // FIXME
    fn with_new_async<'stream, O, E: From<rustacuda::error::CudaError>>(
        param: Self::SyncHostType,
        stream: &'stream rustacuda::stream::Stream,
        inner: impl for<'b> FnOnce(Self::AsyncHostType<'stream, 'b>) -> Result<O, E>,
    ) -> Result<O, E>;

    #[doc(hidden)]
    #[cfg(feature = "host")]
    fn with_async_as_ptx_jit<O>(
        param: &Self::AsyncHostType<'_, '_>,
        inner: impl for<'p> FnOnce(Option<&'p NonNull<[u8]>>) -> O,
    ) -> O;

    #[doc(hidden)]
    #[cfg(feature = "host")]
    fn async_to_ffi<'stream, 'b>(
        param: Self::AsyncHostType<'stream, 'b>,
    ) -> Self::FfiType<'stream, 'b>;

    #[doc(hidden)]
    #[cfg(all(not(feature = "host"), target_os = "cuda"))]
    fn with_ffi_as_device<O, const PARAM: usize>(
        param: Self::FfiType<'static, 'static>,
        inner: impl for<'b> FnOnce(Self::DeviceType<'b>) -> O,
    ) -> O;
}

pub struct PtxJit<T> {
    never: !,
    _marker: PhantomData<T>,
}

impl<T> Deref for PtxJit<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.never
    }
}

impl<T> DerefMut for PtxJit<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.never
    }
}

pub struct PerThreadShallowCopy<
    T: crate::safety::SafeDeviceCopy
        + crate::safety::NoSafeAliasing
        + const_type_layout::TypeGraphLayout,
> {
    never: !,
    _marker: PhantomData<T>,
}

impl<
        T: crate::safety::SafeDeviceCopy
            + crate::safety::NoSafeAliasing
            + const_type_layout::TypeGraphLayout,
    > Deref for PerThreadShallowCopy<T>
{
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.never
    }
}

impl<
        T: crate::safety::SafeDeviceCopy
            + crate::safety::NoSafeAliasing
            + const_type_layout::TypeGraphLayout,
    > DerefMut for PerThreadShallowCopy<T>
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.never
    }
}

impl<
        T: crate::safety::SafeDeviceCopy
            + crate::safety::NoSafeAliasing
            + const_type_layout::TypeGraphLayout,
    > CudaKernelParameter for PerThreadShallowCopy<T>
{
    #[cfg(feature = "host")]
    type AsyncHostType<'stream, 'b> = crate::utils::device_copy::SafeDeviceCopyWrapper<T>;
    #[cfg(all(not(feature = "host"), target_os = "cuda"))]
    type DeviceType<'b> = T;
    type FfiType<'stream, 'b> = crate::utils::device_copy::SafeDeviceCopyWrapper<T>;
    #[cfg(feature = "host")]
    type SyncHostType = T;

    #[cfg(feature = "host")]
    fn with_new_async<'stream, O, E: From<rustacuda::error::CudaError>>(
        param: Self::SyncHostType,
        _stream: &'stream rustacuda::stream::Stream,
        inner: impl for<'b> FnOnce(Self::AsyncHostType<'stream, 'b>) -> Result<O, E>,
    ) -> Result<O, E> {
        inner(crate::utils::device_copy::SafeDeviceCopyWrapper::from(
            param,
        ))
    }

    #[cfg(feature = "host")]
    fn with_async_as_ptx_jit<O>(
        _param: &Self::AsyncHostType<'_, '_>,
        inner: impl for<'p> FnOnce(Option<&'p NonNull<[u8]>>) -> O,
    ) -> O {
        inner(None)
    }

    #[cfg(feature = "host")]
    fn async_to_ffi<'stream, 'b>(
        param: Self::AsyncHostType<'stream, 'b>,
    ) -> Self::FfiType<'stream, 'b> {
        param
    }

    #[cfg(all(not(feature = "host"), target_os = "cuda"))]
    fn with_ffi_as_device<O, const PARAM: usize>(
        param: Self::FfiType<'static, 'static>,
        inner: impl for<'b> FnOnce(Self::DeviceType<'b>) -> O,
    ) -> O {
        let param = param.into_inner();

        inner(param)
    }
}
impl<
        T: crate::safety::SafeDeviceCopy
            + crate::safety::NoSafeAliasing
            + const_type_layout::TypeGraphLayout,
    > sealed::Sealed for PerThreadShallowCopy<T>
{
}

impl<
        'a,
        T: 'static
            + crate::safety::SafeDeviceCopy
            + crate::safety::NoSafeAliasing
            + const_type_layout::TypeGraphLayout,
    > CudaKernelParameter for &'a PerThreadShallowCopy<T>
{
    #[cfg(feature = "host")]
    type AsyncHostType<'stream, 'b> = crate::host::HostAndDeviceConstRefAsync<
        'stream,
        'b,
        crate::utils::device_copy::SafeDeviceCopyWrapper<T>,
    >;
    #[cfg(all(not(feature = "host"), target_os = "cuda"))]
    type DeviceType<'b> = &'b T;
    type FfiType<'stream, 'b> =
        DeviceConstRef<'b, crate::utils::device_copy::SafeDeviceCopyWrapper<T>>;
    #[cfg(feature = "host")]
    type SyncHostType = &'a T;

    #[cfg(feature = "host")]
    fn with_new_async<'stream, O, E: From<rustacuda::error::CudaError>>(
        param: Self::SyncHostType,
        _stream: &'stream rustacuda::stream::Stream,
        inner: impl for<'b> FnOnce(Self::AsyncHostType<'stream, 'b>) -> Result<O, E>,
    ) -> Result<O, E> {
        let host_box = crate::host::HostDeviceBox::from(rustacuda::memory::DeviceBox::new(
            crate::utils::device_copy::SafeDeviceCopyWrapper::from_ref(param),
        )?);

        // Safety: `host_box` contains exactly the device copy of `param`
        let const_ref = unsafe {
            crate::host::HostAndDeviceConstRef::new(
                &host_box,
                crate::utils::device_copy::SafeDeviceCopyWrapper::from_ref(param),
            )
        };

        inner(const_ref.as_async())
    }

    #[cfg(feature = "host")]
    fn with_async_as_ptx_jit<O>(
        _param: &Self::AsyncHostType<'_, '_>,
        inner: impl for<'p> FnOnce(Option<&'p NonNull<[u8]>>) -> O,
    ) -> O {
        inner(None)
    }

    #[cfg(feature = "host")]
    fn async_to_ffi<'stream, 'b>(
        param: Self::AsyncHostType<'stream, 'b>,
    ) -> Self::FfiType<'stream, 'b> {
        unsafe { param.for_device_async() }
    }

    #[cfg(all(not(feature = "host"), target_os = "cuda"))]
    fn with_ffi_as_device<O, const PARAM: usize>(
        param: Self::FfiType<'static, 'static>,
        inner: impl for<'b> FnOnce(Self::DeviceType<'b>) -> O,
    ) -> O {
        let param = param.as_ref().into_ref();

        inner(param)
    }
}
impl<
        'a,
        T: crate::safety::SafeDeviceCopy
            + crate::safety::NoSafeAliasing
            + const_type_layout::TypeGraphLayout,
    > sealed::Sealed for &'a PerThreadShallowCopy<T>
{
}

impl<
        'a,
        T: 'static
            + crate::safety::SafeDeviceCopy
            + crate::safety::NoSafeAliasing
            + const_type_layout::TypeGraphLayout,
    > CudaKernelParameter for &'a PtxJit<PerThreadShallowCopy<T>>
{
    #[cfg(feature = "host")]
    type AsyncHostType<'stream, 'b> =
        <&'a PerThreadShallowCopy<T> as CudaKernelParameter>::AsyncHostType<'stream, 'b>;
    #[cfg(all(not(feature = "host"), target_os = "cuda"))]
    type DeviceType<'b> = <&'a PerThreadShallowCopy<T> as CudaKernelParameter>::DeviceType<'b>;
    type FfiType<'stream, 'b> =
        <&'a PerThreadShallowCopy<T> as CudaKernelParameter>::FfiType<'stream, 'b>;
    #[cfg(feature = "host")]
    type SyncHostType = <&'a PerThreadShallowCopy<T> as CudaKernelParameter>::SyncHostType;

    #[cfg(feature = "host")]
    fn with_new_async<'stream, O, E: From<rustacuda::error::CudaError>>(
        param: Self::SyncHostType,
        stream: &'stream rustacuda::stream::Stream,
        inner: impl for<'b> FnOnce(Self::AsyncHostType<'stream, 'b>) -> Result<O, E>,
    ) -> Result<O, E> {
        <&'a PerThreadShallowCopy<T> as CudaKernelParameter>::with_new_async(param, stream, inner)
    }

    #[cfg(feature = "host")]
    fn with_async_as_ptx_jit<O>(
        param: &Self::AsyncHostType<'_, '_>,
        inner: impl for<'p> FnOnce(Option<&'p NonNull<[u8]>>) -> O,
    ) -> O {
        inner(Some(&param_as_raw_bytes(param.for_host())))
    }

    #[cfg(feature = "host")]
    fn async_to_ffi<'stream, 'b>(
        param: Self::AsyncHostType<'stream, 'b>,
    ) -> Self::FfiType<'stream, 'b> {
        <&'a PerThreadShallowCopy<T> as CudaKernelParameter>::async_to_ffi(param)
    }

    #[cfg(all(not(feature = "host"), target_os = "cuda"))]
    fn with_ffi_as_device<O, const PARAM: usize>(
        param: Self::FfiType<'static, 'static>,
        inner: impl for<'b> FnOnce(Self::DeviceType<'b>) -> O,
    ) -> O {
        emit_param_ptx_jit_marker::<_, PARAM>(param.as_ref());

        <&'a PerThreadShallowCopy<T> as CudaKernelParameter>::with_ffi_as_device::<O, PARAM>(
            param, inner,
        )
    }
}
impl<
        'a,
        T: crate::safety::SafeDeviceCopy
            + crate::safety::NoSafeAliasing
            + const_type_layout::TypeGraphLayout,
    > sealed::Sealed for &'a PtxJit<PerThreadShallowCopy<T>>
{
}

pub struct ShallowInteriorMutable<T: InteriorMutableSafeDeviceCopy> {
    never: !,
    _marker: PhantomData<T>,
}

impl<T: InteriorMutableSafeDeviceCopy> Deref for ShallowInteriorMutable<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.never
    }
}

impl<'a, T: 'static + InteriorMutableSafeDeviceCopy> CudaKernelParameter
    for &'a ShallowInteriorMutable<T>
{
    #[cfg(feature = "host")]
    type AsyncHostType<'stream, 'b> = crate::host::HostAndDeviceConstRefAsync<
        'stream,
        'b,
        crate::utils::device_copy::SafeDeviceCopyWrapper<T>,
    >;
    #[cfg(all(not(feature = "host"), target_os = "cuda"))]
    type DeviceType<'b> = &'b T;
    type FfiType<'stream, 'b> =
        DeviceConstRef<'b, crate::utils::device_copy::SafeDeviceCopyWrapper<T>>;
    #[cfg(feature = "host")]
    /// The kernel takes a mutable borrow of the interior mutable data to ensure
    /// the interior mutability is limited to just this kernel invocation.
    type SyncHostType = &'a mut T;

    #[cfg(feature = "host")]
    fn with_new_async<'stream, O, E: From<rustacuda::error::CudaError>>(
        param: Self::SyncHostType,
        _stream: &'stream rustacuda::stream::Stream,
        inner: impl for<'b> FnOnce(Self::AsyncHostType<'stream, 'b>) -> Result<O, E>,
    ) -> Result<O, E> {
        let host_box = crate::host::HostDeviceBox::from(rustacuda::memory::DeviceBox::new(
            crate::utils::device_copy::SafeDeviceCopyWrapper::from_ref(param),
        )?);

        // Safety: `host_box` contains exactly the device copy of `param`
        let const_ref = unsafe {
            crate::host::HostAndDeviceConstRef::new(
                &host_box,
                crate::utils::device_copy::SafeDeviceCopyWrapper::from_ref(param),
            )
        };

        let result = inner(const_ref.as_async());

        host_box.copy_to(crate::utils::device_copy::SafeDeviceCopyWrapper::from_mut(
            param,
        ))?;

        result
    }

    #[cfg(feature = "host")]
    fn with_async_as_ptx_jit<O>(
        _param: &Self::AsyncHostType<'_, '_>,
        inner: impl for<'p> FnOnce(Option<&'p NonNull<[u8]>>) -> O,
    ) -> O {
        inner(None)
    }

    #[cfg(feature = "host")]
    fn async_to_ffi<'stream, 'b>(
        param: Self::AsyncHostType<'stream, 'b>,
    ) -> Self::FfiType<'stream, 'b> {
        unsafe { param.for_device_async() }
    }

    #[cfg(all(not(feature = "host"), target_os = "cuda"))]
    fn with_ffi_as_device<O, const PARAM: usize>(
        param: Self::FfiType<'static, 'static>,
        inner: impl for<'b> FnOnce(Self::DeviceType<'b>) -> O,
    ) -> O {
        let param = param.as_ref().into_ref();

        inner(param)
    }
}
impl<'a, T: InteriorMutableSafeDeviceCopy> sealed::Sealed for &'a ShallowInteriorMutable<T> {}

pub trait InteriorMutableSafeDeviceCopy:
    crate::safety::SafeDeviceCopy
    + crate::safety::NoSafeAliasing
    + const_type_layout::TypeGraphLayout
    + sealed::Sealed
{
}

macro_rules! impl_atomic_interior_mutable {
    ($atomic:ident($interior:ty)) => {
        impl InteriorMutableSafeDeviceCopy for core::sync::atomic::$atomic {}
        impl sealed::Sealed for core::sync::atomic::$atomic {}
    };
    ($($atomic:ident($interior:ty)),*) => {
        $(impl_atomic_interior_mutable! { $atomic($interior) })*
    }
}

impl_atomic_interior_mutable! {
    AtomicBool(bool),
    AtomicI8(i8), AtomicI16(i16), AtomicI32(i32), AtomicI64(i64), AtomicIsize(isize),
    AtomicU8(u8), AtomicU16(u16), AtomicU32(u32), AtomicU64(u64), AtomicUsize(usize)
}

// TODO: update const type layout
// impl<T: crate::safety::SafeDeviceCopy + const_type_layout::TypeGraphLayout>
// InteriorMutableSafeDeviceCopy for core::cell::SyncUnsafeCell<T> {}
// impl<T: crate::safety::SafeDeviceCopy> sealed::Sealed for
// core::cell::SyncUnsafeCell<T> {}

pub struct SharedHeapPerThreadShallowCopy<T: RustToCuda + crate::safety::NoSafeAliasing> {
    never: !,
    _marker: PhantomData<T>,
}

impl<T: RustToCuda + crate::safety::NoSafeAliasing> Deref for SharedHeapPerThreadShallowCopy<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.never
    }
}

impl<T: RustToCuda + crate::safety::NoSafeAliasing> DerefMut for SharedHeapPerThreadShallowCopy<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.never
    }
}

impl<
        T: RustToCuda<
                CudaRepresentation: 'static + crate::safety::SafeDeviceCopy,
                CudaAllocation: EmptyCudaAlloc,
            > + crate::safety::NoSafeAliasing,
    > CudaKernelParameter for SharedHeapPerThreadShallowCopy<T>
{
    #[cfg(feature = "host")]
    type AsyncHostType<'stream, 'b> = crate::host::HostAndDeviceOwnedAsync<
        'stream,
        'b,
        DeviceAccessible<<T as RustToCuda>::CudaRepresentation>,
    >;
    #[cfg(all(not(feature = "host"), target_os = "cuda"))]
    type DeviceType<'b> = T;
    type FfiType<'stream, 'b> =
        DeviceOwnedRef<'b, DeviceAccessible<<T as RustToCuda>::CudaRepresentation>>;
    #[cfg(feature = "host")]
    type SyncHostType = T;

    #[cfg(feature = "host")]
    fn with_new_async<'stream, O, E: From<rustacuda::error::CudaError>>(
        param: Self::SyncHostType,
        _stream: &'stream rustacuda::stream::Stream,
        inner: impl for<'b> FnOnce(Self::AsyncHostType<'stream, 'b>) -> Result<O, E>,
    ) -> Result<O, E> {
        crate::host::LendToCuda::move_to_cuda(param, |param| inner(param.into_async()))
    }

    #[cfg(feature = "host")]
    fn with_async_as_ptx_jit<O>(
        _param: &Self::AsyncHostType<'_, '_>,
        inner: impl for<'p> FnOnce(Option<&'p NonNull<[u8]>>) -> O,
    ) -> O {
        inner(None)
    }

    #[cfg(feature = "host")]
    fn async_to_ffi<'stream, 'b>(
        param: Self::AsyncHostType<'stream, 'b>,
    ) -> Self::FfiType<'stream, 'b> {
        unsafe { param.for_device_async() }
    }

    #[cfg(all(not(feature = "host"), target_os = "cuda"))]
    fn with_ffi_as_device<O, const PARAM: usize>(
        param: Self::FfiType<'static, 'static>,
        inner: impl for<'b> FnOnce(Self::DeviceType<'b>) -> O,
    ) -> O {
        unsafe { crate::device::BorrowFromRust::with_moved_from_rust(param, inner) }
    }
}
impl<
        T: RustToCuda<
                CudaRepresentation: crate::safety::SafeDeviceCopy,
                CudaAllocation: EmptyCudaAlloc,
            > + crate::safety::NoSafeAliasing,
    > sealed::Sealed for SharedHeapPerThreadShallowCopy<T>
{
}

impl<'a, T: 'static + RustToCuda + crate::safety::NoSafeAliasing> CudaKernelParameter
    for &'a SharedHeapPerThreadShallowCopy<T>
{
    #[cfg(feature = "host")]
    type AsyncHostType<'stream, 'b> = crate::host::HostAndDeviceConstRefAsync<
        'stream,
        'b,
        DeviceAccessible<<T as RustToCuda>::CudaRepresentation>,
    >;
    #[cfg(all(not(feature = "host"), target_os = "cuda"))]
    type DeviceType<'b> = &'b T;
    type FfiType<'stream, 'b> =
        DeviceConstRef<'b, DeviceAccessible<<T as RustToCuda>::CudaRepresentation>>;
    #[cfg(feature = "host")]
    type SyncHostType = &'a T;

    #[cfg(feature = "host")]
    fn with_new_async<'stream, O, E: From<rustacuda::error::CudaError>>(
        param: Self::SyncHostType,
        _stream: &'stream rustacuda::stream::Stream,
        inner: impl for<'b> FnOnce(Self::AsyncHostType<'stream, 'b>) -> Result<O, E>,
    ) -> Result<O, E> {
        crate::host::LendToCuda::lend_to_cuda(param, |param| inner(param.as_async()))
    }

    #[cfg(feature = "host")]
    fn with_async_as_ptx_jit<O>(
        _param: &Self::AsyncHostType<'_, '_>,
        inner: impl for<'p> FnOnce(Option<&'p NonNull<[u8]>>) -> O,
    ) -> O {
        inner(None)
    }

    #[cfg(feature = "host")]
    fn async_to_ffi<'stream, 'b>(
        param: Self::AsyncHostType<'stream, 'b>,
    ) -> Self::FfiType<'stream, 'b> {
        unsafe { param.for_device_async() }
    }

    #[cfg(all(not(feature = "host"), target_os = "cuda"))]
    fn with_ffi_as_device<O, const PARAM: usize>(
        param: Self::FfiType<'static, 'static>,
        inner: impl for<'b> FnOnce(Self::DeviceType<'b>) -> O,
    ) -> O {
        unsafe { crate::device::BorrowFromRust::with_borrow_from_rust(param, inner) }
    }
}
impl<'a, T: RustToCuda + crate::safety::NoSafeAliasing> sealed::Sealed
    for &'a SharedHeapPerThreadShallowCopy<T>
{
}

impl<'a, T: 'static + RustToCuda + crate::safety::NoSafeAliasing> CudaKernelParameter
    for &'a mut SharedHeapPerThreadShallowCopy<T>
{
    #[cfg(feature = "host")]
    type AsyncHostType<'stream, 'b> = crate::host::HostAndDeviceMutRefAsync<
        'stream,
        'b,
        DeviceAccessible<<T as RustToCuda>::CudaRepresentation>,
    >;
    #[cfg(all(not(feature = "host"), target_os = "cuda"))]
    type DeviceType<'b> = &'b mut T;
    type FfiType<'stream, 'b> =
        DeviceMutRef<'b, DeviceAccessible<<T as RustToCuda>::CudaRepresentation>>;
    #[cfg(feature = "host")]
    type SyncHostType = &'a mut T;

    #[cfg(feature = "host")]
    fn with_new_async<'stream, O, E: From<rustacuda::error::CudaError>>(
        param: Self::SyncHostType,
        _stream: &'stream rustacuda::stream::Stream,
        inner: impl for<'b> FnOnce(Self::AsyncHostType<'stream, 'b>) -> Result<O, E>,
    ) -> Result<O, E> {
        crate::host::LendToCuda::lend_to_cuda_mut(param, |mut param| inner(param.as_async()))
    }

    #[cfg(feature = "host")]
    fn with_async_as_ptx_jit<O>(
        _param: &Self::AsyncHostType<'_, '_>,
        inner: impl for<'p> FnOnce(Option<&'p NonNull<[u8]>>) -> O,
    ) -> O {
        inner(None)
    }

    #[cfg(feature = "host")]
    fn async_to_ffi<'stream, 'b>(
        mut param: Self::AsyncHostType<'stream, 'b>,
    ) -> Self::FfiType<'stream, 'b> {
        unsafe { param.for_device_async() }
    }

    #[cfg(all(not(feature = "host"), target_os = "cuda"))]
    fn with_ffi_as_device<O, const PARAM: usize>(
        param: Self::FfiType<'static, 'static>,
        inner: impl for<'b> FnOnce(Self::DeviceType<'b>) -> O,
    ) -> O {
        unsafe { crate::device::BorrowFromRust::with_borrow_from_rust_mut(param, inner) }
    }
}
impl<'a, T: RustToCuda + crate::safety::NoSafeAliasing> sealed::Sealed
    for &'a mut SharedHeapPerThreadShallowCopy<T>
{
}

impl<
        T: RustToCuda<
                CudaRepresentation: 'static + crate::safety::SafeDeviceCopy,
                CudaAllocation: EmptyCudaAlloc,
            > + crate::safety::NoSafeAliasing,
    > CudaKernelParameter for PtxJit<SharedHeapPerThreadShallowCopy<T>>
{
    #[cfg(feature = "host")]
    type AsyncHostType<'stream, 'b> =
        <SharedHeapPerThreadShallowCopy<T> as CudaKernelParameter>::AsyncHostType<'stream, 'b>;
    #[cfg(all(not(feature = "host"), target_os = "cuda"))]
    type DeviceType<'b> =
        <SharedHeapPerThreadShallowCopy<T> as CudaKernelParameter>::DeviceType<'b>;
    type FfiType<'stream, 'b> =
        <SharedHeapPerThreadShallowCopy<T> as CudaKernelParameter>::FfiType<'stream, 'b>;
    #[cfg(feature = "host")]
    type SyncHostType = <SharedHeapPerThreadShallowCopy<T> as CudaKernelParameter>::SyncHostType;

    #[cfg(feature = "host")]
    fn with_new_async<'stream, O, E: From<rustacuda::error::CudaError>>(
        param: Self::SyncHostType,
        stream: &'stream rustacuda::stream::Stream,
        inner: impl for<'b> FnOnce(Self::AsyncHostType<'stream, 'b>) -> Result<O, E>,
    ) -> Result<O, E> {
        <SharedHeapPerThreadShallowCopy<T> as CudaKernelParameter>::with_new_async(
            param, stream, inner,
        )
    }

    #[cfg(feature = "host")]
    fn with_async_as_ptx_jit<O>(
        param: &Self::AsyncHostType<'_, '_>,
        inner: impl for<'p> FnOnce(Option<&'p NonNull<[u8]>>) -> O,
    ) -> O {
        inner(Some(&param_as_raw_bytes(param.for_host())))
    }

    #[cfg(feature = "host")]
    fn async_to_ffi<'stream, 'b>(
        param: Self::AsyncHostType<'stream, 'b>,
    ) -> Self::FfiType<'stream, 'b> {
        <SharedHeapPerThreadShallowCopy<T> as CudaKernelParameter>::async_to_ffi(param)
    }

    #[cfg(all(not(feature = "host"), target_os = "cuda"))]
    fn with_ffi_as_device<O, const PARAM: usize>(
        param: Self::FfiType<'static, 'static>,
        inner: impl for<'b> FnOnce(Self::DeviceType<'b>) -> O,
    ) -> O {
        emit_param_ptx_jit_marker::<_, PARAM>(param.as_ref());

        <SharedHeapPerThreadShallowCopy<T> as CudaKernelParameter>::with_ffi_as_device::<O, PARAM>(
            param, inner,
        )
    }
}
impl<
        T: RustToCuda<
                CudaRepresentation: crate::safety::SafeDeviceCopy,
                CudaAllocation: EmptyCudaAlloc,
            > + crate::safety::NoSafeAliasing,
    > sealed::Sealed for PtxJit<SharedHeapPerThreadShallowCopy<T>>
{
}

impl<'a, T: 'static + RustToCuda + crate::safety::NoSafeAliasing> CudaKernelParameter
    for &'a PtxJit<SharedHeapPerThreadShallowCopy<T>>
{
    #[cfg(feature = "host")]
    type AsyncHostType<'stream, 'b> =
        <&'a SharedHeapPerThreadShallowCopy<T> as CudaKernelParameter>::AsyncHostType<'stream, 'b>;
    #[cfg(all(not(feature = "host"), target_os = "cuda"))]
    type DeviceType<'b> =
        <&'a SharedHeapPerThreadShallowCopy<T> as CudaKernelParameter>::DeviceType<'b>;
    type FfiType<'stream, 'b> =
        <&'a SharedHeapPerThreadShallowCopy<T> as CudaKernelParameter>::FfiType<'stream, 'b>;
    #[cfg(feature = "host")]
    type SyncHostType =
        <&'a SharedHeapPerThreadShallowCopy<T> as CudaKernelParameter>::SyncHostType;

    #[cfg(feature = "host")]
    fn with_new_async<'stream, O, E: From<rustacuda::error::CudaError>>(
        param: Self::SyncHostType,
        stream: &'stream rustacuda::stream::Stream,
        inner: impl for<'b> FnOnce(Self::AsyncHostType<'stream, 'b>) -> Result<O, E>,
    ) -> Result<O, E> {
        <&'a SharedHeapPerThreadShallowCopy<T> as CudaKernelParameter>::with_new_async(
            param, stream, inner,
        )
    }

    #[cfg(feature = "host")]
    fn with_async_as_ptx_jit<O>(
        param: &Self::AsyncHostType<'_, '_>,
        inner: impl for<'p> FnOnce(Option<&'p NonNull<[u8]>>) -> O,
    ) -> O {
        inner(Some(&param_as_raw_bytes(param.for_host())))
    }

    #[cfg(feature = "host")]
    fn async_to_ffi<'stream, 'b>(
        param: Self::AsyncHostType<'stream, 'b>,
    ) -> Self::FfiType<'stream, 'b> {
        <&'a SharedHeapPerThreadShallowCopy<T> as CudaKernelParameter>::async_to_ffi(param)
    }

    #[cfg(all(not(feature = "host"), target_os = "cuda"))]
    fn with_ffi_as_device<O, const PARAM: usize>(
        param: Self::FfiType<'static, 'static>,
        inner: impl for<'b> FnOnce(Self::DeviceType<'b>) -> O,
    ) -> O {
        emit_param_ptx_jit_marker::<_, PARAM>(param.as_ref());

        <&'a SharedHeapPerThreadShallowCopy<T> as CudaKernelParameter>::with_ffi_as_device::<O, PARAM>(
            param, inner,
        )
    }
}
impl<'a, T: RustToCuda + crate::safety::NoSafeAliasing> sealed::Sealed
    for &'a PtxJit<SharedHeapPerThreadShallowCopy<T>>
{
}

impl<'a, T: 'static + RustToCuda + crate::safety::NoSafeAliasing> CudaKernelParameter
    for &'a mut PtxJit<SharedHeapPerThreadShallowCopy<T>>
{
    #[cfg(feature = "host")]
    type AsyncHostType<'stream, 'b> =
        <&'a mut SharedHeapPerThreadShallowCopy<T> as CudaKernelParameter>::AsyncHostType<
            'stream,
            'b,
        >;
    #[cfg(all(not(feature = "host"), target_os = "cuda"))]
    type DeviceType<'b> =
        <&'a mut SharedHeapPerThreadShallowCopy<T> as CudaKernelParameter>::DeviceType<'b>;
    type FfiType<'stream, 'b> =
        <&'a mut SharedHeapPerThreadShallowCopy<T> as CudaKernelParameter>::FfiType<'stream, 'b>;
    #[cfg(feature = "host")]
    type SyncHostType =
        <&'a mut SharedHeapPerThreadShallowCopy<T> as CudaKernelParameter>::SyncHostType;

    #[cfg(feature = "host")]
    fn with_new_async<'stream, O, E: From<rustacuda::error::CudaError>>(
        param: Self::SyncHostType,
        stream: &'stream rustacuda::stream::Stream,
        inner: impl for<'b> FnOnce(Self::AsyncHostType<'stream, 'b>) -> Result<O, E>,
    ) -> Result<O, E> {
        <&'a mut SharedHeapPerThreadShallowCopy<T> as CudaKernelParameter>::with_new_async(
            param, stream, inner,
        )
    }

    #[cfg(feature = "host")]
    fn with_async_as_ptx_jit<O>(
        param: &Self::AsyncHostType<'_, '_>,
        inner: impl for<'p> FnOnce(Option<&'p NonNull<[u8]>>) -> O,
    ) -> O {
        inner(Some(&param_as_raw_bytes(param.for_host())))
    }

    #[cfg(feature = "host")]
    fn async_to_ffi<'stream, 'b>(
        param: Self::AsyncHostType<'stream, 'b>,
    ) -> Self::FfiType<'stream, 'b> {
        <&'a mut SharedHeapPerThreadShallowCopy<T> as CudaKernelParameter>::async_to_ffi(param)
    }

    #[cfg(all(not(feature = "host"), target_os = "cuda"))]
    fn with_ffi_as_device<O, const PARAM: usize>(
        param: Self::FfiType<'static, 'static>,
        inner: impl for<'b> FnOnce(Self::DeviceType<'b>) -> O,
    ) -> O {
        emit_param_ptx_jit_marker::<_, PARAM>(param.as_ref());

        <&'a mut SharedHeapPerThreadShallowCopy<T> as CudaKernelParameter>::with_ffi_as_device::<
            O,
            PARAM,
        >(param, inner)
    }
}
impl<'a, T: RustToCuda + crate::safety::NoSafeAliasing> sealed::Sealed
    for &'a mut PtxJit<SharedHeapPerThreadShallowCopy<T>>
{
}

#[cfg(feature = "host")]
fn param_as_raw_bytes<T: ?Sized>(r: &T) -> NonNull<[u8]> {
    NonNull::slice_from_raw_parts(NonNull::from(r).cast::<u8>(), core::mem::size_of_val(r))
}

#[cfg(all(not(feature = "host"), target_os = "cuda"))]
fn emit_param_ptx_jit_marker<T: ?Sized, const INDEX: usize>(param: &T) {
    unsafe {
        core::arch::asm!(
            "// <rust-cuda-ptx-jit-const-load-{}-{}> //",
            in(reg32) *(core::ptr::from_ref(param).cast::<u32>()),
            const(INDEX),
        );
    }
}

mod private_shared {
    use const_type_layout::TypeGraphLayout;
    use rustacuda_core::DeviceCopy;

    #[doc(hidden)]
    #[derive(TypeLayout)]
    #[repr(C)]
    pub struct ThreadBlockSharedFfi<T: 'static + TypeGraphLayout> {
        pub(super) _marker: [T; 0],
    }

    // Safety: there is nothing to copy, this is just a zero-sized marker type
    unsafe impl<T: 'static + TypeGraphLayout> DeviceCopy for ThreadBlockSharedFfi<T> {}
}

impl<'a, T: 'static + TypeGraphLayout> CudaKernelParameter
    for &'a mut crate::utils::shared::r#static::ThreadBlockShared<T>
{
    #[cfg(feature = "host")]
    type AsyncHostType<'stream, 'b> = &'b mut crate::utils::shared::r#static::ThreadBlockShared<T>;
    #[cfg(all(not(feature = "host"), target_os = "cuda"))]
    type DeviceType<'b> = &'b mut crate::utils::shared::r#static::ThreadBlockShared<T>;
    type FfiType<'stream, 'b> = private_shared::ThreadBlockSharedFfi<T>;
    #[cfg(feature = "host")]
    type SyncHostType = Self;

    #[cfg(feature = "host")]
    fn with_new_async<'stream, O, E: From<rustacuda::error::CudaError>>(
        param: Self::SyncHostType,
        _stream: &'stream rustacuda::stream::Stream,
        inner: impl for<'b> FnOnce(Self::AsyncHostType<'stream, 'b>) -> Result<O, E>,
    ) -> Result<O, E> {
        inner(param)
    }

    #[cfg(feature = "host")]
    fn with_async_as_ptx_jit<O>(
        _param: &Self::AsyncHostType<'_, '_>,
        inner: impl for<'p> FnOnce(Option<&'p NonNull<[u8]>>) -> O,
    ) -> O {
        inner(None)
    }

    #[cfg(feature = "host")]
    fn async_to_ffi<'stream, 'b>(
        _param: Self::AsyncHostType<'stream, 'b>,
    ) -> Self::FfiType<'stream, 'b> {
        private_shared::ThreadBlockSharedFfi { _marker: [] }
    }

    #[cfg(all(not(feature = "host"), target_os = "cuda"))]
    #[allow(clippy::inline_always)]
    #[inline(always)]
    fn with_ffi_as_device<O, const PARAM: usize>(
        _param: Self::FfiType<'static, 'static>,
        inner: impl for<'b> FnOnce(Self::DeviceType<'b>) -> O,
    ) -> O {
        let mut param = crate::utils::shared::r#static::ThreadBlockShared::new_uninit();

        inner(&mut param)
    }
}
impl<'a, T: 'static + TypeGraphLayout> sealed::Sealed
    for &'a mut crate::utils::shared::r#static::ThreadBlockShared<T>
{
}
