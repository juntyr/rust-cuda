#[cfg(any(not(feature = "host"), doc))]
use core::convert::{AsMut, AsRef};
use core::marker::PhantomData;

#[cfg(feature = "host")]
use alloc::fmt;
#[cfg(not(feature = "host"))]
use core::ops::{Deref, DerefMut};
#[cfg(feature = "host")]
use core::{mem::MaybeUninit, ptr::copy_nonoverlapping};

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

    #[cfg(feature = "host")]
    #[doc(cfg(feature = "host"))]
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

    #[cfg(feature = "host")]
    #[doc(cfg(feature = "host"))]
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
    #[cfg(feature = "host")]
    #[doc(cfg(feature = "host"))]
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

    #[cfg(feature = "host")]
    #[doc(cfg(feature = "host"))]
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

    #[cfg(any(not(feature = "host"), doc))]
    #[doc(cfg(not(feature = "host")))]
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
pub struct DeviceOwnedRef<T: DeviceCopy> {
    pub(super) pointer: *mut T,
    pub(super) marker: PhantomData<T>,
}

// TODO: when should the drop run???
#[cfg(feature = "host")]
impl<T: DeviceCopy> Drop for DeviceOwnedRef<T> {
    fn drop(&mut self) {
        // Safety: pointer comes from [`DeviceBox::into_device`]
        //         i.e. this function completes the roundtrip
        let device_box = unsafe { rustacuda::memory::DeviceBox::from_raw(self.pointer) };

        core::mem::drop(crate::host::CudaDropWrapper::from(device_box));
    }
}

unsafe impl<T: DeviceCopy> DeviceCopy for DeviceOwnedRef<T> {}

#[cfg(any(not(feature = "host"), doc))]
#[doc(cfg(not(feature = "host")))]
impl<T: DeviceCopy> AsRef<T> for DeviceOwnedRef<T> {
    fn as_ref(&self) -> &T {
        unsafe { &*self.pointer }
    }
}

#[cfg(any(not(feature = "host"), doc))]
#[doc(cfg(not(feature = "host")))]
impl<T: DeviceCopy> AsMut<T> for DeviceOwnedRef<T> {
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
impl<T: private::empty::Sealed> EmptyCudaAlloc for T {}

pub trait CudaAlloc: crate_private::alloc::Sealed {}
impl<T: crate_private::alloc::Sealed> CudaAlloc for T {}

impl<T: CudaAlloc> crate_private::alloc::Sealed for Option<T> {}

pub struct NoCudaAlloc;
impl crate_private::alloc::Sealed for NoCudaAlloc {}
impl private::empty::Sealed for NoCudaAlloc {}

pub struct SomeCudaAlloc(());
impl crate_private::alloc::Sealed for SomeCudaAlloc {}
impl !private::empty::Sealed for SomeCudaAlloc {}

pub struct CombinedCudaAlloc<A: CudaAlloc, B: CudaAlloc>(A, B);
impl<A: CudaAlloc, B: CudaAlloc> crate_private::alloc::Sealed for CombinedCudaAlloc<A, B> {}
impl<A: CudaAlloc + EmptyCudaAlloc, B: CudaAlloc + EmptyCudaAlloc> private::empty::Sealed
    for CombinedCudaAlloc<A, B>
{
}
impl<A: CudaAlloc, B: CudaAlloc> CombinedCudaAlloc<A, B> {
    pub fn new(front: A, tail: B) -> Self {
        Self(front, tail)
    }

    pub fn split(self) -> (A, B) {
        (self.0, self.1)
    }
}

mod sealed {
    pub trait Sealed {}
}

// TODO: doc cfg
pub trait CudaKernelParameter: sealed::Sealed {
    #[cfg(feature = "host")]
    type SyncHostType;
    #[cfg(feature = "host")]
    type AsyncHostType<'stream, 'b>;
    type FfiType<'stream, 'b>: rustacuda_core::DeviceCopy;
    type DeviceType;

    #[cfg(feature = "host")]
    #[allow(clippy::missing_errors_doc)] // FIXME
    fn with_new_async<'stream, O, E: From<rustacuda::error::CudaError>>(
        param: Self::SyncHostType,
        stream: &'stream rustacuda::stream::Stream,
        inner: impl for<'b> FnOnce(Self::AsyncHostType<'stream, 'b>) -> Result<O, E>,
    ) -> Result<O, E>;

    #[cfg(feature = "host")]
    fn async_to_ffi<'stream, 'b>(
        param: Self::AsyncHostType<'stream, 'b>,
    ) -> Self::FfiType<'stream, 'b>;

    #[cfg(not(feature = "host"))]
    fn with_ffi_as_device(
        param: Self::FfiType<'static, 'static>,
        inner: impl FnOnce(Self::DeviceType),
    );
}

#[repr(transparent)]
pub struct PerThreadShallowCopy<
    T: crate::safety::SafeDeviceCopy + const_type_layout::TypeGraphLayout,
>(T);

#[cfg(not(feature = "host"))]
impl<T: crate::safety::SafeDeviceCopy + const_type_layout::TypeGraphLayout>
    PerThreadShallowCopy<T>
{
    #[must_use]
    pub fn into_inner(self) -> T {
        self.0
    }
}

#[cfg(not(feature = "host"))]
impl<T: crate::safety::SafeDeviceCopy + const_type_layout::TypeGraphLayout> core::ops::Deref
    for PerThreadShallowCopy<T>
{
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T: crate::safety::SafeDeviceCopy + const_type_layout::TypeGraphLayout> CudaKernelParameter
    for PerThreadShallowCopy<T>
{
    #[cfg(feature = "host")]
    type AsyncHostType<'stream, 'b> = crate::utils::device_copy::SafeDeviceCopyWrapper<T>;
    type DeviceType = PerThreadShallowCopy<T>;
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
    fn async_to_ffi<'stream, 'b>(
        param: Self::AsyncHostType<'stream, 'b>,
    ) -> Self::FfiType<'stream, 'b> {
        param
    }

    #[cfg(not(feature = "host"))]
    fn with_ffi_as_device(
        param: Self::FfiType<'static, 'static>,
        inner: impl FnOnce(Self::DeviceType),
    ) {
        let param = PerThreadShallowCopy(param.into_inner());

        inner(param)
    }
}
impl<T: crate::safety::SafeDeviceCopy + const_type_layout::TypeGraphLayout> sealed::Sealed
    for PerThreadShallowCopy<T>
{
}

impl<'a, T: 'static + crate::safety::SafeDeviceCopy + const_type_layout::TypeGraphLayout>
    CudaKernelParameter for &'a PerThreadShallowCopy<T>
{
    #[cfg(feature = "host")]
    type AsyncHostType<'stream, 'b> = crate::host::HostAndDeviceConstRefAsync<
        'stream,
        'b,
        crate::utils::device_copy::SafeDeviceCopyWrapper<T>,
    >;
    type DeviceType = &'a PerThreadShallowCopy<T>;
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
    fn async_to_ffi<'stream, 'b>(
        param: Self::AsyncHostType<'stream, 'b>,
    ) -> Self::FfiType<'stream, 'b> {
        unsafe { param.for_device_async() }
    }

    #[cfg(not(feature = "host"))]
    fn with_ffi_as_device(
        param: Self::FfiType<'static, 'static>,
        inner: impl FnOnce(Self::DeviceType),
    ) {
        let param = param.as_ref().into_ref();
        // Safety: PerThreadShallowCopy<T> is a transparent newtype wrapper around T
        let param = unsafe { &*(param as *const T).cast::<PerThreadShallowCopy<T>>() };

        inner(param)
    }
}
impl<'a, T: crate::safety::SafeDeviceCopy + const_type_layout::TypeGraphLayout> sealed::Sealed
    for &'a PerThreadShallowCopy<T>
{
}

#[repr(transparent)]
pub struct ShallowInteriorMutable<T: InteriorMutableSafeDeviceCopy>(T);

#[cfg(not(feature = "host"))]
impl<T: InteriorMutableSafeDeviceCopy> core::ops::Deref for ShallowInteriorMutable<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.0
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
    type DeviceType = &'a ShallowInteriorMutable<T>;
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
    fn async_to_ffi<'stream, 'b>(
        param: Self::AsyncHostType<'stream, 'b>,
    ) -> Self::FfiType<'stream, 'b> {
        unsafe { param.for_device_async() }
    }

    #[cfg(not(feature = "host"))]
    fn with_ffi_as_device(
        param: Self::FfiType<'static, 'static>,
        inner: impl FnOnce(Self::DeviceType),
    ) {
        let param = param.as_ref().into_ref();
        // Safety: ShallowInteriorMutable<T> is a transparent newtype wrapper around T
        let param = unsafe { &*(param as *const T).cast::<ShallowInteriorMutable<T>>() };

        inner(param)
    }
}
impl<'a, T: InteriorMutableSafeDeviceCopy> sealed::Sealed for &'a ShallowInteriorMutable<T> {}

pub trait InteriorMutableSafeDeviceCopy:
    crate::safety::SafeDeviceCopy + const_type_layout::TypeGraphLayout + sealed::Sealed
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

#[repr(transparent)]
pub struct SharedHeapPerThreadShallowCopy<T: RustToCuda>(core::mem::ManuallyDrop<T>);

#[cfg(not(feature = "host"))]
impl<T: RustToCuda> SharedHeapPerThreadShallowCopy<T> {
    #[must_use]
    fn new(value: T) -> Self {
        Self(core::mem::ManuallyDrop::new(value))
    }
}

#[cfg(not(feature = "host"))]
impl<
        T: RustToCuda<
            CudaRepresentation: crate::safety::SafeDeviceCopy,
            CudaAllocation: EmptyCudaAlloc,
        >,
    > SharedHeapPerThreadShallowCopy<T>
{
    #[must_use]
    pub fn into_inner(self) -> T {
        core::mem::ManuallyDrop::into_inner(self.0)
    }
}

#[cfg(not(feature = "host"))]
impl<T: RustToCuda> core::ops::Deref for SharedHeapPerThreadShallowCopy<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[cfg(not(feature = "host"))]
impl<T: RustToCuda> core::ops::DerefMut for SharedHeapPerThreadShallowCopy<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<
        T: RustToCuda<
            CudaRepresentation: crate::safety::SafeDeviceCopy,
            CudaAllocation: EmptyCudaAlloc,
        >,
    > CudaKernelParameter for SharedHeapPerThreadShallowCopy<T>
{
    #[cfg(feature = "host")]
    type AsyncHostType<'stream, 'b> = crate::host::HostAndDeviceOwnedAsync<
        'stream,
        DeviceAccessible<<T as RustToCuda>::CudaRepresentation>,
    >;
    type DeviceType = SharedHeapPerThreadShallowCopy<T>;
    // TODO: where does the drop happen?
    type FfiType<'stream, 'b> =
        DeviceOwnedRef<DeviceAccessible<<T as RustToCuda>::CudaRepresentation>>;
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
    fn async_to_ffi<'stream, 'b>(
        param: Self::AsyncHostType<'stream, 'b>,
    ) -> Self::FfiType<'stream, 'b> {
        unsafe { param.for_device_async() }
    }

    #[cfg(not(feature = "host"))]
    fn with_ffi_as_device(
        param: Self::FfiType<'static, 'static>,
        inner: impl FnOnce(Self::DeviceType),
    ) {
        let param =
            SharedHeapPerThreadShallowCopy::new(unsafe { CudaAsRust::as_rust(param.as_ref()) });

        inner(param)
    }
}
impl<
        T: RustToCuda<
            CudaRepresentation: crate::safety::SafeDeviceCopy,
            CudaAllocation: EmptyCudaAlloc,
        >,
    > sealed::Sealed for SharedHeapPerThreadShallowCopy<T>
{
}

impl<'a, T: 'static + RustToCuda> CudaKernelParameter for &'a SharedHeapPerThreadShallowCopy<T> {
    #[cfg(feature = "host")]
    type AsyncHostType<'stream, 'b> = crate::host::HostAndDeviceConstRefAsync<
        'stream,
        'b,
        DeviceAccessible<<T as RustToCuda>::CudaRepresentation>,
    >;
    type DeviceType = &'a SharedHeapPerThreadShallowCopy<T>;
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
    fn async_to_ffi<'stream, 'b>(
        param: Self::AsyncHostType<'stream, 'b>,
    ) -> Self::FfiType<'stream, 'b> {
        unsafe { param.for_device_async() }
    }

    #[cfg(not(feature = "host"))]
    fn with_ffi_as_device(
        param: Self::FfiType<'static, 'static>,
        inner: impl FnOnce(Self::DeviceType),
    ) {
        // param must never be dropped as we do NOT own any of the
        //  heap memory it might reference
        let param =
            SharedHeapPerThreadShallowCopy::new(unsafe { CudaAsRust::as_rust(param.as_ref()) });

        inner(&param)
    }
}
impl<'a, T: RustToCuda> sealed::Sealed for &'a SharedHeapPerThreadShallowCopy<T> {}

impl<'a, T: 'static + RustToCuda> CudaKernelParameter
    for &'a mut SharedHeapPerThreadShallowCopy<T>
{
    #[cfg(feature = "host")]
    type AsyncHostType<'stream, 'b> = crate::host::HostAndDeviceMutRefAsync<
        'stream,
        'b,
        DeviceAccessible<<T as RustToCuda>::CudaRepresentation>,
    >;
    type DeviceType = &'a mut SharedHeapPerThreadShallowCopy<T>;
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
    fn async_to_ffi<'stream, 'b>(
        mut param: Self::AsyncHostType<'stream, 'b>,
    ) -> Self::FfiType<'stream, 'b> {
        unsafe { param.for_device_async() }
    }

    #[cfg(not(feature = "host"))]
    fn with_ffi_as_device(
        mut param: Self::FfiType<'static, 'static>,
        inner: impl FnOnce(Self::DeviceType),
    ) {
        // param must never be dropped as we do NOT own any of the
        //  heap memory it might reference
        let mut param =
            SharedHeapPerThreadShallowCopy::new(unsafe { CudaAsRust::as_rust(param.as_mut()) });

        inner(&mut param)
    }
}
impl<'a, T: RustToCuda> sealed::Sealed for &'a mut SharedHeapPerThreadShallowCopy<T> {}
