#[cfg(feature = "device")]
use core::convert::AsRef;
use core::{
    marker::PhantomData,
    ops::{Deref, DerefMut},
};

#[cfg(feature = "host")]
use std::{alloc::Layout, ptr::NonNull};

use const_type_layout::TypeGraphLayout;

use crate::{
    alloc::EmptyCudaAlloc,
    kernel::{sealed, CudaKernelParameter},
    lend::RustToCuda,
    safety::PortableBitSemantics,
    utils::ffi::{DeviceAccessible, DeviceConstRef, DeviceOwnedRef},
};

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
    T: crate::safety::StackOnly + crate::safety::PortableBitSemantics + TypeGraphLayout,
> {
    never: !,
    _marker: PhantomData<T>,
}

impl<T: crate::safety::StackOnly + crate::safety::PortableBitSemantics + TypeGraphLayout> Deref
    for PerThreadShallowCopy<T>
{
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.never
    }
}

impl<T: crate::safety::StackOnly + crate::safety::PortableBitSemantics + TypeGraphLayout> DerefMut
    for PerThreadShallowCopy<T>
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.never
    }
}

impl<
        T: Copy
            + Send
            + crate::safety::StackOnly
            + crate::safety::PortableBitSemantics
            + TypeGraphLayout,
    > CudaKernelParameter for PerThreadShallowCopy<T>
{
    #[cfg(feature = "host")]
    type AsyncHostType<'stream, 'b> =
        crate::utils::adapter::RustToCudaWithPortableBitCopySemantics<T>;
    #[cfg(any(feature = "device", doc))]
    type DeviceType<'b> = T;
    type FfiType<'stream, 'b> = crate::utils::adapter::RustToCudaWithPortableBitCopySemantics<T>;
    #[cfg(feature = "host")]
    type SyncHostType = T;

    #[cfg(feature = "host")]
    fn with_new_async<'stream, O, E: From<rustacuda::error::CudaError>>(
        param: Self::SyncHostType,
        _stream: &'stream rustacuda::stream::Stream,
        inner: impl for<'b> FnOnce(Self::AsyncHostType<'stream, 'b>) -> Result<O, E>,
    ) -> Result<O, E> {
        inner(crate::utils::adapter::RustToCudaWithPortableBitCopySemantics::from(param))
    }

    #[cfg(feature = "host")]
    fn with_async_as_ptx_jit<O>(
        _param: &Self::AsyncHostType<'_, '_>,
        _token: sealed::Token,
        inner: impl for<'p> FnOnce(Option<&'p NonNull<[u8]>>) -> O,
    ) -> O {
        inner(None)
    }

    #[cfg(feature = "host")]
    fn shared_layout_for_async(
        _param: &Self::AsyncHostType<'_, '_>,
        _token: sealed::Token,
    ) -> Layout {
        Layout::new::<()>()
    }

    #[cfg(feature = "host")]
    fn async_to_ffi<'stream, 'b, E: From<rustacuda::error::CudaError>>(
        param: Self::AsyncHostType<'stream, 'b>,
        _token: sealed::Token,
    ) -> Result<Self::FfiType<'stream, 'b>, E> {
        Ok(param)
    }

    #[cfg(feature = "device")]
    unsafe fn with_ffi_as_device<O, const PARAM: usize>(
        param: Self::FfiType<'static, 'static>,
        inner: impl for<'b> FnOnce(Self::DeviceType<'b>) -> O,
    ) -> O {
        let param = param.into_inner();

        inner(param)
    }
}
impl<
        T: Copy
            + Send
            + crate::safety::StackOnly
            + crate::safety::PortableBitSemantics
            + TypeGraphLayout,
    > sealed::Sealed for PerThreadShallowCopy<T>
{
}

impl<
        'a,
        T: 'static
            + Sync
            + crate::safety::StackOnly
            + crate::safety::PortableBitSemantics
            + TypeGraphLayout,
    > CudaKernelParameter for &'a PerThreadShallowCopy<T>
{
    #[cfg(feature = "host")]
    type AsyncHostType<'stream, 'b> = crate::utils::r#async::Async<
        'b,
        'stream,
        crate::host::HostAndDeviceConstRef<'b, T>,
        crate::utils::r#async::NoCompletion,
    >;
    #[cfg(any(feature = "device", doc))]
    type DeviceType<'b> = &'b T;
    type FfiType<'stream, 'b> = DeviceConstRef<'b, T>;
    #[cfg(feature = "host")]
    type SyncHostType = &'a T;

    #[cfg(feature = "host")]
    fn with_new_async<'stream, O, E: From<rustacuda::error::CudaError>>(
        param: Self::SyncHostType,
        stream: &'stream rustacuda::stream::Stream,
        inner: impl for<'b> FnOnce(Self::AsyncHostType<'stream, 'b>) -> Result<O, E>,
    ) -> Result<O, E> {
        crate::host::HostAndDeviceConstRef::with_new(param, |const_ref| {
            inner(const_ref.as_async(stream))
        })
    }

    #[cfg(feature = "host")]
    fn with_async_as_ptx_jit<O>(
        _param: &Self::AsyncHostType<'_, '_>,
        _token: sealed::Token,
        inner: impl for<'p> FnOnce(Option<&'p NonNull<[u8]>>) -> O,
    ) -> O {
        inner(None)
    }

    #[cfg(feature = "host")]
    fn shared_layout_for_async(
        _param: &Self::AsyncHostType<'_, '_>,
        _token: sealed::Token,
    ) -> Layout {
        Layout::new::<()>()
    }

    #[cfg(feature = "host")]
    fn async_to_ffi<'stream, 'b, E: From<rustacuda::error::CudaError>>(
        param: Self::AsyncHostType<'stream, 'b>,
        _token: sealed::Token,
    ) -> Result<Self::FfiType<'stream, 'b>, E> {
        let (param, _completion): (_, Option<crate::utils::r#async::NoCompletion>) =
            unsafe { param.unwrap_unchecked()? };
        Ok(param.for_device())
    }

    #[cfg(feature = "device")]
    unsafe fn with_ffi_as_device<O, const PARAM: usize>(
        param: Self::FfiType<'static, 'static>,
        inner: impl for<'b> FnOnce(Self::DeviceType<'b>) -> O,
    ) -> O {
        let param = param.as_ref();

        inner(param)
    }
}
impl<
        'a,
        T: 'static
            + Sync
            + crate::safety::StackOnly
            + crate::safety::PortableBitSemantics
            + TypeGraphLayout,
    > sealed::Sealed for &'a PerThreadShallowCopy<T>
{
}

impl<
        'a,
        T: 'static
            + Sync
            + crate::safety::StackOnly
            + crate::safety::PortableBitSemantics
            + TypeGraphLayout,
    > CudaKernelParameter for &'a PtxJit<PerThreadShallowCopy<T>>
{
    #[cfg(feature = "host")]
    type AsyncHostType<'stream, 'b> =
        <&'a PerThreadShallowCopy<T> as CudaKernelParameter>::AsyncHostType<'stream, 'b>;
    #[cfg(any(feature = "device", doc))]
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
        _token: sealed::Token,
        inner: impl for<'p> FnOnce(Option<&'p NonNull<[u8]>>) -> O,
    ) -> O {
        let param = unsafe { param.unwrap_ref_unchecked() };
        inner(Some(&param_as_raw_bytes(param.for_host())))
    }

    #[cfg(feature = "host")]
    fn shared_layout_for_async(
        _param: &Self::AsyncHostType<'_, '_>,
        _token: sealed::Token,
    ) -> Layout {
        Layout::new::<()>()
    }

    #[cfg(feature = "host")]
    fn async_to_ffi<'stream, 'b, E: From<rustacuda::error::CudaError>>(
        param: Self::AsyncHostType<'stream, 'b>,
        token: sealed::Token,
    ) -> Result<Self::FfiType<'stream, 'b>, E> {
        <&'a PerThreadShallowCopy<T> as CudaKernelParameter>::async_to_ffi(param, token)
    }

    #[cfg(feature = "device")]
    unsafe fn with_ffi_as_device<O, const PARAM: usize>(
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
        T: 'static
            + Sync
            + crate::safety::StackOnly
            + crate::safety::PortableBitSemantics
            + TypeGraphLayout,
    > sealed::Sealed for &'a PtxJit<PerThreadShallowCopy<T>>
{
}

pub struct ShallowInteriorMutable<
    T: Sync
        + crate::safety::StackOnly
        + crate::safety::PortableBitSemantics
        + TypeGraphLayout
        + InteriorMutableSync,
> {
    never: !,
    _marker: PhantomData<T>,
}

impl<
        T: Sync
            + crate::safety::StackOnly
            + crate::safety::PortableBitSemantics
            + TypeGraphLayout
            + InteriorMutableSync,
    > Deref for ShallowInteriorMutable<T>
{
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.never
    }
}

impl<
        'a,
        T: 'static
            + Sync
            + crate::safety::StackOnly
            + crate::safety::PortableBitSemantics
            + TypeGraphLayout
            + InteriorMutableSync,
    > CudaKernelParameter for &'a ShallowInteriorMutable<T>
{
    #[cfg(feature = "host")]
    type AsyncHostType<'stream, 'b> = crate::utils::r#async::Async<
        'b,
        'stream,
        crate::host::HostAndDeviceConstRef<'b, T>,
        crate::utils::r#async::NoCompletion,
    >;
    #[cfg(any(feature = "device", doc))]
    type DeviceType<'b> = &'b T;
    type FfiType<'stream, 'b> = DeviceConstRef<'b, T>;
    #[cfg(feature = "host")]
    /// The kernel takes a mutable borrow of the interior mutable data to ensure
    /// the interior mutability is limited to just this kernel invocation.
    type SyncHostType = &'a mut T;

    #[cfg(feature = "host")]
    fn with_new_async<'stream, O, E: From<rustacuda::error::CudaError>>(
        param: Self::SyncHostType,
        stream: &'stream rustacuda::stream::Stream,
        inner: impl for<'b> FnOnce(Self::AsyncHostType<'stream, 'b>) -> Result<O, E>,
    ) -> Result<O, E> {
        crate::host::HostAndDeviceMutRef::with_new(param, |const_ref| {
            inner(const_ref.as_ref().as_async(stream))
        })
    }

    #[cfg(feature = "host")]
    fn with_async_as_ptx_jit<O>(
        _param: &Self::AsyncHostType<'_, '_>,
        _token: sealed::Token,
        inner: impl for<'p> FnOnce(Option<&'p NonNull<[u8]>>) -> O,
    ) -> O {
        inner(None)
    }

    #[cfg(feature = "host")]
    fn shared_layout_for_async(
        _param: &Self::AsyncHostType<'_, '_>,
        _token: sealed::Token,
    ) -> Layout {
        Layout::new::<()>()
    }

    #[cfg(feature = "host")]
    fn async_to_ffi<'stream, 'b, E: From<rustacuda::error::CudaError>>(
        param: Self::AsyncHostType<'stream, 'b>,
        _token: sealed::Token,
    ) -> Result<Self::FfiType<'stream, 'b>, E> {
        let (param, _completion): (_, Option<crate::utils::r#async::NoCompletion>) =
            unsafe { param.unwrap_unchecked()? };
        Ok(param.for_device())
    }

    #[cfg(feature = "device")]
    unsafe fn with_ffi_as_device<O, const PARAM: usize>(
        param: Self::FfiType<'static, 'static>,
        inner: impl for<'b> FnOnce(Self::DeviceType<'b>) -> O,
    ) -> O {
        let param = param.as_ref();

        inner(param)
    }
}
impl<
        'a,
        T: crate::safety::StackOnly
            + Sync
            + crate::safety::PortableBitSemantics
            + TypeGraphLayout
            + InteriorMutableSync,
    > sealed::Sealed for &'a ShallowInteriorMutable<T>
{
}

pub trait InteriorMutableSync: Sync + sealed::Sealed {}

macro_rules! impl_atomic_interior_mutable {
    ($atomic:ident($interior:ty)) => {
        impl InteriorMutableSync for core::sync::atomic::$atomic {}
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

impl<T: crate::safety::StackOnly + crate::safety::PortableBitSemantics + Sync> InteriorMutableSync
    for core::cell::SyncUnsafeCell<T>
{
}
impl<T: crate::safety::StackOnly + crate::safety::PortableBitSemantics + Sync> sealed::Sealed
    for core::cell::SyncUnsafeCell<T>
{
}

pub struct SharedHeapPerThreadShallowCopy<T: RustToCuda> {
    never: !,
    _marker: PhantomData<T>,
}

impl<T: RustToCuda> Deref for SharedHeapPerThreadShallowCopy<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.never
    }
}

impl<
        T: Send
            + Clone
            + RustToCuda<
                CudaRepresentation: 'static + crate::safety::StackOnly,
                CudaAllocation: EmptyCudaAlloc,
            >,
    > CudaKernelParameter for SharedHeapPerThreadShallowCopy<T>
{
    #[cfg(feature = "host")]
    type AsyncHostType<'stream, 'b> = crate::utils::r#async::Async<
        'b,
        'stream,
        crate::host::HostAndDeviceOwned<
            'b,
            DeviceAccessible<<T as RustToCuda>::CudaRepresentation>,
        >,
        crate::utils::r#async::NoCompletion,
    >;
    #[cfg(any(feature = "device", doc))]
    type DeviceType<'b> = T;
    type FfiType<'stream, 'b> =
        DeviceOwnedRef<'b, DeviceAccessible<<T as RustToCuda>::CudaRepresentation>>;
    #[cfg(feature = "host")]
    type SyncHostType = T;

    #[cfg(feature = "host")]
    fn with_new_async<'stream, O, E: From<rustacuda::error::CudaError>>(
        param: Self::SyncHostType,
        stream: &'stream rustacuda::stream::Stream,
        inner: impl for<'b> FnOnce(Self::AsyncHostType<'stream, 'b>) -> Result<O, E>,
    ) -> Result<O, E> {
        crate::lend::LendToCuda::move_to_cuda(param, |param| inner(param.into_async(stream)))
    }

    #[cfg(feature = "host")]
    fn with_async_as_ptx_jit<O>(
        _param: &Self::AsyncHostType<'_, '_>,
        _token: sealed::Token,
        inner: impl for<'p> FnOnce(Option<&'p NonNull<[u8]>>) -> O,
    ) -> O {
        inner(None)
    }

    #[cfg(feature = "host")]
    fn shared_layout_for_async(
        _param: &Self::AsyncHostType<'_, '_>,
        _token: sealed::Token,
    ) -> Layout {
        Layout::new::<()>()
    }

    #[cfg(feature = "host")]
    fn async_to_ffi<'stream, 'b, E: From<rustacuda::error::CudaError>>(
        param: Self::AsyncHostType<'stream, 'b>,
        _token: sealed::Token,
    ) -> Result<Self::FfiType<'stream, 'b>, E> {
        let (param, _completion): (_, Option<crate::utils::r#async::NoCompletion>) =
            unsafe { param.unwrap_unchecked()? };
        Ok(param.for_device())
    }

    #[cfg(feature = "device")]
    unsafe fn with_ffi_as_device<O, const PARAM: usize>(
        param: Self::FfiType<'static, 'static>,
        inner: impl for<'b> FnOnce(Self::DeviceType<'b>) -> O,
    ) -> O {
        unsafe { crate::lend::BorrowFromRust::with_moved_from_rust(param, inner) }
    }
}
impl<
        T: Send
            + Clone
            + RustToCuda<
                CudaRepresentation: 'static + crate::safety::StackOnly,
                CudaAllocation: EmptyCudaAlloc,
            >,
    > sealed::Sealed for SharedHeapPerThreadShallowCopy<T>
{
}

impl<'a, T: 'static + Sync + RustToCuda> CudaKernelParameter
    for &'a SharedHeapPerThreadShallowCopy<T>
{
    #[cfg(feature = "host")]
    type AsyncHostType<'stream, 'b> = crate::utils::r#async::Async<
        'b,
        'stream,
        crate::host::HostAndDeviceConstRef<
            'b,
            DeviceAccessible<<T as RustToCuda>::CudaRepresentation>,
        >,
        crate::utils::r#async::NoCompletion,
    >;
    #[cfg(any(feature = "device", doc))]
    type DeviceType<'b> = &'b T;
    type FfiType<'stream, 'b> =
        DeviceConstRef<'b, DeviceAccessible<<T as RustToCuda>::CudaRepresentation>>;
    #[cfg(feature = "host")]
    type SyncHostType = &'a T;

    #[cfg(feature = "host")]
    fn with_new_async<'stream, O, E: From<rustacuda::error::CudaError>>(
        param: Self::SyncHostType,
        stream: &'stream rustacuda::stream::Stream,
        inner: impl for<'b> FnOnce(Self::AsyncHostType<'stream, 'b>) -> Result<O, E>,
    ) -> Result<O, E> {
        crate::lend::LendToCuda::lend_to_cuda(param, |param| inner(param.as_async(stream)))
    }

    #[cfg(feature = "host")]
    fn with_async_as_ptx_jit<O>(
        _param: &Self::AsyncHostType<'_, '_>,
        _token: sealed::Token,
        inner: impl for<'p> FnOnce(Option<&'p NonNull<[u8]>>) -> O,
    ) -> O {
        inner(None)
    }

    #[cfg(feature = "host")]
    fn shared_layout_for_async(
        _param: &Self::AsyncHostType<'_, '_>,
        _token: sealed::Token,
    ) -> Layout {
        Layout::new::<()>()
    }

    #[cfg(feature = "host")]
    fn async_to_ffi<'stream, 'b, E: From<rustacuda::error::CudaError>>(
        param: Self::AsyncHostType<'stream, 'b>,
        _token: sealed::Token,
    ) -> Result<Self::FfiType<'stream, 'b>, E> {
        let (param, _completion): (_, Option<crate::utils::r#async::NoCompletion>) =
            unsafe { param.unwrap_unchecked()? };
        Ok(param.for_device())
    }

    #[cfg(feature = "device")]
    unsafe fn with_ffi_as_device<O, const PARAM: usize>(
        param: Self::FfiType<'static, 'static>,
        inner: impl for<'b> FnOnce(Self::DeviceType<'b>) -> O,
    ) -> O {
        unsafe { crate::lend::BorrowFromRust::with_borrow_from_rust(param, inner) }
    }
}
impl<'a, T: Sync + RustToCuda> sealed::Sealed for &'a SharedHeapPerThreadShallowCopy<T> {}

impl<
        T: Send
            + Clone
            + RustToCuda<
                CudaRepresentation: 'static + crate::safety::StackOnly,
                CudaAllocation: EmptyCudaAlloc,
            >,
    > CudaKernelParameter for PtxJit<SharedHeapPerThreadShallowCopy<T>>
{
    #[cfg(feature = "host")]
    type AsyncHostType<'stream, 'b> =
        <SharedHeapPerThreadShallowCopy<T> as CudaKernelParameter>::AsyncHostType<'stream, 'b>;
    #[cfg(any(feature = "device", doc))]
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
        _token: sealed::Token,
        inner: impl for<'p> FnOnce(Option<&'p NonNull<[u8]>>) -> O,
    ) -> O {
        let param = unsafe { param.unwrap_ref_unchecked() };
        inner(Some(&param_as_raw_bytes(param.for_host())))
    }

    #[cfg(feature = "host")]
    fn async_to_ffi<'stream, 'b, E: From<rustacuda::error::CudaError>>(
        param: Self::AsyncHostType<'stream, 'b>,
        token: sealed::Token,
    ) -> Result<Self::FfiType<'stream, 'b>, E> {
        <SharedHeapPerThreadShallowCopy<T> as CudaKernelParameter>::async_to_ffi(param, token)
    }

    #[cfg(feature = "host")]
    fn shared_layout_for_async(
        _param: &Self::AsyncHostType<'_, '_>,
        _token: sealed::Token,
    ) -> Layout {
        Layout::new::<()>()
    }

    #[cfg(feature = "device")]
    unsafe fn with_ffi_as_device<O, const PARAM: usize>(
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
        T: Send
            + Clone
            + RustToCuda<
                CudaRepresentation: 'static + crate::safety::StackOnly,
                CudaAllocation: EmptyCudaAlloc,
            >,
    > sealed::Sealed for PtxJit<SharedHeapPerThreadShallowCopy<T>>
{
}

impl<'a, T: 'static + Sync + RustToCuda> CudaKernelParameter
    for &'a PtxJit<SharedHeapPerThreadShallowCopy<T>>
{
    #[cfg(feature = "host")]
    type AsyncHostType<'stream, 'b> =
        <&'a SharedHeapPerThreadShallowCopy<T> as CudaKernelParameter>::AsyncHostType<'stream, 'b>;
    #[cfg(any(feature = "device", doc))]
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
        _token: sealed::Token,
        inner: impl for<'p> FnOnce(Option<&'p NonNull<[u8]>>) -> O,
    ) -> O {
        let param = unsafe { param.unwrap_ref_unchecked() };
        inner(Some(&param_as_raw_bytes(param.for_host())))
    }

    #[cfg(feature = "host")]
    fn shared_layout_for_async(
        _param: &Self::AsyncHostType<'_, '_>,
        _token: sealed::Token,
    ) -> Layout {
        Layout::new::<()>()
    }

    #[cfg(feature = "host")]
    fn async_to_ffi<'stream, 'b, E: From<rustacuda::error::CudaError>>(
        param: Self::AsyncHostType<'stream, 'b>,
        token: sealed::Token,
    ) -> Result<Self::FfiType<'stream, 'b>, E> {
        <&'a SharedHeapPerThreadShallowCopy<T> as CudaKernelParameter>::async_to_ffi(param, token)
    }

    #[cfg(feature = "device")]
    unsafe fn with_ffi_as_device<O, const PARAM: usize>(
        param: Self::FfiType<'static, 'static>,
        inner: impl for<'b> FnOnce(Self::DeviceType<'b>) -> O,
    ) -> O {
        emit_param_ptx_jit_marker::<_, PARAM>(param.as_ref());

        <&'a SharedHeapPerThreadShallowCopy<T> as CudaKernelParameter>::with_ffi_as_device::<O, PARAM>(
            param, inner,
        )
    }
}
impl<'a, T: 'static + Sync + RustToCuda> sealed::Sealed
    for &'a PtxJit<SharedHeapPerThreadShallowCopy<T>>
{
}

#[cfg(feature = "host")]
fn param_as_raw_bytes<T: ?Sized>(r: &T) -> NonNull<[u8]> {
    NonNull::slice_from_raw_parts(NonNull::from(r).cast::<u8>(), core::mem::size_of_val(r))
}

#[cfg(feature = "device")]
fn emit_param_ptx_jit_marker<T: ?Sized, const INDEX: usize>(param: &T) {
    unsafe {
        core::arch::asm!(
            "// <rust-cuda-ptx-jit-const-load-{param_reg}-{param_index}> //",
            param_reg = in(reg32) *(core::ptr::from_ref(param).cast::<u32>()),
            param_index = const(INDEX),
        );
    }
}

mod private_shared {
    use core::marker::PhantomData;

    use const_type_layout::{TypeGraphLayout, TypeLayout};

    use crate::safety::PortableBitSemantics;

    #[doc(hidden)]
    #[derive(TypeLayout)]
    #[repr(C)]
    pub struct ThreadBlockSharedFfi<T: 'static> {
        pub(super) _dummy: [u8; 0],
        pub(super) _marker: PhantomData<T>,
    }

    #[doc(hidden)]
    #[derive(TypeLayout)]
    #[repr(C)]
    pub struct ThreadBlockSharedSliceFfi<T: 'static + PortableBitSemantics + TypeGraphLayout> {
        pub(super) len: usize,
        pub(super) _marker: [T; 0],
    }
}

impl<'a, T: 'static> CudaKernelParameter for &'a mut crate::utils::shared::ThreadBlockShared<T> {
    #[cfg(feature = "host")]
    type AsyncHostType<'stream, 'b> = &'b mut crate::utils::shared::ThreadBlockShared<T>;
    #[cfg(any(feature = "device", doc))]
    type DeviceType<'b> = &'b mut crate::utils::shared::ThreadBlockShared<T>;
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
        _token: sealed::Token,
        inner: impl for<'p> FnOnce(Option<&'p NonNull<[u8]>>) -> O,
    ) -> O {
        inner(None)
    }

    #[cfg(feature = "host")]
    fn shared_layout_for_async(
        _param: &Self::AsyncHostType<'_, '_>,
        _token: sealed::Token,
    ) -> Layout {
        Layout::new::<()>()
    }

    #[cfg(feature = "host")]
    fn async_to_ffi<'stream, 'b, E: From<rustacuda::error::CudaError>>(
        _param: Self::AsyncHostType<'stream, 'b>,
        _token: sealed::Token,
    ) -> Result<Self::FfiType<'stream, 'b>, E> {
        Ok(private_shared::ThreadBlockSharedFfi {
            _dummy: [],
            _marker: PhantomData::<T>,
        })
    }

    #[cfg(feature = "device")]
    #[allow(clippy::inline_always)]
    #[inline(always)]
    unsafe fn with_ffi_as_device<O, const PARAM: usize>(
        _param: Self::FfiType<'static, 'static>,
        inner: impl for<'b> FnOnce(Self::DeviceType<'b>) -> O,
    ) -> O {
        let mut param = crate::utils::shared::ThreadBlockShared::new_uninit();

        inner(&mut param)
    }
}
impl<'a, T: 'static> sealed::Sealed for &'a mut crate::utils::shared::ThreadBlockShared<T> {}

impl<'a, T: 'static + PortableBitSemantics + TypeGraphLayout> CudaKernelParameter
    for &'a mut crate::utils::shared::ThreadBlockSharedSlice<T>
{
    #[cfg(feature = "host")]
    type AsyncHostType<'stream, 'b> = &'b mut crate::utils::shared::ThreadBlockSharedSlice<T>;
    #[cfg(any(feature = "device", doc))]
    type DeviceType<'b> = &'b mut crate::utils::shared::ThreadBlockSharedSlice<T>;
    type FfiType<'stream, 'b> = private_shared::ThreadBlockSharedSliceFfi<T>;
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
        _token: sealed::Token,
        inner: impl for<'p> FnOnce(Option<&'p NonNull<[u8]>>) -> O,
    ) -> O {
        inner(None)
    }

    #[cfg(feature = "host")]
    fn shared_layout_for_async(
        param: &Self::AsyncHostType<'_, '_>,
        _token: sealed::Token,
    ) -> Layout {
        param.layout()
    }

    #[cfg(feature = "host")]
    fn async_to_ffi<'stream, 'b, E: From<rustacuda::error::CudaError>>(
        param: Self::AsyncHostType<'stream, 'b>,
        _token: sealed::Token,
    ) -> Result<Self::FfiType<'stream, 'b>, E> {
        Ok(private_shared::ThreadBlockSharedSliceFfi {
            len: param.len(),
            _marker: [],
        })
    }

    #[cfg(feature = "device")]
    #[allow(clippy::inline_always)]
    #[inline(always)]
    unsafe fn with_ffi_as_device<O, const PARAM: usize>(
        param: Self::FfiType<'static, 'static>,
        inner: impl for<'b> FnOnce(Self::DeviceType<'b>) -> O,
    ) -> O {
        unsafe {
            crate::utils::shared::ThreadBlockSharedSlice::with_uninit_for_len(param.len, inner)
        }
    }
}
impl<'a, T: 'static + PortableBitSemantics + TypeGraphLayout> sealed::Sealed
    for &'a mut crate::utils::shared::ThreadBlockSharedSlice<T>
{
}
