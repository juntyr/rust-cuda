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
    safety::{PortableBitSemantics, SafeMutableAliasing},
    utils::ffi::{DeviceAccessible, DeviceConstRef, DeviceMutRef, DeviceOwnedRef},
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
    type AsyncHostType<'stream, 'b> = T where Self: 'b;
    #[cfg(any(feature = "device", doc))]
    type DeviceType<'b> = T where Self: 'b;
    type FfiType<'stream, 'b> = crate::utils::adapter::RustToCudaWithPortableBitCopySemantics<T> where Self: 'b;
    #[cfg(feature = "host")]
    type SyncHostType = T;

    #[cfg(feature = "host")]
    fn with_new_async<'stream, 'b, O, E: From<rustacuda::error::CudaError>>(
        param: Self::SyncHostType,
        _stream: crate::host::Stream<'stream>,
        #[cfg(not(doc))] inner: impl super::WithNewAsync<'stream, Self, O, E>,
        #[cfg(doc)] inner: impl FnOnce(Self::AsyncHostType<'stream, 'b>) -> Result<O, E>,
    ) -> Result<O, E>
    where
        Self: 'b,
    {
        inner.with(param)
    }

    #[cfg(feature = "host")]
    fn with_async_as_ptx_jit<'stream, 'b, O>(
        _param: &Self::AsyncHostType<'stream, 'b>,
        _token: sealed::Token,
        inner: impl for<'p> FnOnce(Option<&'p NonNull<[u8]>>) -> O,
    ) -> O
    where
        Self: 'b,
    {
        inner(None)
    }

    #[cfg(feature = "host")]
    fn shared_layout_for_async<'stream, 'b>(
        _param: &Self::AsyncHostType<'stream, 'b>,
        _token: sealed::Token,
    ) -> Layout
    where
        Self: 'b,
    {
        Layout::new::<()>()
    }

    #[cfg(feature = "host")]
    fn async_to_ffi<'stream, 'b, E: From<rustacuda::error::CudaError>>(
        param: Self::AsyncHostType<'stream, 'b>,
        _token: sealed::Token,
    ) -> Result<Self::FfiType<'stream, 'b>, E>
    where
        Self: 'b,
    {
        Ok(crate::utils::adapter::RustToCudaWithPortableBitCopySemantics::from(param))
    }

    #[cfg(feature = "device")]
    unsafe fn with_ffi_as_device<'short, O, const PARAM: usize>(
        param: Self::FfiType<'static, 'short>,
        inner: impl super::WithFfiAsDevice<Self, O>,
    ) -> O
    where
        Self: 'short,
    {
        let param = param.into_inner();

        inner.with(param)
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
        T: Sync + crate::safety::StackOnly + crate::safety::PortableBitSemantics + TypeGraphLayout,
    > CudaKernelParameter for &'a PerThreadShallowCopy<T>
{
    #[cfg(feature = "host")]
    type AsyncHostType<'stream, 'b> = crate::utils::r#async::AsyncProj<
        'b,
        'stream,
        crate::host::HostAndDeviceConstRef<'b, T>,
    > where Self: 'b;
    #[cfg(any(feature = "device", doc))]
    type DeviceType<'b> = &'b T where Self: 'b;
    type FfiType<'stream, 'b> = DeviceConstRef<'b, T> where Self: 'b;
    #[cfg(feature = "host")]
    type SyncHostType = &'a T;

    #[cfg(feature = "host")]
    fn with_new_async<'stream, 'b, O, E: From<rustacuda::error::CudaError>>(
        param: Self::SyncHostType,
        stream: crate::host::Stream<'stream>,
        #[cfg(not(doc))] inner: impl super::WithNewAsync<'stream, Self, O, E>,
        #[cfg(doc)] inner: impl FnOnce(Self::AsyncHostType<'stream, 'b>) -> Result<O, E>,
    ) -> Result<O, E>
    where
        Self: 'b,
    {
        let _ = stream;
        crate::host::HostAndDeviceConstRef::with_new(param, |const_ref| {
            inner.with(unsafe { crate::utils::r#async::AsyncProj::new(const_ref, None) })
        })
    }

    #[cfg(feature = "host")]
    fn with_async_as_ptx_jit<'stream, 'b, O>(
        _param: &Self::AsyncHostType<'stream, 'b>,
        _token: sealed::Token,
        inner: impl for<'p> FnOnce(Option<&'p NonNull<[u8]>>) -> O,
    ) -> O
    where
        Self: 'b,
    {
        inner(None)
    }

    #[cfg(feature = "host")]
    fn shared_layout_for_async<'stream, 'b>(
        _param: &Self::AsyncHostType<'stream, 'b>,
        _token: sealed::Token,
    ) -> Layout
    where
        Self: 'b,
    {
        Layout::new::<()>()
    }

    #[cfg(feature = "host")]
    fn async_to_ffi<'stream, 'b, E: From<rustacuda::error::CudaError>>(
        param: Self::AsyncHostType<'stream, 'b>,
        _token: sealed::Token,
    ) -> Result<Self::FfiType<'stream, 'b>, E>
    where
        Self: 'b,
    {
        let param = unsafe { param.unwrap_unchecked() };
        Ok(param.for_device())
    }

    #[cfg(feature = "device")]
    unsafe fn with_ffi_as_device<'short, O, const PARAM: usize>(
        param: Self::FfiType<'static, 'short>,
        inner: impl super::WithFfiAsDevice<Self, O>,
    ) -> O
    where
        Self: 'short,
    {
        let param = param.as_ref();

        inner.with(param)
    }
}
impl<
        'a,
        T: Sync + crate::safety::StackOnly + crate::safety::PortableBitSemantics + TypeGraphLayout,
    > sealed::Sealed for &'a PerThreadShallowCopy<T>
{
}

impl<
        'a,
        T: Sync + crate::safety::StackOnly + crate::safety::PortableBitSemantics + TypeGraphLayout,
    > CudaKernelParameter for &'a PtxJit<PerThreadShallowCopy<T>>
{
    #[cfg(feature = "host")]
    type AsyncHostType<'stream, 'b> =
        <&'a PerThreadShallowCopy<T> as CudaKernelParameter>::AsyncHostType<'stream, 'b> where Self: 'b;
    #[cfg(any(feature = "device", doc))]
    type DeviceType<'b> = <&'a PerThreadShallowCopy<T> as CudaKernelParameter>::DeviceType<'b> where Self: 'b;
    type FfiType<'stream, 'b> =
        <&'a PerThreadShallowCopy<T> as CudaKernelParameter>::FfiType<'stream, 'b> where Self: 'b;
    #[cfg(feature = "host")]
    type SyncHostType = <&'a PerThreadShallowCopy<T> as CudaKernelParameter>::SyncHostType;

    #[cfg(feature = "host")]
    fn with_new_async<'stream, 'b, O, E: From<rustacuda::error::CudaError>>(
        param: Self::SyncHostType,
        stream: crate::host::Stream<'stream>,
        #[cfg(not(doc))] inner: impl super::WithNewAsync<'stream, Self, O, E>,
        #[cfg(doc)] inner: impl FnOnce(Self::AsyncHostType<'stream, 'b>) -> Result<O, E>,
    ) -> Result<O, E>
    where
        Self: 'b,
    {
        let _ = stream;
        // FIXME: forward impl
        crate::host::HostAndDeviceConstRef::with_new(param, |const_ref| {
            inner.with(unsafe { crate::utils::r#async::AsyncProj::new(const_ref, None) })
        })
    }

    #[cfg(feature = "host")]
    fn with_async_as_ptx_jit<'stream, 'b, O>(
        param: &Self::AsyncHostType<'stream, 'b>,
        _token: sealed::Token,
        inner: impl for<'p> FnOnce(Option<&'p NonNull<[u8]>>) -> O,
    ) -> O
    where
        Self: 'b,
    {
        let param_ref = param.proj_ref();
        let param = unsafe { param_ref.unwrap_ref_unchecked() };
        inner(Some(&param_as_raw_bytes(param.for_host())))
    }

    #[cfg(feature = "host")]
    fn shared_layout_for_async<'stream, 'b>(
        _param: &Self::AsyncHostType<'stream, 'b>,
        _token: sealed::Token,
    ) -> Layout
    where
        Self: 'b,
    {
        Layout::new::<()>()
    }

    #[cfg(feature = "host")]
    fn async_to_ffi<'stream, 'b, E: From<rustacuda::error::CudaError>>(
        param: Self::AsyncHostType<'stream, 'b>,
        token: sealed::Token,
    ) -> Result<Self::FfiType<'stream, 'b>, E>
    where
        Self: 'b,
    {
        <&'a PerThreadShallowCopy<T> as CudaKernelParameter>::async_to_ffi(param, token)
    }

    #[cfg(feature = "device")]
    unsafe fn with_ffi_as_device<'short, O, const PARAM: usize>(
        param: Self::FfiType<'static, 'short>,
        inner: impl super::WithFfiAsDevice<Self, O>,
    ) -> O
    where
        Self: 'short,
    {
        emit_param_ptx_jit_marker::<_, PARAM>(param.as_ref());

        // FIXME: forward impl
        let param = param.as_ref();

        inner.with(param)
    }
}
impl<
        'a,
        T: Sync + crate::safety::StackOnly + crate::safety::PortableBitSemantics + TypeGraphLayout,
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
        T: Sync
            + crate::safety::StackOnly
            + crate::safety::PortableBitSemantics
            + TypeGraphLayout
            + InteriorMutableSync,
    > CudaKernelParameter for &'a ShallowInteriorMutable<T>
{
    #[cfg(feature = "host")]
    type AsyncHostType<'stream, 'b> = crate::utils::r#async::AsyncProj<
        'b,
        'stream,
        crate::host::HostAndDeviceConstRef<'b, T>
    > where Self: 'b;
    #[cfg(any(feature = "device", doc))]
    type DeviceType<'b> = &'b T where Self: 'b;
    type FfiType<'stream, 'b> = DeviceConstRef<'b, T> where Self: 'b;
    #[cfg(feature = "host")]
    /// The kernel takes a mutable borrow of the interior mutable data to ensure
    /// the interior mutability is limited to just this kernel invocation.
    type SyncHostType = &'a mut T;

    #[cfg(feature = "host")]
    fn with_new_async<'stream, 'b, O, E: From<rustacuda::error::CudaError>>(
        param: Self::SyncHostType,
        stream: crate::host::Stream<'stream>,
        #[cfg(not(doc))] inner: impl super::WithNewAsync<'stream, Self, O, E>,
        #[cfg(doc)] inner: impl FnOnce(Self::AsyncHostType<'stream, 'b>) -> Result<O, E>,
    ) -> Result<O, E>
    where
        Self: 'b,
    {
        let _ = stream;
        crate::host::HostAndDeviceMutRef::with_new(param, |mut_ref| {
            inner.with(unsafe { crate::utils::r#async::AsyncProj::new(mut_ref.as_ref(), None) })
        })
    }

    #[cfg(feature = "host")]
    fn with_async_as_ptx_jit<'stream, 'b, O>(
        _param: &Self::AsyncHostType<'stream, 'b>,
        _token: sealed::Token,
        inner: impl for<'p> FnOnce(Option<&'p NonNull<[u8]>>) -> O,
    ) -> O
    where
        Self: 'b,
    {
        inner(None)
    }

    #[cfg(feature = "host")]
    fn shared_layout_for_async<'stream, 'b>(
        _param: &Self::AsyncHostType<'stream, 'b>,
        _token: sealed::Token,
    ) -> Layout
    where
        Self: 'b,
    {
        Layout::new::<()>()
    }

    #[cfg(feature = "host")]
    fn async_to_ffi<'stream, 'b, E: From<rustacuda::error::CudaError>>(
        param: Self::AsyncHostType<'stream, 'b>,
        _token: sealed::Token,
    ) -> Result<Self::FfiType<'stream, 'b>, E>
    where
        Self: 'b,
    {
        let param = unsafe { param.unwrap_unchecked() };
        Ok(param.for_device())
    }

    #[cfg(feature = "device")]
    unsafe fn with_ffi_as_device<'short, O, const PARAM: usize>(
        param: Self::FfiType<'static, 'short>,
        inner: impl super::WithFfiAsDevice<Self, O>,
    ) -> O
    where
        Self: 'short,
    {
        let param = param.as_ref();

        inner.with(param)
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

pub struct DeepPerThreadBorrow<T: RustToCuda> {
    never: !,
    _marker: PhantomData<T>,
}

impl<T: RustToCuda> Deref for DeepPerThreadBorrow<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.never
    }
}

impl<
        T: Send
            + Clone
            + RustToCuda<CudaRepresentation: crate::safety::StackOnly, CudaAllocation: EmptyCudaAlloc>,
    > CudaKernelParameter for DeepPerThreadBorrow<T>
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
    > where Self: 'b;
    #[cfg(any(feature = "device", doc))]
    type DeviceType<'b> = T where Self: 'b;
    type FfiType<'stream, 'b> =
        DeviceOwnedRef<'b, DeviceAccessible<<T as RustToCuda>::CudaRepresentation>> where Self: 'b;
    #[cfg(feature = "host")]
    type SyncHostType = T;

    #[cfg(feature = "host")]
    fn with_new_async<'stream, 'b, O, E: From<rustacuda::error::CudaError>>(
        param: Self::SyncHostType,
        stream: crate::host::Stream<'stream>,
        #[cfg(not(doc))] inner: impl super::WithNewAsync<'stream, Self, O, E>,
        #[cfg(doc)] inner: impl FnOnce(Self::AsyncHostType<'stream, 'b>) -> Result<O, E>,
    ) -> Result<O, E>
    where
        Self: 'b,
    {
        crate::lend::LendToCuda::move_to_cuda(param, |param| inner.with(param.into_async(stream)))
    }

    #[cfg(feature = "host")]
    fn with_async_as_ptx_jit<'stream, 'b, O>(
        _param: &Self::AsyncHostType<'stream, 'b>,
        _token: sealed::Token,
        inner: impl for<'p> FnOnce(Option<&'p NonNull<[u8]>>) -> O,
    ) -> O
    where
        Self: 'b,
    {
        inner(None)
    }

    #[cfg(feature = "host")]
    fn shared_layout_for_async<'stream, 'b>(
        _param: &Self::AsyncHostType<'stream, 'b>,
        _token: sealed::Token,
    ) -> Layout
    where
        Self: 'b,
    {
        Layout::new::<()>()
    }

    #[cfg(feature = "host")]
    fn async_to_ffi<'stream, 'b, E: From<rustacuda::error::CudaError>>(
        param: Self::AsyncHostType<'stream, 'b>,
        _token: sealed::Token,
    ) -> Result<Self::FfiType<'stream, 'b>, E>
    where
        Self: 'b,
    {
        let (param, _completion): (_, Option<crate::utils::r#async::NoCompletion>) =
            unsafe { param.unwrap_unchecked()? };
        Ok(param.for_device())
    }

    #[cfg(feature = "device")]
    unsafe fn with_ffi_as_device<'short, O, const PARAM: usize>(
        param: Self::FfiType<'static, 'short>,
        inner: impl super::WithFfiAsDevice<Self, O>,
    ) -> O
    where
        Self: 'short,
    {
        unsafe {
            crate::lend::BorrowFromRust::with_moved_from_rust(param, |param| inner.with(param))
        }
    }
}
impl<
        T: Send
            + Clone
            + RustToCuda<CudaRepresentation: crate::safety::StackOnly, CudaAllocation: EmptyCudaAlloc>,
    > sealed::Sealed for DeepPerThreadBorrow<T>
{
}

impl<'a, T: Sync + RustToCuda> CudaKernelParameter for &'a DeepPerThreadBorrow<T> {
    #[cfg(feature = "host")]
    type AsyncHostType<'stream, 'b> = crate::utils::r#async::AsyncProj<
        'b,
        'stream,
        crate::host::HostAndDeviceConstRef<
            'b,
            DeviceAccessible<<T as RustToCuda>::CudaRepresentation>,
        >,
    > where Self: 'b;
    #[cfg(any(feature = "device", doc))]
    type DeviceType<'b> = &'b T where Self: 'b;
    type FfiType<'stream, 'b> =
        DeviceConstRef<'b, DeviceAccessible<<T as RustToCuda>::CudaRepresentation>> where Self: 'b;
    #[cfg(feature = "host")]
    type SyncHostType = &'a T;

    #[cfg(feature = "host")]
    fn with_new_async<'stream, 'b, O, E: From<rustacuda::error::CudaError>>(
        param: Self::SyncHostType,
        stream: crate::host::Stream<'stream>,
        #[cfg(not(doc))] inner: impl super::WithNewAsync<'stream, Self, O, E>,
        #[cfg(doc)] inner: impl FnOnce(Self::AsyncHostType<'stream, 'b>) -> Result<O, E>,
    ) -> Result<O, E>
    where
        Self: 'b,
    {
        let _ = stream;
        crate::lend::LendToCuda::lend_to_cuda(param, |param| {
            inner.with(unsafe { crate::utils::r#async::AsyncProj::new(param, None) })
        })
    }

    #[cfg(feature = "host")]
    fn with_async_as_ptx_jit<'stream, 'b, O>(
        _param: &Self::AsyncHostType<'stream, 'b>,
        _token: sealed::Token,
        inner: impl for<'p> FnOnce(Option<&'p NonNull<[u8]>>) -> O,
    ) -> O
    where
        Self: 'b,
    {
        inner(None)
    }

    #[cfg(feature = "host")]
    fn shared_layout_for_async<'stream, 'b>(
        _param: &Self::AsyncHostType<'stream, 'b>,
        _token: sealed::Token,
    ) -> Layout
    where
        Self: 'b,
    {
        Layout::new::<()>()
    }

    #[cfg(feature = "host")]
    fn async_to_ffi<'stream, 'b, E: From<rustacuda::error::CudaError>>(
        param: Self::AsyncHostType<'stream, 'b>,
        _token: sealed::Token,
    ) -> Result<Self::FfiType<'stream, 'b>, E>
    where
        Self: 'b,
    {
        let param = unsafe { param.unwrap_unchecked() };
        Ok(param.for_device())
    }

    #[cfg(feature = "device")]
    unsafe fn with_ffi_as_device<'short, O, const PARAM: usize>(
        param: Self::FfiType<'static, 'short>,
        inner: impl super::WithFfiAsDevice<Self, O>,
    ) -> O
    where
        Self: 'short,
    {
        unsafe {
            crate::lend::BorrowFromRust::with_borrow_from_rust(param, |param| inner.with(param))
        }
    }
}
impl<'a, T: Sync + RustToCuda> sealed::Sealed for &'a DeepPerThreadBorrow<T> {}

impl<'a, T: Sync + RustToCuda + SafeMutableAliasing> CudaKernelParameter
    for &'a mut DeepPerThreadBorrow<T>
{
    #[cfg(feature = "host")]
    type AsyncHostType<'stream, 'b> = crate::utils::r#async::AsyncProj<
        'b,
        'stream,
        crate::host::HostAndDeviceMutRef<
            'b,
            DeviceAccessible<<T as RustToCuda>::CudaRepresentation>,
        >,
    > where Self: 'b;
    #[cfg(any(feature = "device", doc))]
    type DeviceType<'b> = &'b mut T where Self: 'b;
    type FfiType<'stream, 'b> =
        DeviceMutRef<'b, DeviceAccessible<<T as RustToCuda>::CudaRepresentation>> where Self: 'b;
    #[cfg(feature = "host")]
    type SyncHostType = &'a mut T;

    #[cfg(feature = "host")]
    fn with_new_async<'stream, 'b, O, E: From<rustacuda::error::CudaError>>(
        param: Self::SyncHostType,
        stream: crate::host::Stream<'stream>,
        #[cfg(not(doc))] inner: impl super::WithNewAsync<'stream, Self, O, E>,
        #[cfg(doc)] inner: impl FnOnce(Self::AsyncHostType<'stream, 'b>) -> Result<O, E>,
    ) -> Result<O, E>
    where
        Self: 'b,
    {
        crate::lend::LendToCuda::lend_to_cuda_mut(param, |param| {
            // FIXME: express the same with param.into_async(stream).as_mut()
            let _ = stream;
            inner.with({
                // Safety: this projection cannot be moved to a different stream
                //         without first exiting lend_to_cuda_mut and synchronizing
                unsafe { crate::utils::r#async::AsyncProj::new(param.into_mut(), None) }
            })
        })
    }

    #[cfg(feature = "host")]
    fn with_async_as_ptx_jit<'stream, 'b, O>(
        _param: &Self::AsyncHostType<'stream, 'b>,
        _token: sealed::Token,
        inner: impl for<'p> FnOnce(Option<&'p NonNull<[u8]>>) -> O,
    ) -> O
    where
        Self: 'b,
    {
        inner(None)
    }

    #[cfg(feature = "host")]
    fn shared_layout_for_async<'stream, 'b>(
        _param: &Self::AsyncHostType<'stream, 'b>,
        _token: sealed::Token,
    ) -> Layout
    where
        Self: 'b,
    {
        Layout::new::<()>()
    }

    #[cfg(feature = "host")]
    fn async_to_ffi<'stream, 'b, E: From<rustacuda::error::CudaError>>(
        mut param: Self::AsyncHostType<'stream, 'b>,
        _token: sealed::Token,
    ) -> Result<Self::FfiType<'stream, 'b>, E>
    where
        Self: 'b,
    {
        param.record_mut_use()?;
        let mut param = unsafe { param.unwrap_unchecked() };
        Ok(param.for_device())
    }

    #[cfg(feature = "device")]
    unsafe fn with_ffi_as_device<'short, O, const PARAM: usize>(
        param: Self::FfiType<'static, 'short>,
        inner: impl super::WithFfiAsDevice<Self, O>,
    ) -> O
    where
        Self: 'short,
    {
        unsafe {
            crate::lend::BorrowFromRust::with_borrow_from_rust_mut(param, |param| inner.with(param))
        }
    }
}
impl<'a, T: Sync + RustToCuda + SafeMutableAliasing> sealed::Sealed
    for &'a mut DeepPerThreadBorrow<T>
{
}

impl<
        T: Send
            + Clone
            + RustToCuda<CudaRepresentation: crate::safety::StackOnly, CudaAllocation: EmptyCudaAlloc>,
    > CudaKernelParameter for PtxJit<DeepPerThreadBorrow<T>>
{
    #[cfg(feature = "host")]
    type AsyncHostType<'stream, 'b> =
        <DeepPerThreadBorrow<T> as CudaKernelParameter>::AsyncHostType<'stream, 'b> where Self: 'b;
    #[cfg(any(feature = "device", doc))]
    type DeviceType<'b> = <DeepPerThreadBorrow<T> as CudaKernelParameter>::DeviceType<'b> where Self: 'b;
    type FfiType<'stream, 'b> =
        <DeepPerThreadBorrow<T> as CudaKernelParameter>::FfiType<'stream, 'b> where Self: 'b;
    #[cfg(feature = "host")]
    type SyncHostType = <DeepPerThreadBorrow<T> as CudaKernelParameter>::SyncHostType;

    #[cfg(feature = "host")]
    fn with_new_async<'stream, 'b, O, E: From<rustacuda::error::CudaError>>(
        param: Self::SyncHostType,
        stream: crate::host::Stream<'stream>,
        #[cfg(not(doc))] inner: impl super::WithNewAsync<'stream, Self, O, E>,
        #[cfg(doc)] inner: impl FnOnce(Self::AsyncHostType<'stream, 'b>) -> Result<O, E>,
    ) -> Result<O, E>
    where
        Self: 'b,
    {
        // FIXME: forward impl
        crate::lend::LendToCuda::move_to_cuda(param, |param| inner.with(param.into_async(stream)))
    }

    #[cfg(feature = "host")]
    fn with_async_as_ptx_jit<'stream, 'b, O>(
        param: &Self::AsyncHostType<'stream, 'b>,
        _token: sealed::Token,
        inner: impl for<'p> FnOnce(Option<&'p NonNull<[u8]>>) -> O,
    ) -> O
    where
        Self: 'b,
    {
        let param = unsafe { param.as_ref().unwrap_unchecked() };
        inner(Some(&param_as_raw_bytes(param.for_host())))
    }

    #[cfg(feature = "host")]
    fn async_to_ffi<'stream, 'b, E: From<rustacuda::error::CudaError>>(
        param: Self::AsyncHostType<'stream, 'b>,
        token: sealed::Token,
    ) -> Result<Self::FfiType<'stream, 'b>, E>
    where
        Self: 'b,
    {
        <DeepPerThreadBorrow<T> as CudaKernelParameter>::async_to_ffi(param, token)
    }

    #[cfg(feature = "host")]
    fn shared_layout_for_async<'stream, 'b>(
        _param: &Self::AsyncHostType<'stream, 'b>,
        _token: sealed::Token,
    ) -> Layout
    where
        Self: 'b,
    {
        Layout::new::<()>()
    }

    #[cfg(feature = "device")]
    unsafe fn with_ffi_as_device<'short, O, const PARAM: usize>(
        param: Self::FfiType<'static, 'short>,
        inner: impl super::WithFfiAsDevice<Self, O>,
    ) -> O
    where
        Self: 'short,
    {
        emit_param_ptx_jit_marker::<_, PARAM>(param.as_ref());

        // FIXME: forward impl
        unsafe {
            crate::lend::BorrowFromRust::with_moved_from_rust(param, |param| inner.with(param))
        }
    }
}
impl<
        T: Send
            + Clone
            + RustToCuda<CudaRepresentation: crate::safety::StackOnly, CudaAllocation: EmptyCudaAlloc>,
    > sealed::Sealed for PtxJit<DeepPerThreadBorrow<T>>
{
}

impl<'a, T: Sync + RustToCuda> CudaKernelParameter for &'a PtxJit<DeepPerThreadBorrow<T>> {
    #[cfg(feature = "host")]
    type AsyncHostType<'stream, 'b> =
        <&'a DeepPerThreadBorrow<T> as CudaKernelParameter>::AsyncHostType<'stream, 'b> where Self: 'b;
    #[cfg(any(feature = "device", doc))]
    type DeviceType<'b> = <&'a DeepPerThreadBorrow<T> as CudaKernelParameter>::DeviceType<'b> where Self: 'b;
    type FfiType<'stream, 'b> =
        <&'a DeepPerThreadBorrow<T> as CudaKernelParameter>::FfiType<'stream, 'b> where Self: 'b;
    #[cfg(feature = "host")]
    type SyncHostType = <&'a DeepPerThreadBorrow<T> as CudaKernelParameter>::SyncHostType;

    #[cfg(feature = "host")]
    fn with_new_async<'stream, 'b, O, E: From<rustacuda::error::CudaError>>(
        param: Self::SyncHostType,
        stream: crate::host::Stream<'stream>,
        #[cfg(not(doc))] inner: impl super::WithNewAsync<'stream, Self, O, E>,
        #[cfg(doc)] inner: impl FnOnce(Self::AsyncHostType<'stream, 'b>) -> Result<O, E>,
    ) -> Result<O, E>
    where
        Self: 'b,
    {
        // FIXME: forward impl
        let _ = stream;
        crate::lend::LendToCuda::lend_to_cuda(param, |param| {
            inner.with(unsafe { crate::utils::r#async::AsyncProj::new(param, None) })
        })
    }

    #[cfg(feature = "host")]
    fn with_async_as_ptx_jit<'stream, 'b, O>(
        param: &Self::AsyncHostType<'stream, 'b>,
        _token: sealed::Token,
        inner: impl for<'p> FnOnce(Option<&'p NonNull<[u8]>>) -> O,
    ) -> O
    where
        Self: 'b,
    {
        let param_ref = param.proj_ref();
        let param = unsafe { param_ref.unwrap_unchecked() };
        inner(Some(&param_as_raw_bytes(param.for_host())))
    }

    #[cfg(feature = "host")]
    fn shared_layout_for_async<'stream, 'b>(
        _param: &Self::AsyncHostType<'stream, 'b>,
        _token: sealed::Token,
    ) -> Layout
    where
        Self: 'b,
    {
        Layout::new::<()>()
    }

    #[cfg(feature = "host")]
    fn async_to_ffi<'stream, 'b, E: From<rustacuda::error::CudaError>>(
        param: Self::AsyncHostType<'stream, 'b>,
        token: sealed::Token,
    ) -> Result<Self::FfiType<'stream, 'b>, E>
    where
        Self: 'b,
    {
        <&'a DeepPerThreadBorrow<T> as CudaKernelParameter>::async_to_ffi(param, token)
    }

    #[cfg(feature = "device")]
    unsafe fn with_ffi_as_device<'short, O, const PARAM: usize>(
        param: Self::FfiType<'static, 'short>,
        inner: impl super::WithFfiAsDevice<Self, O>,
    ) -> O
    where
        Self: 'short,
    {
        emit_param_ptx_jit_marker::<_, PARAM>(param.as_ref());

        // FIXME: forward impl
        unsafe {
            crate::lend::BorrowFromRust::with_borrow_from_rust(param, |param| inner.with(param))
        }
    }
}
impl<'a, T: Sync + RustToCuda> sealed::Sealed for &'a PtxJit<DeepPerThreadBorrow<T>> {}

impl<'a, T: Sync + RustToCuda + SafeMutableAliasing> CudaKernelParameter
    for &'a mut PtxJit<DeepPerThreadBorrow<T>>
{
    #[cfg(feature = "host")]
    type AsyncHostType<'stream, 'b> =
        <&'a mut DeepPerThreadBorrow<T> as CudaKernelParameter>::AsyncHostType<'stream, 'b> where Self: 'b;
    #[cfg(any(feature = "device", doc))]
    type DeviceType<'b> = <&'a mut DeepPerThreadBorrow<T> as CudaKernelParameter>::DeviceType<'b> where Self: 'b;
    type FfiType<'stream, 'b> =
        <&'a mut DeepPerThreadBorrow<T> as CudaKernelParameter>::FfiType<'stream, 'b> where Self: 'b;
    #[cfg(feature = "host")]
    type SyncHostType = <&'a mut DeepPerThreadBorrow<T> as CudaKernelParameter>::SyncHostType;

    #[cfg(feature = "host")]
    fn with_new_async<'stream, 'b, O, E: From<rustacuda::error::CudaError>>(
        param: Self::SyncHostType,
        stream: crate::host::Stream<'stream>,
        #[cfg(not(doc))] inner: impl super::WithNewAsync<'stream, Self, O, E>,
        #[cfg(doc)] inner: impl FnOnce(Self::AsyncHostType<'stream, 'b>) -> Result<O, E>,
    ) -> Result<O, E>
    where
        Self: 'b,
    {
        // FIXME: forward impl
        crate::lend::LendToCuda::lend_to_cuda_mut(param, |param| {
            // FIXME: express the same with param.as_async(stream).as_mut()
            let _ = stream;
            inner.with({
                // Safety: this projection cannot be moved to a different stream
                //         without first exiting lend_to_cuda_mut and synchronizing
                unsafe { crate::utils::r#async::AsyncProj::new(param.into_mut(), None) }
            })
        })
    }

    #[cfg(feature = "host")]
    fn with_async_as_ptx_jit<'stream, 'b, O>(
        param: &Self::AsyncHostType<'stream, 'b>,
        _token: sealed::Token,
        inner: impl for<'p> FnOnce(Option<&'p NonNull<[u8]>>) -> O,
    ) -> O
    where
        Self: 'b,
    {
        let param_ref = param.proj_ref();
        let param = unsafe { param_ref.unwrap_unchecked() };
        inner(Some(&param_as_raw_bytes(param.for_host())))
    }

    #[cfg(feature = "host")]
    fn shared_layout_for_async<'stream, 'b>(
        _param: &Self::AsyncHostType<'stream, 'b>,
        _token: sealed::Token,
    ) -> Layout
    where
        Self: 'b,
    {
        Layout::new::<()>()
    }

    #[cfg(feature = "host")]
    fn async_to_ffi<'stream, 'b, E: From<rustacuda::error::CudaError>>(
        param: Self::AsyncHostType<'stream, 'b>,
        token: sealed::Token,
    ) -> Result<Self::FfiType<'stream, 'b>, E>
    where
        Self: 'b,
    {
        <&'a mut DeepPerThreadBorrow<T> as CudaKernelParameter>::async_to_ffi(param, token)
    }

    #[cfg(feature = "device")]
    unsafe fn with_ffi_as_device<'short, O, const PARAM: usize>(
        param: Self::FfiType<'static, 'short>,
        inner: impl super::WithFfiAsDevice<Self, O>,
    ) -> O
    where
        Self: 'short,
    {
        emit_param_ptx_jit_marker::<_, PARAM>(param.as_ref());

        // FIXME: forward impl
        unsafe {
            crate::lend::BorrowFromRust::with_borrow_from_rust_mut(param, |param| inner.with(param))
        }
    }
}
impl<'a, T: Sync + RustToCuda + SafeMutableAliasing> sealed::Sealed
    for &'a mut PtxJit<DeepPerThreadBorrow<T>>
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
    type AsyncHostType<'stream, 'b> = &'b mut crate::utils::shared::ThreadBlockShared<T> where Self: 'b;
    #[cfg(any(feature = "device", doc))]
    type DeviceType<'b> = &'b mut crate::utils::shared::ThreadBlockShared<T> where Self: 'b;
    type FfiType<'stream, 'b> = private_shared::ThreadBlockSharedFfi<T> where Self: 'b;
    #[cfg(feature = "host")]
    type SyncHostType = Self;

    #[cfg(feature = "host")]
    fn with_new_async<'stream, 'b, O, E: From<rustacuda::error::CudaError>>(
        param: Self::SyncHostType,
        _stream: crate::host::Stream<'stream>,
        #[cfg(not(doc))] inner: impl super::WithNewAsync<'stream, Self, O, E>,
        #[cfg(doc)] inner: impl FnOnce(Self::AsyncHostType<'stream, 'b>) -> Result<O, E>,
    ) -> Result<O, E>
    where
        Self: 'b,
    {
        inner.with(param)
    }

    #[cfg(feature = "host")]
    fn with_async_as_ptx_jit<'stream, 'b, O>(
        _param: &Self::AsyncHostType<'stream, 'b>,
        _token: sealed::Token,
        inner: impl for<'p> FnOnce(Option<&'p NonNull<[u8]>>) -> O,
    ) -> O
    where
        Self: 'b,
    {
        inner(None)
    }

    #[cfg(feature = "host")]
    fn shared_layout_for_async<'stream, 'b>(
        _param: &Self::AsyncHostType<'stream, 'b>,
        _token: sealed::Token,
    ) -> Layout
    where
        Self: 'b,
    {
        Layout::new::<()>()
    }

    #[cfg(feature = "host")]
    fn async_to_ffi<'stream, 'b, E: From<rustacuda::error::CudaError>>(
        _param: Self::AsyncHostType<'stream, 'b>,
        _token: sealed::Token,
    ) -> Result<Self::FfiType<'stream, 'b>, E>
    where
        Self: 'b,
    {
        Ok(private_shared::ThreadBlockSharedFfi {
            _dummy: [],
            _marker: PhantomData::<T>,
        })
    }

    #[cfg(feature = "device")]
    #[expect(clippy::inline_always)]
    #[inline(always)]
    unsafe fn with_ffi_as_device<'short, O, const PARAM: usize>(
        _param: Self::FfiType<'static, 'short>,
        inner: impl super::WithFfiAsDevice<Self, O>,
    ) -> O
    where
        Self: 'short,
    {
        let mut param = crate::utils::shared::ThreadBlockShared::new_uninit();

        inner.with(&mut param)
    }
}
impl<'a, T: 'static> sealed::Sealed for &'a mut crate::utils::shared::ThreadBlockShared<T> {}

impl<'a, T: 'static + PortableBitSemantics + TypeGraphLayout> CudaKernelParameter
    for &'a mut crate::utils::shared::ThreadBlockSharedSlice<T>
{
    #[cfg(feature = "host")]
    type AsyncHostType<'stream, 'b> = &'b mut crate::utils::shared::ThreadBlockSharedSlice<T> where Self: 'b;
    #[cfg(any(feature = "device", doc))]
    type DeviceType<'b> = &'b mut crate::utils::shared::ThreadBlockSharedSlice<T> where Self: 'b;
    type FfiType<'stream, 'b> = private_shared::ThreadBlockSharedSliceFfi<T> where Self: 'b;
    #[cfg(feature = "host")]
    type SyncHostType = Self;

    #[cfg(feature = "host")]
    fn with_new_async<'stream, 'b, O, E: From<rustacuda::error::CudaError>>(
        param: Self::SyncHostType,
        _stream: crate::host::Stream<'stream>,
        #[cfg(not(doc))] inner: impl super::WithNewAsync<'stream, Self, O, E>,
        #[cfg(doc)] inner: impl FnOnce(Self::AsyncHostType<'stream, 'b>) -> Result<O, E>,
    ) -> Result<O, E>
    where
        Self: 'b,
    {
        inner.with(param)
    }

    #[cfg(feature = "host")]
    fn with_async_as_ptx_jit<'stream, 'b, O>(
        _param: &Self::AsyncHostType<'stream, 'b>,
        _token: sealed::Token,
        inner: impl for<'p> FnOnce(Option<&'p NonNull<[u8]>>) -> O,
    ) -> O
    where
        Self: 'b,
    {
        inner(None)
    }

    #[cfg(feature = "host")]
    fn shared_layout_for_async<'stream, 'b>(
        param: &Self::AsyncHostType<'stream, 'b>,
        _token: sealed::Token,
    ) -> Layout
    where
        Self: 'b,
    {
        param.layout()
    }

    #[cfg(feature = "host")]
    fn async_to_ffi<'stream, 'b, E: From<rustacuda::error::CudaError>>(
        param: Self::AsyncHostType<'stream, 'b>,
        _token: sealed::Token,
    ) -> Result<Self::FfiType<'stream, 'b>, E>
    where
        Self: 'b,
    {
        Ok(private_shared::ThreadBlockSharedSliceFfi {
            len: param.len(),
            _marker: [],
        })
    }

    #[cfg(feature = "device")]
    #[expect(clippy::inline_always)]
    #[inline(always)]
    unsafe fn with_ffi_as_device<'short, O, const PARAM: usize>(
        param: Self::FfiType<'static, 'short>,
        inner: impl super::WithFfiAsDevice<Self, O>,
    ) -> O
    where
        Self: 'short,
    {
        unsafe {
            crate::utils::shared::ThreadBlockSharedSlice::with_uninit_for_len(param.len, |param| {
                inner.with(param)
            })
        }
    }
}
impl<'a, T: 'static + PortableBitSemantics + TypeGraphLayout> sealed::Sealed
    for &'a mut crate::utils::shared::ThreadBlockSharedSlice<T>
{
}
