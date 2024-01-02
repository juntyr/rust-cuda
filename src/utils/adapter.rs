#![allow(clippy::trait_duplication_in_bounds)]

use const_type_layout::{TypeGraphLayout, TypeLayout};

use crate::{
    alloc::NoCudaAlloc,
    lend::{CudaAsRust, RustToCuda, RustToCudaAsync},
    safety::PortableBitSemantics,
};

#[cfg(any(feature = "host", feature = "device"))]
use crate::utils::ffi::DeviceAccessible;

#[cfg(feature = "host")]
use crate::alloc::{CombinedCudaAlloc, CudaAlloc};

#[derive(Copy, Clone, Debug, TypeLayout)]
#[repr(transparent)]
pub struct RustToCudaWithPortableBitCopySemantics<T: Copy + PortableBitSemantics + TypeGraphLayout>(
    T,
);

impl<T: Copy + PortableBitSemantics + TypeGraphLayout> From<T>
    for RustToCudaWithPortableBitCopySemantics<T>
{
    fn from(value: T) -> Self {
        Self(value)
    }
}

impl<T: Copy + PortableBitSemantics + TypeGraphLayout> RustToCudaWithPortableBitCopySemantics<T> {
    #[must_use]
    pub const fn from_copy(value: &T) -> Self {
        Self(*value)
    }

    #[must_use]
    pub const fn into_inner(self) -> T {
        self.0
    }

    #[must_use]
    pub const fn from_ref(reference: &T) -> &Self {
        // Safety: [`RustToCudaWithPortableBitCopySemantics`] is a transparent newtype
        //         around `T`
        unsafe { &*core::ptr::from_ref(reference).cast() }
    }

    #[must_use]
    pub const fn into_ref(&self) -> &T {
        // Safety: [`RustToCudaWithPortableBitCopySemantics`] is a transparent newtype
        //         around `T`
        unsafe { &*core::ptr::from_ref(self).cast() }
    }

    #[must_use]
    pub fn from_mut(reference: &mut T) -> &mut Self {
        // Safety: [`RustToCudaWithPortableBitCopySemantics`] is a transparent newtype
        //         around `T`
        unsafe { &mut *core::ptr::from_mut(reference).cast() }
    }

    #[must_use]
    pub fn into_mut(&mut self) -> &mut T {
        // Safety: [`RustToCudaWithPortableBitCopySemantics`] is a transparent newtype
        //         around `T`
        unsafe { &mut *core::ptr::from_mut(self).cast() }
    }

    #[must_use]
    pub const fn from_slice(slice: &[T]) -> &[Self] {
        // Safety: [`RustToCudaWithPortableBitCopySemantics`] is a transparent newtype
        //         around `T`
        unsafe { core::slice::from_raw_parts(slice.as_ptr().cast(), slice.len()) }
    }

    #[must_use]
    pub const fn into_slice(slice: &[Self]) -> &[T] {
        // Safety: [`RustToCudaWithPortableBitCopySemantics`] is a transparent newtype
        //         around `T`
        unsafe { core::slice::from_raw_parts(slice.as_ptr().cast(), slice.len()) }
    }

    #[must_use]
    pub fn from_mut_slice(slice: &mut [T]) -> &mut [Self] {
        // Safety: [`RustToCudaWithPortableBitCopySemantics`] is a transparent newtype
        //         around `T`
        unsafe { core::slice::from_raw_parts_mut(slice.as_mut_ptr().cast(), slice.len()) }
    }

    #[must_use]
    pub fn into_mut_slice(slice: &mut [Self]) -> &mut [T] {
        // Safety: [`RustToCudaWithPortableBitCopySemantics`] is a transparent newtype
        //         around `T`
        unsafe { core::slice::from_raw_parts_mut(slice.as_mut_ptr().cast(), slice.len()) }
    }
}

unsafe impl<T: Copy + PortableBitSemantics + TypeGraphLayout> RustToCuda
    for RustToCudaWithPortableBitCopySemantics<T>
{
    type CudaAllocation = NoCudaAlloc;
    type CudaRepresentation = Self;

    #[cfg(feature = "host")]
    #[allow(clippy::type_complexity)]
    unsafe fn borrow<A: CudaAlloc>(
        &self,
        alloc: A,
    ) -> rustacuda::error::CudaResult<(
        DeviceAccessible<Self::CudaRepresentation>,
        CombinedCudaAlloc<Self::CudaAllocation, A>,
    )> {
        let alloc = CombinedCudaAlloc::new(NoCudaAlloc, alloc);
        Ok((DeviceAccessible::from(*self), alloc))
    }

    #[cfg(feature = "host")]
    unsafe fn restore<A: CudaAlloc>(
        &mut self,
        alloc: CombinedCudaAlloc<Self::CudaAllocation, A>,
    ) -> rustacuda::error::CudaResult<A> {
        let (_alloc_front, alloc_tail): (NoCudaAlloc, A) = alloc.split();

        Ok(alloc_tail)
    }
}

unsafe impl<T: Copy + PortableBitSemantics + TypeGraphLayout> RustToCudaAsync
    for RustToCudaWithPortableBitCopySemantics<T>
{
    type CudaAllocationAsync = NoCudaAlloc;

    #[cfg(feature = "host")]
    #[allow(clippy::type_complexity)]
    unsafe fn borrow_async<'stream, A: CudaAlloc>(
        &self,
        alloc: A,
        stream: &'stream rustacuda::stream::Stream,
    ) -> rustacuda::error::CudaResult<(
        crate::utils::r#async::Async<'_, 'stream, DeviceAccessible<Self::CudaRepresentation>>,
        CombinedCudaAlloc<Self::CudaAllocation, A>,
    )> {
        let alloc = CombinedCudaAlloc::new(NoCudaAlloc, alloc);
        Ok((
            crate::utils::r#async::Async::ready(DeviceAccessible::from(*self), stream),
            alloc,
        ))
    }

    #[cfg(feature = "host")]
    unsafe fn restore_async<'a, 'stream, A: CudaAlloc, O>(
        this: owning_ref::BoxRefMut<'a, O, Self>,
        alloc: CombinedCudaAlloc<Self::CudaAllocation, A>,
        stream: &'stream rustacuda::stream::Stream,
    ) -> rustacuda::error::CudaResult<(
        crate::utils::r#async::Async<
            'a,
            'stream,
            owning_ref::BoxRefMut<'a, O, Self>,
            crate::utils::r#async::CompletionFnMut<'a, Self>,
        >,
        A,
    )> {
        let (_alloc_front, alloc_tail): (NoCudaAlloc, A) = alloc.split();

        let r#async = crate::utils::r#async::Async::<
            _,
            crate::utils::r#async::CompletionFnMut<'a, Self>,
        >::pending(this, stream, Box::new(|_this| Ok(())))?;

        Ok((r#async, alloc_tail))
    }
}

unsafe impl<T: Copy + PortableBitSemantics + TypeGraphLayout> CudaAsRust
    for RustToCudaWithPortableBitCopySemantics<T>
{
    type RustRepresentation = Self;

    #[cfg(feature = "device")]
    unsafe fn as_rust(this: &DeviceAccessible<Self>) -> Self::RustRepresentation {
        let mut uninit = core::mem::MaybeUninit::uninit();
        core::ptr::copy_nonoverlapping(&**this, uninit.as_mut_ptr(), 1);
        uninit.assume_init()
    }
}

#[allow(clippy::module_name_repetitions)]
#[derive(Copy, Clone, Debug, TypeLayout)]
#[repr(transparent)]
pub struct DeviceCopyWithPortableBitSemantics<T: PortableBitSemantics + TypeGraphLayout>(T);

unsafe impl<T: PortableBitSemantics + TypeGraphLayout> rustacuda_core::DeviceCopy
    for DeviceCopyWithPortableBitSemantics<T>
{
}

impl<T: PortableBitSemantics + TypeGraphLayout> From<T> for DeviceCopyWithPortableBitSemantics<T> {
    fn from(value: T) -> Self {
        Self(value)
    }
}

impl<T: PortableBitSemantics + TypeGraphLayout> DeviceCopyWithPortableBitSemantics<T> {
    #[must_use]
    pub fn into_inner(self) -> T {
        self.0
    }

    #[must_use]
    pub const fn from_ref(reference: &T) -> &Self {
        // Safety: [`DeviceCopyWithPortableBitSemantics`] is a transparent newtype
        //         around `T`
        unsafe { &*core::ptr::from_ref(reference).cast() }
    }

    #[must_use]
    pub const fn into_ref(&self) -> &T {
        // Safety: [`DeviceCopyWithPortableBitSemantics`] is a transparent newtype
        //         around `T`
        unsafe { &*core::ptr::from_ref(self).cast() }
    }

    #[must_use]
    pub fn from_mut(reference: &mut T) -> &mut Self {
        // Safety: [`DeviceCopyWithPortableBitSemantics`] is a transparent newtype
        //         around `T`
        unsafe { &mut *core::ptr::from_mut(reference).cast() }
    }

    #[must_use]
    pub fn into_mut(&mut self) -> &mut T {
        // Safety: [`DeviceCopyWithPortableBitSemantics`] is a transparent newtype
        //         around `T`
        unsafe { &mut *core::ptr::from_mut(self).cast() }
    }

    #[must_use]
    pub const fn from_slice(slice: &[T]) -> &[Self] {
        // Safety: [`DeviceCopyWithPortableBitSemantics`] is a transparent newtype
        //         around `T`
        unsafe { core::slice::from_raw_parts(slice.as_ptr().cast(), slice.len()) }
    }

    #[must_use]
    pub const fn into_slice(slice: &[Self]) -> &[T] {
        // Safety: [`DeviceCopyWithPortableBitSemantics`] is a transparent newtype
        //         around `T`
        unsafe { core::slice::from_raw_parts(slice.as_ptr().cast(), slice.len()) }
    }

    #[must_use]
    pub fn from_mut_slice(slice: &mut [T]) -> &mut [Self] {
        // Safety: [`DeviceCopyWithPortableBitSemantics`] is a transparent newtype
        //         around `T`
        unsafe { core::slice::from_raw_parts_mut(slice.as_mut_ptr().cast(), slice.len()) }
    }

    #[must_use]
    pub fn into_mut_slice(slice: &mut [Self]) -> &mut [T] {
        // Safety: [`DeviceCopyWithPortableBitSemantics`] is a transparent newtype
        //         around `T`
        unsafe { core::slice::from_raw_parts_mut(slice.as_mut_ptr().cast(), slice.len()) }
    }
}
