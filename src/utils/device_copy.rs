#![allow(clippy::trait_duplication_in_bounds)]

use const_type_layout::{TypeGraphLayout, TypeLayout};

use crate::{
    alloc::NoCudaAlloc,
    lend::{CudaAsRust, RustToCuda, RustToCudaAsync},
    safety::SafeDeviceCopy,
};

#[cfg(any(feature = "host", feature = "device"))]
use crate::utils::ffi::DeviceAccessible;

#[cfg(feature = "host")]
use crate::alloc::{CombinedCudaAlloc, CudaAlloc};

#[derive(Copy, Clone, Debug, TypeLayout)]
#[repr(transparent)]
pub struct SafeDeviceCopyWrapper<T>(T)
where
    T: SafeDeviceCopy + TypeGraphLayout;

unsafe impl<T: SafeDeviceCopy + TypeGraphLayout> rustacuda_core::DeviceCopy
    for SafeDeviceCopyWrapper<T>
{
}

impl<T: SafeDeviceCopy + TypeGraphLayout> From<T> for SafeDeviceCopyWrapper<T> {
    fn from(value: T) -> Self {
        Self(value)
    }
}

impl<T: SafeDeviceCopy + TypeGraphLayout> SafeDeviceCopyWrapper<T> {
    #[must_use]
    pub fn into_inner(self) -> T {
        self.0
    }

    #[must_use]
    pub const fn from_ref(reference: &T) -> &Self {
        // Safety: [`SafeDeviceCopyWrapper`] is a transparent newtype around [`T`]
        unsafe { &*(reference as *const T).cast() }
    }

    #[must_use]
    pub const fn into_ref(&self) -> &T {
        // Safety: [`SafeDeviceCopyWrapper`] is a transparent newtype around [`T`]
        unsafe { &*(self as *const Self).cast() }
    }

    #[must_use]
    pub fn from_mut(reference: &mut T) -> &mut Self {
        // Safety: [`SafeDeviceCopyWrapper`] is a transparent newtype around [`T`]
        unsafe { &mut *(reference as *mut T).cast() }
    }

    #[must_use]
    pub fn into_mut(&mut self) -> &mut T {
        // Safety: [`SafeDeviceCopyWrapper`] is a transparent newtype around [`T`]
        unsafe { &mut *(self as *mut Self).cast() }
    }

    #[must_use]
    pub const fn from_slice(slice: &[T]) -> &[Self] {
        // Safety: [`SafeDeviceCopyWrapper`] is a transparent newtype around [`T`]
        unsafe { core::slice::from_raw_parts(slice.as_ptr().cast(), slice.len()) }
    }

    #[must_use]
    pub const fn into_slice(slice: &[Self]) -> &[T] {
        // Safety: [`SafeDeviceCopyWrapper`] is a transparent newtype around [`T`]
        unsafe { core::slice::from_raw_parts(slice.as_ptr().cast(), slice.len()) }
    }

    #[must_use]
    pub fn from_mut_slice(slice: &mut [T]) -> &mut [Self] {
        // Safety: [`SafeDeviceCopyWrapper`] is a transparent newtype around [`T`]
        unsafe { core::slice::from_raw_parts_mut(slice.as_mut_ptr().cast(), slice.len()) }
    }

    #[must_use]
    pub fn into_mut_slice(slice: &mut [Self]) -> &mut [T] {
        // Safety: [`SafeDeviceCopyWrapper`] is a transparent newtype around [`T`]
        unsafe { core::slice::from_raw_parts_mut(slice.as_mut_ptr().cast(), slice.len()) }
    }
}

unsafe impl<T: SafeDeviceCopy + TypeGraphLayout> RustToCuda for SafeDeviceCopyWrapper<T> {
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
        Ok((DeviceAccessible::from(&self.0), alloc))
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

unsafe impl<T: SafeDeviceCopy + TypeGraphLayout> RustToCudaAsync for SafeDeviceCopyWrapper<T> {
    #[cfg(feature = "host")]
    #[allow(clippy::type_complexity)]
    unsafe fn borrow_async<A: CudaAlloc>(
        &self,
        alloc: A,
        _stream: &rustacuda::stream::Stream,
    ) -> rustacuda::error::CudaResult<(
        DeviceAccessible<Self::CudaRepresentation>,
        CombinedCudaAlloc<Self::CudaAllocation, A>,
    )> {
        let alloc = CombinedCudaAlloc::new(NoCudaAlloc, alloc);
        Ok((DeviceAccessible::from(&self.0), alloc))
    }

    #[cfg(feature = "host")]
    unsafe fn restore_async<A: CudaAlloc>(
        &mut self,
        alloc: CombinedCudaAlloc<Self::CudaAllocation, A>,
        _stream: &rustacuda::stream::Stream,
    ) -> rustacuda::error::CudaResult<A> {
        let (_alloc_front, alloc_tail): (NoCudaAlloc, A) = alloc.split();

        Ok(alloc_tail)
    }
}

unsafe impl<T: SafeDeviceCopy + TypeGraphLayout> CudaAsRust for SafeDeviceCopyWrapper<T> {
    type RustRepresentation = Self;

    #[cfg(feature = "device")]
    unsafe fn as_rust(this: &DeviceAccessible<Self>) -> Self::RustRepresentation {
        let mut uninit = core::mem::MaybeUninit::uninit();
        core::ptr::copy_nonoverlapping(&**this, uninit.as_mut_ptr(), 1);
        uninit.assume_init()
    }
}
