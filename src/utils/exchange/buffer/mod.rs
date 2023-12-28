#[cfg(any(feature = "host", feature = "device"))]
use core::{
    mem::MaybeUninit,
    ops::{Deref, DerefMut},
};

use const_type_layout::TypeLayout;

use const_type_layout::TypeGraphLayout;

use crate::safety::{PortableBitSemantics, StackOnly};

#[cfg(any(feature = "host", feature = "device"))]
use crate::{
    alloc::NoCudaAlloc,
    lend::{RustToCuda, RustToCudaAsync},
};

#[cfg(feature = "host")]
use crate::{
    alloc::{CombinedCudaAlloc, CudaAlloc},
    utils::ffi::DeviceAccessible,
};

#[cfg(any(feature = "host", feature = "device"))]
use self::common::CudaExchangeBufferCudaRepresentation;

#[cfg(any(feature = "host", feature = "device"))]
mod common;
#[cfg(feature = "device")]
mod device;
#[cfg(feature = "host")]
mod host;

#[cfg(any(feature = "host", feature = "device"))]
#[allow(clippy::module_name_repetitions)]
pub struct CudaExchangeBuffer<
    T: StackOnly + PortableBitSemantics + TypeGraphLayout,
    const M2D: bool,
    const M2H: bool,
> {
    #[cfg(feature = "host")]
    inner: host::CudaExchangeBufferHost<T, M2D, M2H>,
    #[cfg(all(feature = "device", not(feature = "host")))]
    inner: device::CudaExchangeBufferDevice<T, M2D, M2H>,
}

#[cfg(feature = "host")]
impl<
        T: Clone + StackOnly + PortableBitSemantics + TypeGraphLayout,
        const M2D: bool,
        const M2H: bool,
    > CudaExchangeBuffer<T, M2D, M2H>
{
    /// # Errors
    /// Returns a [`rustacuda::error::CudaError`] iff an error occurs inside
    /// CUDA
    pub fn new(elem: &T, capacity: usize) -> rustacuda::error::CudaResult<Self> {
        Ok(Self {
            inner: host::CudaExchangeBufferHost::new(elem, capacity)?,
        })
    }
}

#[cfg(feature = "host")]
impl<T: StackOnly + PortableBitSemantics + TypeGraphLayout, const M2D: bool, const M2H: bool>
    CudaExchangeBuffer<T, M2D, M2H>
{
    /// # Errors
    /// Returns a [`rustacuda::error::CudaError`] iff an error occurs inside
    /// CUDA
    pub fn from_vec(vec: Vec<T>) -> rustacuda::error::CudaResult<Self> {
        Ok(Self {
            inner: host::CudaExchangeBufferHost::from_vec(vec)?,
        })
    }
}

#[cfg(any(feature = "host", feature = "device"))]
impl<T: StackOnly + PortableBitSemantics + TypeGraphLayout, const M2D: bool, const M2H: bool> Deref
    for CudaExchangeBuffer<T, M2D, M2H>
{
    type Target = [CudaExchangeItem<T, M2D, M2H>];

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

#[cfg(any(feature = "host", feature = "device"))]
impl<T: StackOnly + PortableBitSemantics + TypeGraphLayout, const M2D: bool, const M2H: bool>
    DerefMut for CudaExchangeBuffer<T, M2D, M2H>
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

#[cfg(any(feature = "host", feature = "device"))]
unsafe impl<T: StackOnly + PortableBitSemantics + TypeGraphLayout, const M2D: bool, const M2H: bool>
    RustToCuda for CudaExchangeBuffer<T, M2D, M2H>
{
    type CudaAllocation = NoCudaAlloc;
    type CudaRepresentation = CudaExchangeBufferCudaRepresentation<T, M2D, M2H>;

    #[cfg(feature = "host")]
    #[allow(clippy::type_complexity)]
    unsafe fn borrow<A: CudaAlloc>(
        &self,
        alloc: A,
    ) -> rustacuda::error::CudaResult<(
        DeviceAccessible<Self::CudaRepresentation>,
        CombinedCudaAlloc<Self::CudaAllocation, A>,
    )> {
        self.inner.borrow(alloc)
    }

    #[cfg(feature = "host")]
    #[allow(clippy::type_complexity)]
    unsafe fn restore<A: CudaAlloc>(
        &mut self,
        alloc: CombinedCudaAlloc<Self::CudaAllocation, A>,
    ) -> rustacuda::error::CudaResult<A> {
        self.inner.restore(alloc)
    }
}

#[cfg(any(feature = "host", feature = "device"))]
unsafe impl<T: StackOnly + PortableBitSemantics + TypeGraphLayout, const M2D: bool, const M2H: bool>
    RustToCudaAsync for CudaExchangeBuffer<T, M2D, M2H>
{
    type CudaAllocationAsync = NoCudaAlloc;

    #[cfg(feature = "host")]
    #[allow(clippy::type_complexity)]
    unsafe fn borrow_async<A: CudaAlloc>(
        &self,
        alloc: A,
        stream: &rustacuda::stream::Stream,
    ) -> rustacuda::error::CudaResult<(
        DeviceAccessible<Self::CudaRepresentation>,
        CombinedCudaAlloc<Self::CudaAllocation, A>,
    )> {
        self.inner.borrow_async(alloc, stream)
    }

    #[cfg(feature = "host")]
    #[allow(clippy::type_complexity)]
    unsafe fn restore_async<A: CudaAlloc>(
        &mut self,
        alloc: CombinedCudaAlloc<Self::CudaAllocation, A>,
        stream: &rustacuda::stream::Stream,
    ) -> rustacuda::error::CudaResult<A> {
        self.inner.restore_async(alloc, stream)
    }
}

#[repr(transparent)]
#[derive(Clone, Copy, TypeLayout)]
pub struct CudaExchangeItem<
    T: StackOnly + PortableBitSemantics + TypeGraphLayout,
    const M2D: bool,
    const M2H: bool,
>(T);

impl<T: StackOnly + PortableBitSemantics + TypeGraphLayout, const M2D: bool>
    CudaExchangeItem<T, M2D, true>
{
    #[cfg(feature = "host")]
    pub const fn read(&self) -> &T {
        &self.0
    }

    #[cfg(feature = "device")]
    pub fn write(&mut self, value: T) {
        self.0 = value;
    }
}

impl<T: StackOnly + PortableBitSemantics + TypeGraphLayout, const M2H: bool>
    CudaExchangeItem<T, true, M2H>
{
    #[cfg(feature = "device")]
    pub const fn read(&self) -> &T {
        &self.0
    }

    #[cfg(feature = "host")]
    pub fn write(&mut self, value: T) {
        self.0 = value;
    }
}

impl<T: StackOnly + PortableBitSemantics + TypeGraphLayout> AsMut<T>
    for CudaExchangeItem<T, true, true>
{
    fn as_mut(&mut self) -> &mut T {
        &mut self.0
    }
}

impl<T: StackOnly + PortableBitSemantics + TypeGraphLayout> CudaExchangeItem<T, false, true> {
    #[cfg(feature = "host")]
    pub const fn as_scratch(&self) -> &T {
        &self.0
    }

    #[cfg(feature = "host")]
    pub fn as_scratch_mut(&mut self) -> &mut T {
        &mut self.0
    }
}

impl<T: StackOnly + PortableBitSemantics + TypeGraphLayout> CudaExchangeItem<T, true, false> {
    #[cfg(feature = "device")]
    pub const fn as_scratch(&self) -> &T {
        &self.0
    }

    #[cfg(feature = "device")]
    pub fn as_scratch_mut(&mut self) -> &mut T {
        &mut self.0
    }
}

impl<T: StackOnly + PortableBitSemantics + TypeGraphLayout> CudaExchangeItem<T, true, false> {
    #[cfg(feature = "host")]
    pub const fn as_uninit(&self) -> &MaybeUninit<T> {
        // Safety:
        // - MaybeUninit is a transparent newtype union
        // - CudaExchangeItem is a transparent newtype
        unsafe { &*core::ptr::from_ref(self).cast() }
    }

    #[cfg(feature = "host")]
    pub fn as_uninit_mut(&mut self) -> &mut MaybeUninit<T> {
        // Safety:
        // - MaybeUninit is a transparent newtype union
        // - CudaExchangeItem is a transparent newtype
        unsafe { &mut *core::ptr::from_mut(self).cast() }
    }
}

impl<T: StackOnly + PortableBitSemantics + TypeGraphLayout> CudaExchangeItem<T, false, true> {
    #[cfg(feature = "device")]
    pub const fn as_uninit(&self) -> &MaybeUninit<T> {
        // Safety:
        // - MaybeUninit is a transparent newtype union
        // - CudaExchangeItem is a transparent newtype
        unsafe { &*core::ptr::from_ref(self).cast() }
    }

    #[cfg(feature = "device")]
    pub fn as_uninit_mut(&mut self) -> &mut MaybeUninit<T> {
        // Safety:
        // - MaybeUninit is a transparent newtype union
        // - CudaExchangeItem is a transparent newtype
        unsafe { &mut *core::ptr::from_mut(self).cast() }
    }
}
