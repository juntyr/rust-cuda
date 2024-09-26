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
    utils::r#async::{Async, CompletionFnMut},
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
#[expect(clippy::module_name_repetitions)]
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

#[cfg(any(feature = "host", feature = "device"))]
unsafe impl<
        T: StackOnly + PortableBitSemantics + TypeGraphLayout + Sync,
        const M2D: bool,
        const M2H: bool,
    > Sync for CudaExchangeBuffer<T, M2D, M2H>
{
}

#[cfg(feature = "host")]
impl<
        T: Clone + StackOnly + PortableBitSemantics + TypeGraphLayout,
        const M2D: bool,
        const M2H: bool,
    > CudaExchangeBuffer<T, M2D, M2H>
{
    /// # Errors
    /// Returns a [`cust::error::CudaError`] iff an error occurs inside
    /// CUDA
    pub fn new(elem: &T, capacity: usize) -> cust::error::CudaResult<Self> {
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
    /// Returns a [`cust::error::CudaError`] iff an error occurs inside
    /// CUDA
    pub fn from_vec(vec: Vec<T>) -> cust::error::CudaResult<Self> {
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
    unsafe fn borrow<A: CudaAlloc>(
        &self,
        alloc: A,
    ) -> cust::error::CudaResult<(
        DeviceAccessible<Self::CudaRepresentation>,
        CombinedCudaAlloc<Self::CudaAllocation, A>,
    )> {
        self.inner.borrow(alloc)
    }

    #[cfg(feature = "host")]
    unsafe fn restore<A: CudaAlloc>(
        &mut self,
        alloc: CombinedCudaAlloc<Self::CudaAllocation, A>,
    ) -> cust::error::CudaResult<A> {
        self.inner.restore(alloc)
    }
}

#[cfg(any(feature = "host", feature = "device"))]
unsafe impl<T: StackOnly + PortableBitSemantics + TypeGraphLayout, const M2D: bool, const M2H: bool>
    RustToCudaAsync for CudaExchangeBuffer<T, M2D, M2H>
{
    type CudaAllocationAsync = NoCudaAlloc;

    #[cfg(feature = "host")]
    unsafe fn borrow_async<'stream, A: CudaAlloc>(
        &self,
        alloc: A,
        stream: crate::host::Stream<'stream>,
    ) -> cust::error::CudaResult<(
        Async<'_, 'stream, DeviceAccessible<Self::CudaRepresentation>>,
        CombinedCudaAlloc<Self::CudaAllocationAsync, A>,
    )> {
        self.inner.borrow_async(alloc, stream)
    }

    #[cfg(feature = "host")]
    unsafe fn restore_async<'a, 'stream, A: CudaAlloc, O>(
        this: owning_ref::BoxRefMut<'a, O, Self>,
        alloc: CombinedCudaAlloc<Self::CudaAllocationAsync, A>,
        stream: crate::host::Stream<'stream>,
    ) -> cust::error::CudaResult<(
        Async<'a, 'stream, owning_ref::BoxRefMut<'a, O, Self>, CompletionFnMut<'a, Self>>,
        A,
    )> {
        let this_backup = unsafe { std::mem::ManuallyDrop::new(std::ptr::read(&this)) };

        let (r#async, alloc_tail) = host::CudaExchangeBufferHost::restore_async(
            this.map_mut(|this| &mut this.inner),
            alloc,
            stream,
        )?;

        let (inner, on_completion) = unsafe { r#async.unwrap_unchecked()? };

        std::mem::forget(inner);
        let this = std::mem::ManuallyDrop::into_inner(this_backup);

        if let Some(on_completion) = on_completion {
            let r#async = Async::<_, CompletionFnMut<'a, Self>>::pending(
                this,
                stream,
                Box::new(|this: &mut Self| on_completion(&mut this.inner)),
            )?;
            Ok((r#async, alloc_tail))
        } else {
            let r#async = Async::ready(this, stream);
            Ok((r#async, alloc_tail))
        }
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
