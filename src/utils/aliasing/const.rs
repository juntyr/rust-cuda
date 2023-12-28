#[cfg(any(feature = "host", feature = "device"))]
use core::{
    borrow::{Borrow, BorrowMut},
    convert::{AsMut, AsRef},
    ops::{Deref, DerefMut},
};

use const_type_layout::TypeLayout;

use crate::{
    lend::{CudaAsRust, RustToCuda, RustToCudaAsync},
    utils::ffi::DeviceAccessible,
};

#[repr(transparent)]
#[derive(Clone, TypeLayout)]
pub struct SplitSliceOverCudaThreadsConstStride<T, const STRIDE: usize>(T);

impl<T, const STRIDE: usize> SplitSliceOverCudaThreadsConstStride<T, STRIDE> {
    #[cfg(feature = "host")]
    #[must_use]
    pub const fn new(inner: T) -> Self {
        Self(inner)
    }
}

#[cfg(feature = "device")]
fn split_slice_const_stride<E, const STRIDE: usize>(slice: &[E]) -> &[E] {
    let offset: usize = crate::device::thread::Thread::this().index() * STRIDE;
    let len = slice.len().min(offset + STRIDE).saturating_sub(offset);

    unsafe { core::slice::from_raw_parts(slice.as_ptr().add(offset), len) }
}

#[cfg(feature = "device")]
fn split_slice_const_stride_mut<E, const STRIDE: usize>(slice: &mut [E]) -> &mut [E] {
    let offset: usize = crate::device::thread::Thread::this().index() * STRIDE;
    let len = slice.len().min(offset + STRIDE).saturating_sub(offset);

    unsafe { core::slice::from_raw_parts_mut(slice.as_mut_ptr().add(offset), len) }
}

#[cfg(feature = "device")]
impl<T, const STRIDE: usize> SplitSliceOverCudaThreadsConstStride<T, STRIDE> {
    /// # Safety
    ///
    /// All cross-CUDA-thread aliasing guarantees are lost with this method.
    /// Instead, the caller must ensure that no two threads in a kernel launch
    /// access the same underlying elements.
    pub const unsafe fn alias_unchecked(&self) -> &T {
        &self.0
    }

    /// # Safety
    ///
    /// All cross-CUDA-thread aliasing guarantees are lost with this method.
    /// Instead, the caller must ensure that no two threads in a kernel launch
    /// access the same underlying elements.
    pub unsafe fn alias_mut_unchecked(&mut self) -> &mut T {
        &mut self.0
    }
}

#[cfg(any(feature = "device", doc))]
#[doc(cfg(any(feature = "device", feature = "host")))]
impl<E, T: Deref<Target = [E]>, const STRIDE: usize> Deref
    for SplitSliceOverCudaThreadsConstStride<T, STRIDE>
{
    type Target = [E];

    fn deref(&self) -> &Self::Target {
        split_slice_const_stride::<E, STRIDE>(&self.0)
    }
}

#[cfg(any(feature = "device", doc))]
#[doc(cfg(any(feature = "device", feature = "host")))]
impl<E, T: DerefMut<Target = [E]>, const STRIDE: usize> DerefMut
    for SplitSliceOverCudaThreadsConstStride<T, STRIDE>
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        split_slice_const_stride_mut::<E, STRIDE>(&mut self.0)
    }
}

#[cfg(any(feature = "device", doc))]
#[doc(cfg(any(feature = "device", feature = "host")))]
impl<E, T: AsRef<[E]>, const STRIDE: usize> AsRef<[E]>
    for SplitSliceOverCudaThreadsConstStride<T, STRIDE>
{
    fn as_ref(&self) -> &[E] {
        split_slice_const_stride::<E, STRIDE>(self.0.as_ref())
    }
}

#[cfg(any(feature = "device", doc))]
#[doc(cfg(any(feature = "device", feature = "host")))]
impl<E, T: AsMut<[E]>, const STRIDE: usize> AsMut<[E]>
    for SplitSliceOverCudaThreadsConstStride<T, STRIDE>
{
    fn as_mut(&mut self) -> &mut [E] {
        split_slice_const_stride_mut::<E, STRIDE>(self.0.as_mut())
    }
}

#[cfg(any(feature = "device", doc))]
#[doc(cfg(any(feature = "device", feature = "host")))]
impl<E, T: Borrow<[E]>, const STRIDE: usize> Borrow<[E]>
    for SplitSliceOverCudaThreadsConstStride<T, STRIDE>
{
    fn borrow(&self) -> &[E] {
        split_slice_const_stride::<E, STRIDE>(self.0.borrow())
    }
}

#[cfg(any(feature = "device", doc))]
#[doc(cfg(any(feature = "device", feature = "host")))]
impl<E, T: BorrowMut<[E]>, const STRIDE: usize> BorrowMut<[E]>
    for SplitSliceOverCudaThreadsConstStride<T, STRIDE>
{
    fn borrow_mut(&mut self) -> &mut [E] {
        split_slice_const_stride_mut::<E, STRIDE>(self.0.borrow_mut())
    }
}

#[cfg(all(feature = "host", not(doc)))]
impl<E, T: Deref<Target = [E]>, const STRIDE: usize> Deref
    for SplitSliceOverCudaThreadsConstStride<T, STRIDE>
{
    type Target = [E];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[cfg(all(feature = "host", not(doc)))]
impl<E, T: DerefMut<Target = [E]>, const STRIDE: usize> DerefMut
    for SplitSliceOverCudaThreadsConstStride<T, STRIDE>
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

#[cfg(all(feature = "host", not(doc)))]
impl<E, T: AsRef<[E]>, const STRIDE: usize> AsRef<[E]>
    for SplitSliceOverCudaThreadsConstStride<T, STRIDE>
{
    fn as_ref(&self) -> &[E] {
        self.0.as_ref()
    }
}

#[cfg(all(feature = "host", not(doc)))]
impl<E, T: AsMut<[E]>, const STRIDE: usize> AsMut<[E]>
    for SplitSliceOverCudaThreadsConstStride<T, STRIDE>
{
    fn as_mut(&mut self) -> &mut [E] {
        self.0.as_mut()
    }
}

#[cfg(all(feature = "host", not(doc)))]
impl<E, T: Borrow<[E]>, const STRIDE: usize> Borrow<[E]>
    for SplitSliceOverCudaThreadsConstStride<T, STRIDE>
{
    fn borrow(&self) -> &[E] {
        self.0.borrow()
    }
}

#[cfg(all(feature = "host", not(doc)))]
impl<E, T: BorrowMut<[E]>, const STRIDE: usize> BorrowMut<[E]>
    for SplitSliceOverCudaThreadsConstStride<T, STRIDE>
{
    fn borrow_mut(&mut self) -> &mut [E] {
        self.0.borrow_mut()
    }
}

unsafe impl<T: RustToCuda, const STRIDE: usize> RustToCuda
    for SplitSliceOverCudaThreadsConstStride<T, STRIDE>
{
    type CudaAllocation = T::CudaAllocation;
    type CudaRepresentation =
        SplitSliceOverCudaThreadsConstStride<DeviceAccessible<T::CudaRepresentation>, STRIDE>;

    #[cfg(feature = "host")]
    #[allow(clippy::type_complexity)]
    unsafe fn borrow<A: crate::alloc::CudaAlloc>(
        &self,
        alloc: A,
    ) -> rustacuda::error::CudaResult<(
        DeviceAccessible<Self::CudaRepresentation>,
        crate::alloc::CombinedCudaAlloc<Self::CudaAllocation, A>,
    )> {
        let (cuda_repr, alloc) = self.0.borrow(alloc)?;

        Ok((
            DeviceAccessible::from(SplitSliceOverCudaThreadsConstStride::new(cuda_repr)),
            alloc,
        ))
    }

    #[cfg(feature = "host")]
    unsafe fn restore<A: crate::alloc::CudaAlloc>(
        &mut self,
        alloc: crate::alloc::CombinedCudaAlloc<Self::CudaAllocation, A>,
    ) -> rustacuda::error::CudaResult<A> {
        self.0.restore(alloc)
    }
}

unsafe impl<T: RustToCudaAsync, const STRIDE: usize> RustToCudaAsync
    for SplitSliceOverCudaThreadsConstStride<T, STRIDE>
{
    type CudaAllocationAsync = T::CudaAllocationAsync;

    #[cfg(feature = "host")]
    #[allow(clippy::type_complexity)]
    unsafe fn borrow_async<A: crate::alloc::CudaAlloc>(
        &self,
        alloc: A,
        stream: &rustacuda::stream::Stream,
    ) -> rustacuda::error::CudaResult<(
        DeviceAccessible<Self::CudaRepresentation>,
        crate::alloc::CombinedCudaAlloc<Self::CudaAllocationAsync, A>,
    )> {
        let (cuda_repr, alloc) = self.0.borrow_async(alloc, stream)?;

        Ok((
            DeviceAccessible::from(SplitSliceOverCudaThreadsConstStride::new(cuda_repr)),
            alloc,
        ))
    }

    #[cfg(feature = "host")]
    unsafe fn restore_async<A: crate::alloc::CudaAlloc>(
        &mut self,
        alloc: crate::alloc::CombinedCudaAlloc<Self::CudaAllocationAsync, A>,
        stream: &rustacuda::stream::Stream,
    ) -> rustacuda::error::CudaResult<A> {
        self.0.restore_async(alloc, stream)
    }
}

unsafe impl<T: CudaAsRust, const STRIDE: usize> CudaAsRust
    for SplitSliceOverCudaThreadsConstStride<DeviceAccessible<T>, STRIDE>
{
    type RustRepresentation = SplitSliceOverCudaThreadsConstStride<T::RustRepresentation, STRIDE>;

    #[cfg(feature = "device")]
    unsafe fn as_rust(this: &DeviceAccessible<Self>) -> Self::RustRepresentation {
        SplitSliceOverCudaThreadsConstStride(CudaAsRust::as_rust(&this.0))
    }
}
