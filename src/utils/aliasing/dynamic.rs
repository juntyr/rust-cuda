#[cfg(any(feature = "host", feature = "device"))]
use core::{
    borrow::{Borrow, BorrowMut},
    ops::{Deref, DerefMut},
};

use const_type_layout::TypeLayout;

use crate::{
    lend::{CudaAsRust, RustToCuda, RustToCudaAsync},
    utils::ffi::DeviceAccessible,
};

#[repr(C)]
#[derive(PartialEq, Eq, PartialOrd, Ord, Hash, TypeLayout)]
pub struct SplitSliceOverCudaThreadsDynamicStride<T> {
    stride: usize,
    inner: T,
}

impl<T> SplitSliceOverCudaThreadsDynamicStride<T> {
    #[cfg(feature = "host")]
    #[must_use]
    pub const fn new(inner: T, stride: usize) -> Self {
        Self { stride, inner }
    }
}

#[cfg(feature = "device")]
fn split_slice_dynamic_stride<E>(slice: &[E], stride: usize) -> &[E] {
    let offset: usize = crate::device::thread::Thread::this().index() * stride;
    let len = slice.len().min(offset + stride).saturating_sub(offset);

    unsafe { core::slice::from_raw_parts(slice.as_ptr().add(offset), len) }
}

#[cfg(feature = "device")]
fn split_slice_dynamic_stride_mut<E>(slice: &mut [E], stride: usize) -> &mut [E] {
    let offset: usize = crate::device::thread::Thread::this().index() * stride;
    let len = slice.len().min(offset + stride).saturating_sub(offset);

    unsafe { core::slice::from_raw_parts_mut(slice.as_mut_ptr().add(offset), len) }
}

#[cfg(feature = "device")]
impl<T> SplitSliceOverCudaThreadsDynamicStride<T> {
    /// # Safety
    ///
    /// All cross-CUDA-thread aliasing guarantees are lost with this method.
    /// Instead, the caller must ensure that no two threads in a kernel launch
    /// access the same underlying elements.
    pub const unsafe fn alias_unchecked(&self) -> &T {
        &self.inner
    }

    /// # Safety
    ///
    /// All cross-CUDA-thread aliasing guarantees are lost with this method.
    /// Instead, the caller must ensure that no two threads in a kernel launch
    /// access the same underlying elements.
    pub unsafe fn alias_mut_unchecked(&mut self) -> &mut T {
        &mut self.inner
    }
}

#[cfg(any(feature = "device", doc))]
#[doc(cfg(any(feature = "device", feature = "host")))]
impl<E, T: Deref<Target = [E]>> Deref for SplitSliceOverCudaThreadsDynamicStride<T> {
    type Target = [E];

    fn deref(&self) -> &Self::Target {
        split_slice_dynamic_stride(&self.inner, self.stride)
    }
}

#[cfg(any(feature = "device", doc))]
#[doc(cfg(any(feature = "device", feature = "host")))]
impl<E, T: DerefMut<Target = [E]>> DerefMut for SplitSliceOverCudaThreadsDynamicStride<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        split_slice_dynamic_stride_mut(&mut self.inner, self.stride)
    }
}

#[cfg(any(feature = "device", doc))]
#[doc(cfg(any(feature = "device", feature = "host")))]
impl<E, T: AsRef<[E]>> AsRef<[E]> for SplitSliceOverCudaThreadsDynamicStride<T> {
    fn as_ref(&self) -> &[E] {
        split_slice_dynamic_stride(self.inner.as_ref(), self.stride)
    }
}

#[cfg(any(feature = "device", doc))]
#[doc(cfg(any(feature = "device", feature = "host")))]
impl<E, T: AsMut<[E]>> AsMut<[E]> for SplitSliceOverCudaThreadsDynamicStride<T> {
    fn as_mut(&mut self) -> &mut [E] {
        split_slice_dynamic_stride_mut(self.inner.as_mut(), self.stride)
    }
}

#[cfg(any(feature = "device", doc))]
#[doc(cfg(any(feature = "device", feature = "host")))]
impl<E, T: Borrow<[E]>> Borrow<[E]> for SplitSliceOverCudaThreadsDynamicStride<T> {
    fn borrow(&self) -> &[E] {
        split_slice_dynamic_stride(self.inner.borrow(), self.stride)
    }
}

#[cfg(any(feature = "device", doc))]
#[doc(cfg(any(feature = "device", feature = "host")))]
impl<E, T: BorrowMut<[E]>> BorrowMut<[E]> for SplitSliceOverCudaThreadsDynamicStride<T> {
    fn borrow_mut(&mut self) -> &mut [E] {
        split_slice_dynamic_stride_mut(self.inner.borrow_mut(), self.stride)
    }
}

#[cfg(all(feature = "host", not(doc)))]
impl<E, T: Deref<Target = [E]>> Deref for SplitSliceOverCudaThreadsDynamicStride<T> {
    type Target = [E];

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

#[cfg(all(feature = "host", not(doc)))]
impl<E, T: DerefMut<Target = [E]>> DerefMut for SplitSliceOverCudaThreadsDynamicStride<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

#[cfg(all(feature = "host", not(doc)))]
impl<E, T: AsRef<[E]>> AsRef<[E]> for SplitSliceOverCudaThreadsDynamicStride<T> {
    fn as_ref(&self) -> &[E] {
        self.inner.as_ref()
    }
}

#[cfg(all(feature = "host", not(doc)))]
impl<E, T: AsMut<[E]>> AsMut<[E]> for SplitSliceOverCudaThreadsDynamicStride<T> {
    fn as_mut(&mut self) -> &mut [E] {
        self.inner.as_mut()
    }
}

#[cfg(all(feature = "host", not(doc)))]
impl<E, T: Borrow<[E]>> Borrow<[E]> for SplitSliceOverCudaThreadsDynamicStride<T> {
    fn borrow(&self) -> &[E] {
        self.inner.borrow()
    }
}

#[cfg(all(feature = "host", not(doc)))]
impl<E, T: BorrowMut<[E]>> BorrowMut<[E]> for SplitSliceOverCudaThreadsDynamicStride<T> {
    fn borrow_mut(&mut self) -> &mut [E] {
        self.inner.borrow_mut()
    }
}

unsafe impl<T: RustToCuda> RustToCuda for SplitSliceOverCudaThreadsDynamicStride<T> {
    type CudaAllocation = T::CudaAllocation;
    type CudaRepresentation =
        SplitSliceOverCudaThreadsDynamicStride<DeviceAccessible<T::CudaRepresentation>>;

    #[cfg(feature = "host")]
    #[allow(clippy::type_complexity)]
    unsafe fn borrow<A: crate::alloc::CudaAlloc>(
        &self,
        alloc: A,
    ) -> rustacuda::error::CudaResult<(
        DeviceAccessible<Self::CudaRepresentation>,
        crate::alloc::CombinedCudaAlloc<Self::CudaAllocation, A>,
    )> {
        let (cuda_repr, alloc) = self.inner.borrow(alloc)?;

        Ok((
            DeviceAccessible::from(SplitSliceOverCudaThreadsDynamicStride::new(
                cuda_repr,
                self.stride,
            )),
            alloc,
        ))
    }

    #[cfg(feature = "host")]
    unsafe fn restore<A: crate::alloc::CudaAlloc>(
        &mut self,
        alloc: crate::alloc::CombinedCudaAlloc<Self::CudaAllocation, A>,
    ) -> rustacuda::error::CudaResult<A> {
        self.inner.restore(alloc)
    }
}

unsafe impl<T: RustToCudaAsync> RustToCudaAsync for SplitSliceOverCudaThreadsDynamicStride<T> {
    type CudaAllocationAsync = T::CudaAllocationAsync;

    #[cfg(feature = "host")]
    #[allow(clippy::type_complexity)]
    unsafe fn borrow_async<'stream, A: crate::alloc::CudaAlloc>(
        &self,
        alloc: A,
        stream: crate::host::Stream<'stream>,
    ) -> rustacuda::error::CudaResult<(
        crate::utils::r#async::Async<'_, 'stream, DeviceAccessible<Self::CudaRepresentation>>,
        crate::alloc::CombinedCudaAlloc<Self::CudaAllocationAsync, A>,
    )> {
        let (r#async, alloc) = self.inner.borrow_async(alloc, stream)?;
        let (cuda_repr, completion) = unsafe { r#async.unwrap_unchecked()? };

        let cuda_repr = DeviceAccessible::from(SplitSliceOverCudaThreadsDynamicStride::new(
            cuda_repr,
            self.stride,
        ));

        let r#async = if matches!(completion, Some(crate::utils::r#async::NoCompletion)) {
            crate::utils::r#async::Async::pending(
                cuda_repr,
                stream,
                crate::utils::r#async::NoCompletion,
            )?
        } else {
            crate::utils::r#async::Async::ready(cuda_repr, stream)
        };

        Ok((r#async, alloc))
    }

    #[cfg(feature = "host")]
    unsafe fn restore_async<'a, 'stream, A: crate::alloc::CudaAlloc, O>(
        this: owning_ref::BoxRefMut<'a, O, Self>,
        alloc: crate::alloc::CombinedCudaAlloc<Self::CudaAllocationAsync, A>,
        stream: crate::host::Stream<'stream>,
    ) -> rustacuda::error::CudaResult<(
        crate::utils::r#async::Async<
            'a,
            'stream,
            owning_ref::BoxRefMut<'a, O, Self>,
            crate::utils::r#async::CompletionFnMut<'a, Self>,
        >,
        A,
    )> {
        let this_backup = unsafe { std::mem::ManuallyDrop::new(std::ptr::read(&this)) };

        let (r#async, alloc_tail) =
            T::restore_async(this.map_mut(|this| &mut this.inner), alloc, stream)?;

        let (inner, on_completion) = unsafe { r#async.unwrap_unchecked()? };

        std::mem::forget(inner);
        let this = std::mem::ManuallyDrop::into_inner(this_backup);

        if let Some(on_completion) = on_completion {
            let r#async = crate::utils::r#async::Async::<
                _,
                crate::utils::r#async::CompletionFnMut<'a, Self>,
            >::pending(
                this,
                stream,
                Box::new(|this: &mut Self| on_completion(&mut this.inner)),
            )?;
            Ok((r#async, alloc_tail))
        } else {
            let r#async = crate::utils::r#async::Async::ready(this, stream);
            Ok((r#async, alloc_tail))
        }
    }
}

unsafe impl<T: CudaAsRust> CudaAsRust
    for SplitSliceOverCudaThreadsDynamicStride<DeviceAccessible<T>>
{
    type RustRepresentation = SplitSliceOverCudaThreadsDynamicStride<T::RustRepresentation>;

    #[cfg(feature = "device")]
    unsafe fn as_rust(this: &DeviceAccessible<Self>) -> Self::RustRepresentation {
        SplitSliceOverCudaThreadsDynamicStride {
            stride: this.stride,
            inner: CudaAsRust::as_rust(&this.inner),
        }
    }
}
