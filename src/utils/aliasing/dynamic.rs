use core::{
    borrow::{Borrow, BorrowMut},
    convert::{AsMut, AsRef},
    ops::{Deref, DerefMut},
};

#[repr(C)]
pub struct SplitSliceOverCudaThreadsDynamicStride<T> {
    stride: usize,
    inner: T,
}

#[cfg(all(not(feature = "host"), target_os = "cuda"))]
fn split_slice_dynamic_stride<E>(slice: &[E], stride: usize) -> &[E] {
    let offset: usize = crate::device::utils::index() * stride;
    let len = slice.len().min(offset + stride).saturating_sub(offset);

    unsafe { core::slice::from_raw_parts(slice.as_ptr().add(offset), len) }
}

#[cfg(all(not(feature = "host"), target_os = "cuda"))]
fn split_slice_dynamic_stride_mut<E>(slice: &mut [E], stride: usize) -> &mut [E] {
    let offset: usize = crate::device::utils::index() * stride;
    let len = slice.len().min(offset + stride).saturating_sub(offset);

    unsafe { core::slice::from_raw_parts_mut(slice.as_mut_ptr().add(offset), stride) }
}

#[cfg(all(not(feature = "host"), target_os = "cuda"))]
impl<E, T: Deref<Target = [E]>> Deref for SplitSliceOverCudaThreadsDynamicStride<T> {
    type Target = [E];

    fn deref(&self) -> &Self::Target {
        split_slice_dynamic_stride(&*self.inner, self.stride)
    }
}

#[cfg(all(not(feature = "host"), target_os = "cuda"))]
impl<E, T: DerefMut<Target = [E]>> DerefMut for SplitSliceOverCudaThreadsDynamicStride<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        split_slice_dynamic_stride_mut(&mut *self.inner, self.stride)
    }
}

#[cfg(all(not(feature = "host"), target_os = "cuda"))]
impl<E, T: AsRef<[E]>> AsRef<[E]> for SplitSliceOverCudaThreadsDynamicStride<T> {
    fn as_ref(&self) -> &[E] {
        split_slice_dynamic_stride(self.inner.as_ref(), self.stride)
    }
}

#[cfg(all(not(feature = "host"), target_os = "cuda"))]
impl<E, T: AsMut<[E]>> AsMut<[E]> for SplitSliceOverCudaThreadsDynamicStride<T> {
    fn as_mut(&mut self) -> &mut [E] {
        split_slice_dynamic_stride_mut(self.inner.as_mut(), self.stride)
    }
}

#[cfg(all(not(feature = "host"), target_os = "cuda"))]
impl<E, T: Borrow<[E]>> Borrow<[E]> for SplitSliceOverCudaThreadsDynamicStride<T> {
    fn borrow(&self) -> &[E] {
        split_slice_dynamic_stride(self.inner.borrow(), self.stride)
    }
}

#[cfg(all(not(feature = "host"), target_os = "cuda"))]
impl<E, T: BorrowMut<[E]>> BorrowMut<[E]> for SplitSliceOverCudaThreadsDynamicStride<T> {
    fn borrow_mut(&mut self) -> &mut [E] {
        split_slice_dynamic_stride_mut(self.inner.borrow_mut(), self.stride)
    }
}

#[cfg(any(feature = "host", not(target_os = "cuda")))]
impl<E, T: Deref<Target = [E]>> Deref for SplitSliceOverCudaThreadsDynamicStride<T> {
    type Target = [E];

    fn deref(&self) -> &Self::Target {
        &*self.inner
    }
}

#[cfg(any(feature = "host", not(target_os = "cuda")))]
impl<E, T: DerefMut<Target = [E]>> DerefMut for SplitSliceOverCudaThreadsDynamicStride<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut *self.inner
    }
}

#[cfg(any(feature = "host", not(target_os = "cuda")))]
impl<E, T: AsRef<[E]>> AsRef<[E]> for SplitSliceOverCudaThreadsDynamicStride<T> {
    fn as_ref(&self) -> &[E] {
        self.inner.as_ref()
    }
}

#[cfg(any(feature = "host", not(target_os = "cuda")))]
impl<E, T: AsMut<[E]>> AsMut<[E]> for SplitSliceOverCudaThreadsDynamicStride<T> {
    fn as_mut(&mut self) -> &mut [E] {
        self.inner.as_mut()
    }
}

#[cfg(any(feature = "host", not(target_os = "cuda")))]
impl<E, T: Borrow<[E]>> Borrow<[E]> for SplitSliceOverCudaThreadsDynamicStride<T> {
    fn borrow(&self) -> &[E] {
        self.inner.borrow()
    }
}

#[cfg(any(feature = "host", not(target_os = "cuda")))]
impl<E, T: BorrowMut<[E]>> BorrowMut<[E]> for SplitSliceOverCudaThreadsDynamicStride<T> {
    fn borrow_mut(&mut self) -> &mut [E] {
        self.inner.borrow_mut()
    }
}
