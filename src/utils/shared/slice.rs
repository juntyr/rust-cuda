#[cfg(not(target_os = "cuda"))]
use core::marker::PhantomData;

use const_type_layout::TypeGraphLayout;

#[cfg(not(target_os = "cuda"))]
#[allow(clippy::module_name_repetitions)]
#[repr(transparent)]
pub struct ThreadBlockSharedSlice<T: 'static + TypeGraphLayout> {
    len: usize,
    marker: PhantomData<T>,
}

#[cfg(target_os = "cuda")]
#[allow(clippy::module_name_repetitions)]
#[repr(transparent)]
pub struct ThreadBlockSharedSlice<T: 'static + TypeGraphLayout> {
    shared: *mut [T],
}

impl<T: 'static + TypeGraphLayout> ThreadBlockSharedSlice<T> {
    #[cfg(any(not(target_os = "cuda"), doc))]
    #[doc(cfg(not(target_os = "cuda")))]
    #[must_use]
    pub fn new_uninit_with_len(len: usize) -> Self {
        Self {
            len,
            marker: PhantomData::<T>,
        }
    }

    #[cfg(any(not(target_os = "cuda"), doc))]
    #[doc(cfg(not(target_os = "cuda")))]
    #[must_use]
    pub fn with_len(mut self, len: usize) -> Self {
        self.len = len;
        self
    }

    #[cfg(any(not(target_os = "cuda"), doc))]
    #[doc(cfg(not(target_os = "cuda")))]
    #[must_use]
    pub fn with_len_mut(&mut self, len: usize) -> &mut Self {
        self.len = len;
        self
    }

    #[cfg(not(target_os = "cuda"))]
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    #[cfg(target_os = "cuda")]
    #[must_use]
    pub fn len(&self) -> usize {
        core::ptr::metadata(self.shared)
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[cfg(any(target_os = "cuda", doc))]
    #[doc(cfg(target_os = "cuda"))]
    #[must_use]
    pub fn as_mut_ptr(&self) -> *mut T {
        self.shared.cast()
    }

    #[cfg(any(target_os = "cuda", doc))]
    #[doc(cfg(target_os = "cuda"))]
    #[must_use]
    pub fn as_mut_slice_ptr(&self) -> *mut [T] {
        self.shared
    }

    #[cfg(any(target_os = "cuda", doc))]
    #[doc(cfg(target_os = "cuda"))]
    /// Safety:
    ///
    /// The provided `index` must not be out of bounds.
    #[inline]
    #[must_use]
    pub unsafe fn index_mut_unchecked<I: core::slice::SliceIndex<[T]>>(
        &self,
        index: I,
    ) -> *mut <I as core::slice::SliceIndex<[T]>>::Output {
        self.shared.get_unchecked_mut(index)
    }
}
