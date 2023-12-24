use const_type_layout::TypeGraphLayout;

#[allow(clippy::module_name_repetitions)]
#[repr(transparent)]
pub struct ThreadBlockSharedSlice<T: 'static + TypeGraphLayout> {
    #[cfg(not(target_os = "cuda"))]
    // dangling marker s.t. Self is not StackOnly
    dangling: *mut [T],
    #[cfg(target_os = "cuda")]
    shared: *mut [T],
}

impl<T: 'static + TypeGraphLayout> ThreadBlockSharedSlice<T> {
    #[cfg(any(not(target_os = "cuda"), doc))]
    #[doc(cfg(not(target_os = "cuda")))]
    #[must_use]
    pub fn new_uninit_with_len(len: usize) -> Self {
        Self {
            dangling: Self::dangling_slice_with_len(len),
        }
    }

    #[cfg(any(not(target_os = "cuda"), doc))]
    #[doc(cfg(not(target_os = "cuda")))]
    #[must_use]
    pub fn with_len(mut self, len: usize) -> Self {
        self.dangling = Self::dangling_slice_with_len(len);
        self
    }

    #[cfg(any(not(target_os = "cuda"), doc))]
    #[doc(cfg(not(target_os = "cuda")))]
    #[must_use]
    pub fn with_len_mut(&mut self, len: usize) -> &mut Self {
        self.dangling = Self::dangling_slice_with_len(len);
        self
    }

    #[cfg(not(target_os = "cuda"))]
    fn dangling_slice_with_len(len: usize) -> *mut [T] {
        core::ptr::slice_from_raw_parts_mut(core::ptr::NonNull::dangling().as_ptr(), len)
    }

    #[must_use]
    pub fn len(&self) -> usize {
        core::ptr::metadata({
            #[cfg(not(target_os = "cuda"))]
            {
                self.dangling
            }
            #[cfg(target_os = "cuda")]
            {
                self.shared
            }
        })
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[cfg(any(target_os = "cuda", doc))]
    #[doc(cfg(target_os = "cuda"))]
    #[must_use]
    pub const fn as_mut_ptr(&self) -> *mut T {
        self.shared.cast()
    }

    #[cfg(any(target_os = "cuda", doc))]
    #[doc(cfg(target_os = "cuda"))]
    #[must_use]
    pub const fn as_mut_slice_ptr(&self) -> *mut [T] {
        self.shared
    }

    #[cfg(any(target_os = "cuda", doc))]
    #[doc(cfg(target_os = "cuda"))]
    /// # Safety
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
