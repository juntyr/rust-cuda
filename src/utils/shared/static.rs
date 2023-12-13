#[repr(transparent)]
pub struct ThreadBlockShared<T: 'static> {
    #[cfg(not(target_os = "cuda"))]
    // dangling marker s.t. Self is not StackOnly
    _dangling: *mut T,
    #[cfg(target_os = "cuda")]
    shared: *mut T,
}

impl<T: 'static> ThreadBlockShared<T> {
    #[must_use]
    pub fn new_uninit() -> Self {
        #[cfg(not(target_os = "cuda"))]
        {
            Self {
                _dangling: core::ptr::NonNull::dangling().as_ptr(),
            }
        }

        #[cfg(target_os = "cuda")]
        {
            let shared: *mut T;

            unsafe {
                core::arch::asm!(
                    ".shared .align {align} .b8 {reg}_rust_cuda_static_shared[{size}];",
                    "cvta.shared.u64 {reg}, {reg}_rust_cuda_static_shared;",
                    reg = out(reg64) shared,
                    align = const(core::mem::align_of::<T>()),
                    size = const(core::mem::size_of::<T>()),
                );
            }

            Self { shared }
        }
    }

    #[cfg(any(target_os = "cuda", doc))]
    #[doc(cfg(target_os = "cuda"))]
    #[must_use]
    pub fn as_mut_ptr(&self) -> *mut T {
        self.shared
    }
}

impl<T: 'static, const N: usize> ThreadBlockShared<[T; N]> {
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
        core::ptr::slice_from_raw_parts_mut(self.shared.cast::<T>(), N).get_unchecked_mut(index)
    }
}
