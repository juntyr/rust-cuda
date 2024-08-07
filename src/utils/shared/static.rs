#[repr(transparent)]
pub struct ThreadBlockShared<T: 'static> {
    #[cfg_attr(not(feature = "device"), allow(dead_code))]
    shared: *mut T,
}

impl<T: 'static> ThreadBlockShared<T> {
    #[cfg(any(feature = "host", feature = "device"))]
    #[must_use]
    #[expect(clippy::inline_always)]
    #[cfg_attr(feature = "host", expect(clippy::missing_const_for_fn))]
    #[inline(always)]
    pub fn new_uninit() -> Self {
        #[cfg(feature = "host")]
        {
            Self {
                shared: core::ptr::NonNull::dangling().as_ptr(),
            }
        }

        #[cfg(feature = "device")]
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

    #[cfg(feature = "device")]
    #[must_use]
    pub const fn as_mut_ptr(&self) -> *mut T {
        self.shared
    }
}

impl<T: 'static, const N: usize> ThreadBlockShared<[T; N]> {
    #[cfg(feature = "device")]
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
