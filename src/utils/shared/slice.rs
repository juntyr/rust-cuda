use rustacuda_core::DeviceCopy;

#[allow(clippy::module_name_repetitions)]
#[derive(TypeLayout)]
#[repr(C)]
pub struct ThreadBlockSharedSlice<T: 'static> {
    len: usize,
    byte_offset: usize,
    marker: [T; 0],
}

unsafe impl<T: 'static> DeviceCopy for ThreadBlockSharedSlice<T> {}

#[cfg(not(any(all(not(feature = "host"), target_os = "cuda"), doc)))]
#[doc(cfg(not(all(not(feature = "host"), target_os = "cuda"))))]
impl<T: 'static> ThreadBlockSharedSlice<T> {
    #[must_use]
    pub fn with_len(len: usize) -> Self {
        Self {
            len,
            byte_offset: 0,
            marker: [],
        }
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

#[cfg(all(not(feature = "host"), target_os = "cuda"))]
#[doc(cfg(all(not(feature = "host"), target_os = "cuda")))]
impl<T: 'static> ThreadBlockSharedSlice<T> {
    /// # Safety
    ///
    /// The thread-block shared dynamic memory must be initialised once and
    /// only once per kernel.
    pub unsafe fn init() {
        unsafe {
            core::arch::asm!(
                ".shared .align {align} .b8 rust_cuda_dynamic_shared[];",
                align = const(core::mem::align_of::<T>()),
            );
        }
    }

    /// # Safety
    ///
    /// Exposing the [`ThreadBlockSharedSlice`] must be preceded by exactly one
    /// call to [`ThreadBlockSharedSlice::init`] for the type `T` amongst
    /// all `ThreadBlockSharedSlice<T>` that has the largest alignment.
    pub unsafe fn with_uninit<F: FnOnce(*mut [T]) -> Q, Q>(self, inner: F) -> Q {
        let base: *mut u8;

        unsafe {
            core::arch::asm!(
                "cvta.shared.u64 {reg}, rust_cuda_dynamic_shared;",
                reg = out(reg64) base,
            );
        }

        let slice =
            core::ptr::slice_from_raw_parts_mut(base.add(self.byte_offset).cast(), self.len);

        inner(slice)
    }
}
