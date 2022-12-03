use rustacuda_core::DeviceCopy;

#[derive(TypeLayout)]
#[repr(transparent)]
pub struct ThreadBlockShared<T: 'static> {
    marker: [T; 0],
}

unsafe impl<T: 'static> DeviceCopy for ThreadBlockShared<T> {}

#[cfg(not(any(all(not(feature = "host"), target_os = "cuda"), doc)))]
#[doc(cfg(not(all(not(feature = "host"), target_os = "cuda"))))]
impl<T: 'static> ThreadBlockShared<T> {
    #[must_use]
    pub fn uninit() -> Self {
        Self { marker: [] }
    }
}

#[cfg(any(all(not(feature = "host"), target_os = "cuda"), doc))]
#[doc(cfg(all(not(feature = "host"), target_os = "cuda")))]
impl<T: 'static> ThreadBlockShared<T> {
    #[must_use]
    pub fn new_uninit() -> *mut T {
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

        shared
    }

    #[must_use]
    pub fn with_uninit<F: FnOnce(*mut T) -> Q, Q>(self, inner: F) -> Q {
        inner(Self::new_uninit())
    }
}
