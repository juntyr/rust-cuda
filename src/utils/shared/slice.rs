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

#[cfg(all(not(feature = "host"), target_os = "cuda"))]
#[doc(cfg(all(not(feature = "host"), target_os = "cuda")))]
impl<T: 'static + TypeGraphLayout> ThreadBlockSharedSlice<T> {
    /// # Safety
    ///
    /// Exposing the [`ThreadBlockSharedSlice`] must be preceded by exactly one
    /// call to [`init`].
    pub(crate) unsafe fn with_uninit_for_len<F: FnOnce(&mut Self) -> Q, Q>(
        len: usize,
        inner: F,
    ) -> Q {
        let base: *mut u8;

        unsafe {
            core::arch::asm!(
                "mov.u64    {base}, %rust_cuda_dynamic_shared;",
                base = out(reg64) base,
            );
        }

        let aligned_base = base.byte_add(base.align_offset(core::mem::align_of::<T>()));

        let data: *mut T = aligned_base.cast();

        let new_base = data.add(len).cast::<u8>();

        unsafe {
            core::arch::asm!(
                "mov.u64    %rust_cuda_dynamic_shared, {new_base};",
                new_base = in(reg64) new_base,
            );
        }

        let shared = core::ptr::slice_from_raw_parts_mut(data, len);

        inner(&mut Self { shared })
    }
}

#[doc(hidden)]
#[cfg(all(not(feature = "host"), target_os = "cuda"))]
/// # Safety
///
/// The thread-block shared dynamic memory must be initialised once and
/// only once per kernel.
pub unsafe fn init() {
    unsafe {
        core::arch::asm!(".reg .u64    %rust_cuda_dynamic_shared;");
        core::arch::asm!(
            "cvta.shared.u64    %rust_cuda_dynamic_shared, rust_cuda_dynamic_shared_base;",
        );
    }
}

#[cfg(all(not(feature = "host"), target_os = "cuda"))]
core::arch::global_asm!(".extern .shared .align 8 .b8 rust_cuda_dynamic_shared_base[];");
