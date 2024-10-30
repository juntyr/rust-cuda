use core::alloc::Layout;

use const_type_layout::TypeGraphLayout;

#[expect(clippy::module_name_repetitions)]
#[repr(transparent)]
pub struct ThreadBlockSharedSlice<T: 'static + TypeGraphLayout> {
    shared: *mut [T],
}

impl<T: 'static + TypeGraphLayout> ThreadBlockSharedSlice<T> {
    #[cfg(feature = "host")]
    #[must_use]
    pub fn new_uninit_with_len(len: usize) -> Self {
        Self {
            shared: Self::dangling_slice_with_len(len),
        }
    }

    #[cfg(feature = "host")]
    #[must_use]
    pub fn with_len(mut self, len: usize) -> Self {
        self.shared = Self::dangling_slice_with_len(len);
        self
    }

    #[cfg(feature = "host")]
    #[must_use]
    pub fn with_len_mut(&mut self, len: usize) -> &mut Self {
        self.shared = Self::dangling_slice_with_len(len);
        self
    }

    #[cfg(feature = "host")]
    fn dangling_slice_with_len(len: usize) -> *mut [T] {
        core::ptr::slice_from_raw_parts_mut(core::ptr::NonNull::dangling().as_ptr(), len)
    }

    #[must_use]
    pub fn len(&self) -> usize {
        core::ptr::metadata(self.shared)
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[must_use]
    pub fn layout(&self) -> Layout {
        // Safety: the length of self.shared is always initialised
        unsafe { Layout::for_value_raw(self.shared) }
    }

    #[cfg(feature = "device")]
    #[must_use]
    pub const fn as_mut_ptr(&self) -> *mut T {
        self.shared.cast()
    }

    #[cfg(feature = "device")]
    #[must_use]
    pub const fn as_mut_slice_ptr(&self) -> *mut [T] {
        self.shared
    }

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
        self.shared.get_unchecked_mut(index)
    }
}

#[cfg(feature = "device")]
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

#[cfg(feature = "device")]
/// # Safety
///
/// The thread-block shared dynamic memory must be initialised once and
/// only once per kernel.
pub unsafe fn init() {
    #[expect(clippy::multiple_unsafe_ops_per_block)]
    unsafe {
        core::arch::asm!(".reg .u64    %rust_cuda_dynamic_shared;");
        core::arch::asm!(
            "cvta.shared.u64    %rust_cuda_dynamic_shared, rust_cuda_dynamic_shared_base;",
        );
    }
}

#[cfg(feature = "device")]
core::arch::global_asm!(".extern .shared .align 8 .b8 rust_cuda_dynamic_shared_base[];");

#[cfg(feature = "host")]
pub struct SharedMemorySize {
    last_align: usize,
    total_size: usize,
}

#[cfg(feature = "host")]
impl SharedMemorySize {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            // we allocate the shared memory with an alignment of 8
            last_align: 8,
            total_size: 0,
        }
    }

    pub fn add(&mut self, layout: core::alloc::Layout) {
        if layout.align() > self.last_align {
            // in the worst case, we are one element of the smaller alignment
            //  into the larger alignment, so we need to pad the entire rest
            let pessimistic_padding = layout.align() - self.last_align;

            self.total_size += pessimistic_padding;
        }

        self.last_align = layout.align();
        self.total_size += layout.size();
    }

    pub const fn total(self) -> usize {
        self.total_size
    }
}
