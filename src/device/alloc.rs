use alloc::alloc::{GlobalAlloc, Layout};
#[cfg(all(feature = "device", not(doc)))]
use core::arch::nvptx;

/// Memory allocator using CUDA malloc/free
pub struct PTXAllocator;

unsafe impl GlobalAlloc for PTXAllocator {
    #[allow(clippy::inline_always)]
    #[inline(always)]
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        nvptx::malloc(layout.size()).cast()
    }

    #[allow(clippy::inline_always)]
    #[inline(always)]
    unsafe fn dealloc(&self, ptr: *mut u8, _layout: Layout) {
        nvptx::free(ptr.cast());
    }
}
