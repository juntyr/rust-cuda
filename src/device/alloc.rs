use alloc::alloc::{GlobalAlloc, Layout};
#[cfg(target_os = "cuda")]
use core::arch::nvptx;

/// Memory allocator using CUDA malloc/free
pub struct PTXAllocator;

unsafe impl GlobalAlloc for PTXAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        nvptx::malloc(layout.size()).cast()
    }

    unsafe fn dealloc(&self, ptr: *mut u8, _layout: Layout) {
        nvptx::free(ptr.cast());
    }
}
