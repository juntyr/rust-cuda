use alloc::alloc::{AllocError, Allocator, Layout};
use core::ptr::NonNull;

pub struct UnifiedAllocator;

unsafe impl Allocator for UnifiedAllocator {
    #[cfg(feature = "host")]
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        if layout.size() == 0 {
            return Ok(NonNull::<[u8; 0]>::dangling());
        }

        match layout.align() {
            1 => alloc_unified_aligned::<u8>(layout.size()),
            2 => alloc_unified_aligned::<u16>(layout.size() >> 1),
            4 => alloc_unified_aligned::<u32>(layout.size() >> 2),
            8 => alloc_unified_aligned::<u64>(layout.size() >> 3),
            _ => Err(AllocError),
        }
    }

    #[cfg(not(feature = "host"))]
    fn allocate(&self, _layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        Err(AllocError)
    }

    #[cfg(feature = "host")]
    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        use rustacuda::memory::{cuda_free_unified, UnifiedPointer};

        if layout.size() == 0 {
            return;
        }

        let _ = cuda_free_unified(UnifiedPointer::wrap(ptr.as_ptr()));
    }

    #[cfg(not(feature = "host"))]
    unsafe fn deallocate(&self, _ptr: NonNull<u8>, _layout: Layout) {
        // no-op
    }
}

#[cfg(feature = "host")]
fn alloc_unified_aligned<T: rustacuda_core::DeviceCopy>(
    size: usize,
) -> Result<NonNull<[u8]>, AllocError> {
    use rustacuda::memory::cuda_malloc_unified;

    match unsafe { cuda_malloc_unified::<T>(size) } {
        Ok(mut ptr) => {
            let bytes: &mut [u8] = unsafe {
                core::slice::from_raw_parts_mut(
                    ptr.as_raw_mut().cast(),
                    size * core::mem::align_of::<T>(),
                )
            };

            NonNull::new(bytes).ok_or(AllocError)
        },
        Err(_) => Err(AllocError),
    }
}
