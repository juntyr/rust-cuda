#![allow(missing_docs)] // FIXME: use expect
#![no_std]
#![cfg_attr(target_os = "cuda", feature(abi_ptx))]
#![cfg_attr(target_os = "cuda", feature(alloc_error_handler))]
#![cfg_attr(target_os = "cuda", feature(asm_experimental_arch))]
#![feature(cfg_version)]
#![feature(type_alias_impl_trait)]
#![feature(decl_macro)]

extern crate alloc;

#[rust_cuda::kernel::kernel(pub use link! for impl)]
#[kernel(allow(ptx::local_memory_use))]
pub fn kernel<'a, 'b>(
    a: &'a rust_cuda::kernel::param::PerThreadShallowCopy<u32>,
    b: &'b rust_cuda::kernel::param::ShallowInteriorMutable<core::sync::atomic::AtomicU32>,
    c: &rust_cuda::kernel::param::DeepPerThreadBorrow<
        Option<
            rust_cuda::utils::adapter::RustToCudaWithPortableBitCopySemantics<
                core::num::NonZeroU32,
            >,
        >,
    >,
) {
    let _ = (a, c);
    b.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
}

#[cfg(target_os = "cuda")]
mod cuda_prelude {
    use rust_cuda::device::alloc::PTXAllocator;

    #[global_allocator]
    static _GLOBAL_ALLOCATOR: PTXAllocator = PTXAllocator;

    #[panic_handler]
    fn panic(info: &::core::panic::PanicInfo) -> ! {
        // pretty format and print the panic message
        //  but don't allow dynamic formatting
        rust_cuda::device::utils::pretty_print_panic_info(info, false);

        // Safety: no mutable data is shared with the kernel
        unsafe { rust_cuda::device::utils::exit() }
    }

    #[alloc_error_handler]
    #[track_caller]
    fn alloc_error_handler(layout: ::core::alloc::Layout) -> ! {
        rust_cuda::device::utils::pretty_print_alloc_error(layout);

        // Safety: no mutable data is shared with the kernel
        unsafe { rust_cuda::device::utils::exit() }
    }
}
