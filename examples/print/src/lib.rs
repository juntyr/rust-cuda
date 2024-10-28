#![allow(missing_docs)] // FIXME: use expect
#![no_std]
#![cfg_attr(target_os = "cuda", feature(abi_ptx))]
#![cfg_attr(target_os = "cuda", feature(alloc_error_handler))]
#![cfg_attr(target_os = "cuda", feature(asm_experimental_arch))]
#![feature(const_type_name)]
#![feature(cfg_version)]
#![feature(type_alias_impl_trait)]
#![feature(decl_macro)]

extern crate alloc;

#[derive(Copy, Clone, rust_cuda::deps::const_type_layout::TypeLayout)]
#[layout(crate = "rust_cuda::deps::const_type_layout")]
#[repr(C)]
pub enum Action {
    Print,
    Panic,
    AllocError,
}

#[rust_cuda::kernel::kernel(pub use link! for impl)]
#[kernel(allow(ptx::local_memory_use))]
pub fn kernel(action: rust_cuda::kernel::param::PerThreadShallowCopy<Action>) {
    match action {
        Action::Print => rust_cuda::device::utils::println!("println! from CUDA kernel"),
        #[allow(clippy::panic)] // we want to demonstrate a panic
        Action::Panic => panic!("panic! from CUDA kernel"),
        Action::AllocError => {
            ::alloc::alloc::handle_alloc_error(::core::alloc::Layout::new::<i8>())
        },
    }
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
