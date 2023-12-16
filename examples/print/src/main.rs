#![deny(clippy::pedantic)]
#![cfg_attr(target_os = "cuda", no_std)]
#![cfg_attr(target_os = "cuda", no_main)]
#![cfg_attr(target_os = "cuda", feature(abi_ptx))]
#![cfg_attr(target_os = "cuda", feature(alloc_error_handler))]
#![cfg_attr(target_os = "cuda", feature(asm_experimental_arch))]
#![feature(ptr_from_ref)]
#![feature(stdsimd)]
#![feature(c_str_literals)]

extern crate alloc;

#[cfg(not(target_os = "cuda"))]
fn main() {}

#[rust_cuda::common::kernel(use link_kernel! as impl Kernel<KernelArgs, KernelPtx> for Launcher)]
#[kernel(allow(ptx::local_memory_usage))]
pub fn kernel() {
    rust_cuda::device::utils::print(format_args!("println! from CUDA kernel"));

    ::alloc::alloc::handle_alloc_error(::core::alloc::Layout::new::<i8>());
}

#[cfg(not(target_os = "cuda"))]
mod host {
    #[allow(unused_imports)]
    use super::KernelArgs;
    use super::{Kernel, KernelPtx};

    #[allow(dead_code)]
    struct Launcher;

    link_kernel!();

    impl rust_cuda::host::Launcher for Launcher {
        type CompilationWatcher = ();
        type KernelTraitObject = dyn Kernel;

        fn get_launch_package(&mut self) -> rust_cuda::host::LaunchPackage<Self> {
            unimplemented!()
        }
    }
}

#[cfg(target_os = "cuda")]
mod cuda_prelude {
    use rust_cuda::device::alloc::PTXAllocator;

    #[global_allocator]
    static _GLOBAL_ALLOCATOR: PTXAllocator = PTXAllocator;

    #[panic_handler]
    fn panic(info: &::core::panic::PanicInfo) -> ! {
        rust_cuda::device::utils::print(format_args!("{info}\n"));

        rust_cuda::device::utils::abort()
    }

    #[alloc_error_handler]
    #[track_caller]
    fn alloc_error_handler(layout: ::core::alloc::Layout) -> ! {
        #[repr(C)]
        struct FormatArgs {
            size: usize,
            align: usize,
            file_len: u32,
            file_ptr: *const u8,
            line: u32,
            column: u32,
        }

        let location = ::core::panic::Location::caller();

        unsafe {
            ::core::arch::nvptx::vprintf(
                c"memory allocation of %llu bytes with alignment %llu failed at %.*s:%lu:%lu\n"
                    .as_ptr()
                    .cast(),
                #[allow(clippy::cast_possible_truncation)]
                ::core::ptr::from_ref(&FormatArgs {
                    size: layout.size(),
                    align: layout.align(),
                    file_len: location.file().len() as u32,
                    file_ptr: location.file().as_ptr(),
                    line: location.line(),
                    column: location.column(),
                })
                .cast(),
            );
        }

        rust_cuda::device::utils::abort()
    }
}
