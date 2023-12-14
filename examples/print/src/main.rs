#![deny(clippy::pedantic)]
#![cfg_attr(target_os = "cuda", no_std)]
#![cfg_attr(target_os = "cuda", no_main)]
#![cfg_attr(target_os = "cuda", feature(abi_ptx))]
#![cfg_attr(target_os = "cuda", feature(alloc_error_handler))]
#![cfg_attr(target_os = "cuda", feature(asm_experimental_arch))]
#![cfg_attr(target_os = "cuda", feature(core_panic))]

extern crate alloc;

#[cfg(not(target_os = "cuda"))]
fn main() {}

#[rust_cuda::common::kernel(use link_kernel! as impl Kernel<KernelArgs, KernelPtx> for Launcher)]
#[kernel(deny(
    ptx::double_precision_use,
    ptx::local_memory_usage,
    ptx::register_spills,
    ptx::dynamic_stack_size
))]
pub fn kernel() {
    rust_cuda::device::utils::print(format_args!("println! from CUDA kernel"));
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
    fn alloc_error_handler(layout: ::core::alloc::Layout) -> ! {
        let (size, align) = (layout.size(), layout.align());

        ::core::panicking::panic_nounwind_fmt(
            format_args!("memory allocation of {size} bytes with alignment {align} failed\n"),
            true,
        )
    }
}
