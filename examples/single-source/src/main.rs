#![deny(clippy::pedantic)]
#![cfg_attr(target_os = "cuda", no_std)]
#![cfg_attr(target_os = "cuda", no_main)]
#![cfg_attr(target_os = "cuda", feature(abi_ptx))]
#![cfg_attr(target_os = "cuda", feature(alloc_error_handler))]
#![cfg_attr(target_os = "cuda", feature(panic_info_message))]

extern crate alloc;

#[cfg(not(target_os = "cuda"))]
fn main() {}

#[repr(C)]
pub struct Dummy(i32);

unsafe impl rust_cuda::rustacuda_core::DeviceCopy for Dummy {}

#[rust_cuda::common::kernel(use link_kernel! as impl Kernel<KernelArgs> for Launcher)]
pub fn kernel(#[kernel(pass = DeviceCopy)] x: &Dummy) {}

#[cfg(not(target_os = "cuda"))]
mod host {
    use super::{Kernel, KernelArgs};

    #[allow(dead_code)]
    struct Launcher;

    link_kernel!();

    impl rust_cuda::host::Launcher for Launcher {
        type KernelTraitObject = dyn Kernel;

        fn get_config(&self) -> rust_cuda::host::LaunchConfig {
            unimplemented!()
        }

        fn get_stream(&self) -> &rust_cuda::rustacuda::stream::Stream {
            unimplemented!()
        }

        fn get_kernel_mut(&mut self) -> &mut rust_cuda::host::TypedKernel<Self::KernelTraitObject> {
            unimplemented!()
        }
    }
}

#[cfg(target_os = "cuda")]
mod cuda_prelude {
    use rust_cuda::device::{nvptx, utils};

    #[global_allocator]
    static _GLOBAL_ALLOCATOR: utils::PTXAllocator = utils::PTXAllocator;

    #[panic_handler]
    fn panic(_panic_info: &::core::panic::PanicInfo) -> ! {
        unsafe { nvptx::trap() }
    }

    #[alloc_error_handler]
    fn alloc_error_handler(_: core::alloc::Layout) -> ! {
        unsafe { nvptx::trap() }
    }
}
