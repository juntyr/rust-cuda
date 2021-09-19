#![deny(clippy::pedantic)]
#![cfg_attr(target_os = "cuda", no_std)]
#![cfg_attr(target_os = "cuda", no_main)]
#![cfg_attr(target_os = "cuda", feature(abi_ptx))]
#![cfg_attr(target_os = "cuda", feature(alloc_error_handler))]
#![cfg_attr(target_os = "cuda", feature(panic_info_message))]
#![cfg_attr(target_os = "cuda", feature(stdsimd))]
#![cfg_attr(target_os = "cuda", feature(asm))]

extern crate alloc;

#[cfg(not(target_os = "cuda"))]
fn main() {}

#[repr(C)]
pub struct Dummy(i32);

#[derive(rust_cuda::common::LendRustToCuda)]
#[allow(dead_code)]
pub struct Wrapper<T: rust_cuda::common::RustToCuda> {
    #[r2cEmbed]
    inner: T,
}

#[derive(rust_cuda::common::LendRustToCuda)]
pub struct Empty([u8; 0]);

#[repr(C)]
pub struct Tuple(u32, i32);

#[rust_cuda::common::kernel(use link_kernel! as impl Kernel<KernelArgs> for Launcher)]
pub fn kernel<'a, T: rust_cuda::common::RustToCuda>(
    #[kernel(pass = SafeDeviceCopy)] _x: &Dummy,
    #[kernel(pass = LendRustToCuda, jit)] _y: &mut ShallowCopy<Wrapper<T>>,
    #[kernel(pass = LendRustToCuda)] _z: &ShallowCopy<Wrapper<T>>,
    #[kernel(pass = SafeDeviceCopy, jit)] _v @ _w: &'a core::sync::atomic::AtomicU64,
    #[kernel(pass = LendRustToCuda)] _: Wrapper<T>,
    #[kernel(pass = SafeDeviceCopy)] Tuple(_s, mut __t): Tuple,
) where
    <T as rust_cuda::common::RustToCuda>::CudaRepresentation: rust_cuda::memory::StackOnly,
{
}

#[cfg(not(target_os = "cuda"))]
mod host {
    use super::{Kernel, KernelArgs};

    #[allow(dead_code)]
    struct Launcher<T: rust_cuda::common::RustToCuda>(core::marker::PhantomData<T>);

    link_kernel!(crate::Empty);
    link_kernel!(rust_cuda::utils::device_copy::SafeDeviceCopyWrapper<u64>);

    impl<T: rust_cuda::common::RustToCuda> rust_cuda::host::Launcher for Launcher<T> {
        type KernelTraitObject = dyn Kernel<T>;

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
    use core::arch::nvptx;

    use rust_cuda::device::utils;

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
