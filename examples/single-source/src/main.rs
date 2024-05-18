#![deny(clippy::pedantic)]
#![cfg_attr(target_os = "cuda", no_std)]
#![cfg_attr(target_os = "cuda", no_main)]
#![cfg_attr(target_os = "cuda", feature(abi_ptx))]
#![cfg_attr(target_os = "cuda", feature(alloc_error_handler))]
#![cfg_attr(target_os = "cuda", feature(stdarch_nvptx))]
#![cfg_attr(target_os = "cuda", feature(asm_experimental_arch))]
#![feature(const_type_name)]

extern crate alloc;

#[macro_use]
extern crate const_type_layout;

#[cfg(not(target_os = "cuda"))]
fn main() {}

#[repr(C)]
#[derive(TypeLayout)]
pub struct Dummy(i32);

#[derive(rust_cuda::common::LendRustToCuda)]
#[allow(dead_code)]
pub struct Wrapper<T> {
    #[cuda(embed)]
    inner: T,
}

#[derive(rust_cuda::common::LendRustToCuda)]
pub struct Empty([u8; 0]);

#[repr(C)]
#[derive(TypeLayout)]
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
    <T as rust_cuda::common::RustToCuda>::CudaRepresentation: rust_cuda::safety::StackOnly,
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
        type CompilationWatcher = ();
        type KernelTraitObject = dyn Kernel<T>;

        fn get_launch_package(&mut self) -> rust_cuda::host::LaunchPackage<Self> {
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
    fn panic(_: &::core::panic::PanicInfo) -> ! {
        unsafe { nvptx::trap() }
    }

    #[alloc_error_handler]
    fn alloc_error_handler(_: core::alloc::Layout) -> ! {
        unsafe { nvptx::trap() }
    }
}
