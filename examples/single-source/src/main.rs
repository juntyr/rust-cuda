#![deny(clippy::pedantic)]
#![cfg_attr(target_os = "cuda", no_std)]
#![cfg_attr(target_os = "cuda", no_main)]
#![cfg_attr(target_os = "cuda", feature(abi_ptx))]
#![cfg_attr(target_os = "cuda", feature(alloc_error_handler))]
#![cfg_attr(target_os = "cuda", feature(stdsimd))]
#![cfg_attr(target_os = "cuda", feature(asm_experimental_arch))]
#![feature(const_type_name)]
#![feature(offset_of)]

extern crate alloc;

#[cfg(target_os = "cuda")]
use rc::utils::shared::r#static::ThreadBlockShared;

#[cfg(not(target_os = "cuda"))]
fn main() {}

#[repr(C)]
#[derive(rc::const_type_layout::TypeLayout)]
#[layout(crate = "rc::const_type_layout")]
pub struct Dummy(i32);

#[derive(rc::common::LendRustToCuda)]
#[cuda(crate = "rc")]
#[allow(dead_code)]
pub struct Wrapper<T> {
    #[cuda(embed)]
    inner: T,
}

#[derive(rc::common::LendRustToCuda)]
#[cuda(crate = "rc")]
pub struct Empty([u8; 0]);

#[repr(C)]
#[derive(rc::const_type_layout::TypeLayout)]
#[layout(crate = "rc::const_type_layout")]
pub struct Tuple(u32, i32);

#[rc::common::kernel(use link_kernel! as impl Kernel<KernelArgs, KernelPtx> for Launcher)]
#[kernel(crate = "rc")]
#[kernel(
    allow(ptx::double_precision_use),
    forbid(ptx::local_memory_usage, ptx::register_spills)
)]
pub fn kernel<'a, T: rc::common::RustToCuda>(
    #[kernel(pass = SafeDeviceCopy)] _x: &Dummy,
    #[kernel(pass = LendRustToCuda, jit)] _y: &mut ShallowCopy<Wrapper<T>>,
    #[kernel(pass = LendRustToCuda)] _z: &ShallowCopy<Wrapper<T>>,
    #[kernel(pass = SafeDeviceCopy, jit)] _v @ _w: &'a core::sync::atomic::AtomicU64,
    #[kernel(pass = LendRustToCuda)] _: Wrapper<T>,
    #[kernel(pass = SafeDeviceCopy)] Tuple(s, mut __t): Tuple,
    // #[kernel(pass = SafeDeviceCopy)] shared3: ThreadBlockShared<u32>,
) where
    T: rc::safety::StackOnly + rc::safety::NoAliasing,
    <T as rc::common::RustToCuda>::CudaRepresentation: rc::safety::StackOnly,
    <T as rc::common::RustToCuda>::CudaAllocation: rc::common::EmptyCudaAlloc,
{
    let shared: ThreadBlockShared<[Tuple; 3]> = ThreadBlockShared::new_uninit();
    let shared2: ThreadBlockShared<[Tuple; 3]> = ThreadBlockShared::new_uninit();

    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    unsafe {
        (*shared.index_mut_unchecked(1)).0 = (f64::from(s) * 2.0) as u32;
    }
    unsafe {
        (*shared2.index_mut_unchecked(2)).1 = 24;
    }
    // unsafe { core::arch::asm!("hi") }
    // unsafe {
    //     *shared3.as_mut_ptr() = 12;
    // }
}

#[cfg(not(target_os = "cuda"))]
mod host {
    #[allow(unused_imports)]
    use super::KernelArgs;
    use super::{Kernel, KernelPtx};

    #[allow(dead_code)]
    struct Launcher<T: rc::common::RustToCuda>(core::marker::PhantomData<T>);

    link_kernel!(crate::Empty);
    link_kernel!(rc::utils::device_copy::SafeDeviceCopyWrapper<u64>);

    impl<T: rc::common::RustToCuda> rc::host::Launcher for Launcher<T> {
        type CompilationWatcher = ();
        type KernelTraitObject = dyn Kernel<T>;

        fn get_launch_package(&mut self) -> rc::host::LaunchPackage<Self> {
            unimplemented!()
        }
    }
}

#[cfg(target_os = "cuda")]
mod cuda_prelude {
    use core::arch::nvptx;

    use rc::device::alloc::PTXAllocator;

    #[global_allocator]
    static _GLOBAL_ALLOCATOR: PTXAllocator = PTXAllocator;

    #[panic_handler]
    fn panic(_: &::core::panic::PanicInfo) -> ! {
        unsafe { nvptx::trap() }
    }

    #[alloc_error_handler]
    fn alloc_error_handler(_: core::alloc::Layout) -> ! {
        unsafe { nvptx::trap() }
    }
}
