#![deny(clippy::pedantic)]
#![cfg_attr(target_os = "cuda", no_std)]
#![cfg_attr(target_os = "cuda", no_main)]
#![cfg_attr(target_os = "cuda", feature(abi_ptx))]
#![cfg_attr(target_os = "cuda", feature(alloc_error_handler))]
#![cfg_attr(target_os = "cuda", feature(asm_experimental_arch))]
#![feature(const_type_name)]
#![feature(offset_of)]
#![feature(cfg_version)]
#![cfg_attr(not(version("1.76.0")), feature(c_str_literals))]
#![feature(type_alias_impl_trait)]
#![feature(associated_type_bounds)]
#![feature(decl_macro)]
#![recursion_limit = "1024"]

extern crate alloc;

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

#[repr(C)]
#[derive(rc::const_type_layout::TypeLayout)]
#[layout(crate = "rc::const_type_layout")]
pub struct Triple(i32, i32, i32);

#[rc::common::kernel(pub use link! for impl)]
#[kernel(crate = "rc")]
#[kernel(
    allow(ptx::double_precision_use),
    forbid(ptx::local_memory_usage, ptx::register_spills)
)]
pub fn kernel<
    'a,
    T: 'static
        + rc::common::RustToCuda<
            CudaRepresentation: rc::safety::StackOnly,
            CudaAllocation: rc::common::EmptyCudaAlloc,
        >
        + rc::safety::StackOnly
        + rc::safety::NoSafeAliasing,
>(
    _x: &rc::common::PerThreadShallowCopy<Dummy>,
    _z: &rc::common::SharedHeapPerThreadShallowCopy<Wrapper<T>>,
    _v @ _w: &'a rc::common::ShallowInteriorMutable<core::sync::atomic::AtomicU64>,
    _: rc::common::SharedHeapPerThreadShallowCopy<Wrapper<T>>,
    q @ Triple(s, mut __t, _u): rc::common::PerThreadShallowCopy<Triple>,
    shared3: &mut rc::utils::shared::r#static::ThreadBlockShared<u32>,
) {
    let shared = rc::utils::shared::r#static::ThreadBlockShared::<[Tuple; 3]>::new_uninit();
    let shared2 = rc::utils::shared::r#static::ThreadBlockShared::<[Tuple; 3]>::new_uninit();

    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    unsafe {
        (*shared.index_mut_unchecked(1)).0 = (f64::from(s) * 2.0) as u32;
    }
    unsafe {
        (*shared2.index_mut_unchecked(2)).1 = q.0 + q.1 + q.2;
    }

    unsafe {
        *shared3.as_mut_ptr() = 12;
    }
}

#[cfg(not(target_os = "cuda"))]
mod host {
    // use super::{link, kernel};

    // Link several instances of the generic CUDA kernel
    struct KernelPtx<'a, T>(std::marker::PhantomData<&'a T>);
    crate::link! { impl kernel<'a, crate::Empty> for KernelPtx }
    crate::link! { impl kernel<'a, rc::utils::device_copy::SafeDeviceCopyWrapper<u64>> for KernelPtx }
}

#[cfg(target_os = "cuda")]
mod cuda_prelude {
    use rc::device::alloc::PTXAllocator;

    #[global_allocator]
    static _GLOBAL_ALLOCATOR: PTXAllocator = PTXAllocator;

    #[panic_handler]
    fn panic(_: &::core::panic::PanicInfo) -> ! {
        rc::device::utils::abort()
    }

    #[alloc_error_handler]
    fn alloc_error_handler(_: core::alloc::Layout) -> ! {
        rc::device::utils::abort()
    }
}
