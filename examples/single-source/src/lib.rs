#![allow(missing_docs)] // FIXME: use expect
#![allow(dead_code)] // FIXME: use expect
#![cfg_attr(target_os = "cuda", no_std)]
#![cfg_attr(target_os = "cuda", feature(abi_ptx))]
#![cfg_attr(target_os = "cuda", feature(alloc_error_handler))]
#![cfg_attr(target_os = "cuda", feature(asm_experimental_arch))]
#![feature(const_type_name)]
#![feature(cfg_version)]
#![feature(type_alias_impl_trait)]
#![feature(decl_macro)]
#![recursion_limit = "1024"]

extern crate alloc;

#[repr(C)]
#[derive(rc::deps::const_type_layout::TypeLayout)]
#[layout(crate = "rc::deps::const_type_layout")]
pub struct Dummy(i32);

#[derive(Clone, rc::lend::LendRustToCuda)]
#[cuda(crate = "rc")]
pub struct Wrapper<T> {
    #[cuda(embed)]
    inner: T,
}

#[repr(C)]
#[derive(Clone, rc::lend::LendRustToCuda)]
#[cuda(crate = "rc")]
pub struct Empty([u8; 0]);

#[repr(C)]
#[derive(rc::deps::const_type_layout::TypeLayout)]
#[layout(crate = "rc::deps::const_type_layout")]
pub struct Tuple(u32, i32);

#[repr(C)]
#[derive(Copy, Clone, rc::deps::const_type_layout::TypeLayout)]
#[layout(crate = "rc::deps::const_type_layout")]
pub struct Triple(i32, i32, i32);

#[rc::kernel::kernel(pub use link! for impl)]
#[kernel(crate = "rc")]
#[kernel(
    allow(ptx::double_precision_use),
    forbid(ptx::local_memory_use, ptx::register_spills)
)]
pub fn kernel<
    'a,
    T: 'static
        + Send
        + Sync
        + Clone
        + rc::lend::RustToCuda<
            CudaRepresentation: rc::safety::StackOnly,
            CudaAllocation: rc::alloc::EmptyCudaAlloc,
        >
        + rc::safety::StackOnly,
>(
    _x: &rc::kernel::param::PerThreadShallowCopy<Dummy>,
    _z: &rc::kernel::param::DeepPerThreadBorrow<Wrapper<T>>,
    _v @ _w: &'a rc::kernel::param::ShallowInteriorMutable<core::sync::atomic::AtomicU64>,
    _: rc::kernel::param::DeepPerThreadBorrow<Wrapper<T>>,
    q @ Triple(s, mut __t, _u): rc::kernel::param::PerThreadShallowCopy<Triple>,
    shared3: &mut rc::utils::shared::ThreadBlockShared<u32>,
    dynamic: &mut rc::utils::shared::ThreadBlockSharedSlice<Dummy>,
) {
    let shared = rc::utils::shared::ThreadBlockShared::<[Tuple; 3]>::new_uninit();
    let shared2 = rc::utils::shared::ThreadBlockShared::<[Tuple; 3]>::new_uninit();

    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::multiple_unsafe_ops_per_block
    )]
    // Safety: [a]
    // 1. index is in bounds
    // 2. all threads withe the same result, so no write race is observable
    unsafe {
        (*shared.index_mut_unchecked(1)).0 = (f64::from(s) * 2.0) as u32;
    }
    #[expect(clippy::multiple_unsafe_ops_per_block)]
    // Safety: same as in [a]
    unsafe {
        (*shared2.index_mut_unchecked(2)).1 = q.0 + q.1 + q.2;
    }

    // Safety: all threads withe the same result, so no write race is observable
    unsafe {
        *shared3.as_mut_ptr() = 12;
    }

    let index = rc::device::thread::Thread::this().index();
    if index < dynamic.len() {
        #[expect(clippy::multiple_unsafe_ops_per_block)]
        // Safety:
        // 1. index has been checked to be in bounds
        // 2. each location is written to by only one thread
        unsafe {
            *dynamic.index_mut_unchecked(index) = Dummy(42);
        }
    }
}

#[cfg(not(target_os = "cuda"))]
mod host {
    // Link several instances of the generic CUDA kernel
    struct KernelPtx<'a, T>(std::marker::PhantomData<&'a T>);
    crate::link! { impl kernel<'a, crate::Empty> for KernelPtx }
    crate::link! { impl kernel<'a, rc::utils::adapter::RustToCudaWithPortableBitCopySemantics<u64>> for KernelPtx }
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
