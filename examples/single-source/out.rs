#![feature(prelude_import)]
#![deny(clippy::pedantic)]
#![no_std]
#![no_main]
#![feature(abi_ptx)]
#![feature(alloc_error_handler)]
#![feature(panic_info_message)]
#[prelude_import]
use core::prelude::rust_2018::*;
#[macro_use]
extern crate core;
#[macro_use]
extern crate compiler_builtins;
extern crate alloc;
struct Dummy(alloc::vec::Vec<u128>);
unsafe impl rust_cuda::rustacuda_core::DeviceCopy for Dummy {}
#[cfg(target_os = "cuda")]
#[deny(improper_ctypes)]
extern "C" {}
unsafe trait KernelArgs {
    type __T_0;
}
unsafe impl KernelArgs for () {
    type __T_0 = Dummy;
}
#[cfg(target_os = "cuda")]
#[no_mangle]
pub unsafe extern "ptx-kernel" fn kernel_kernel(
    x: rust_cuda::common::DeviceBoxConst<<() as KernelArgs>::__T_0>,
) {
    #[allow(dead_code, non_camel_case_types)]
    enum Both_the_CPU_and_NVIDIA_GPU_must_have_8b_aligned_u8 {}
    const _: [Both_the_CPU_and_NVIDIA_GPU_must_have_8b_aligned_u8; 1 - {
        const ASSERT: bool = (::core::mem::align_of::<u8>() == 1);
        ASSERT
    } as usize] = [];
    #[allow(dead_code, non_camel_case_types)]
    enum Both_the_CPU_and_NVIDIA_GPU_must_have_16b_aligned_u16 {}
    const _: [Both_the_CPU_and_NVIDIA_GPU_must_have_16b_aligned_u16; 1 - {
        const ASSERT: bool = (::core::mem::align_of::<u16>() == 2);
        ASSERT
    } as usize] = [];
    #[allow(dead_code, non_camel_case_types)]
    enum Both_the_CPU_and_NVIDIA_GPU_must_have_32b_aligned_u32 {}
    const _: [Both_the_CPU_and_NVIDIA_GPU_must_have_32b_aligned_u32; 1 - {
        const ASSERT: bool = (::core::mem::align_of::<u32>() == 4);
        ASSERT
    } as usize] = [];
    #[allow(dead_code, non_camel_case_types)]
    enum Both_the_CPU_and_NVIDIA_GPU_must_have_64b_aligned_u64 {}
    const _: [Both_the_CPU_and_NVIDIA_GPU_must_have_64b_aligned_u64; 1 - {
        const ASSERT: bool = (::core::mem::align_of::<u64>() == 8);
        ASSERT
    } as usize] = [];
    #[allow(dead_code, non_camel_case_types)]
    enum Both_the_CPU_and_NVIDIA_GPU_must_have_8b_aligned_u8_vectors {}
    const _: [Both_the_CPU_and_NVIDIA_GPU_must_have_8b_aligned_u8_vectors; 1 - {
        const ASSERT: bool = (::core::mem::align_of::<(u8, u8)>() == 1);
        ASSERT
    } as usize] = [];
    #[allow(dead_code, non_camel_case_types)]
    enum Both_the_CPU_and_NVIDIA_GPU_must_have_16b_aligned_u16_vectors {}
    const _: [Both_the_CPU_and_NVIDIA_GPU_must_have_16b_aligned_u16_vectors; 1 - {
        const ASSERT: bool = (::core::mem::align_of::<(u16, u16)>() == 2);
        ASSERT
    } as usize] = [];
    #[allow(dead_code, non_camel_case_types)]
    enum Both_the_CPU_and_NVIDIA_GPU_must_have_32b_aligned_u32_vectors {}
    const _: [Both_the_CPU_and_NVIDIA_GPU_must_have_32b_aligned_u32_vectors; 1 - {
        const ASSERT: bool = (::core::mem::align_of::<(u32, u32)>() == 4);
        ASSERT
    } as usize] = [];
    #[allow(dead_code, non_camel_case_types)]
    enum Both_the_CPU_and_NVIDIA_GPU_must_have_64b_aligned_u64_vectors {}
    const _: [Both_the_CPU_and_NVIDIA_GPU_must_have_64b_aligned_u64_vectors; 1 - {
        const ASSERT: bool = (::core::mem::align_of::<(u64, u64)>() == 8);
        ASSERT
    } as usize] = [];
    #[allow(dead_code, non_camel_case_types)]
    enum Kernel_parameter_x_must_fit_into_64b_or_be_a_reference {}
    const _: [Kernel_parameter_x_must_fit_into_64b_or_be_a_reference; 1 - {
        const ASSERT: bool = (::core::mem::size_of::<
            rust_cuda::common::DeviceBoxConst<<() as KernelArgs>::__T_0>,
        >() <= 8);
        ASSERT
    } as usize] = [];
    #[deny(improper_ctypes)]
    mod __rust_cuda_ffi_safe_assert {
        use super::KernelArgs;
        extern "C" {
            #[allow(dead_code)]
            static x: rust_cuda::common::DeviceBoxConst<<() as KernelArgs>::__T_0>;
        }
    }
    if false {
        #[allow(dead_code)]
        fn assert_impl_devicecopy<T: rust_cuda::rustacuda_core::DeviceCopy>(_val: &T) {}
        assert_impl_devicecopy(&x);
    };
    {
        let x = x.as_ref();
        kernel(x)
    }
}
#[cfg(target_os = "cuda")]
fn kernel(x: &Dummy) {}
#[cfg(target_os = "cuda")]
mod cuda_prelude {
    use rust_cuda::device::{nvptx, utils};
    static _GLOBAL_ALLOCATOR: utils::PTXAllocator = utils::PTXAllocator;
    const _: () = {
        #[rustc_std_internal_symbol]
        unsafe fn __rg_alloc(arg0: usize, arg1: usize) -> *mut u8 {
            ::core::alloc::GlobalAlloc::alloc(
                &_GLOBAL_ALLOCATOR,
                ::core::alloc::Layout::from_size_align_unchecked(arg0, arg1),
            ) as *mut u8
        }
        #[rustc_std_internal_symbol]
        unsafe fn __rg_dealloc(arg0: *mut u8, arg1: usize, arg2: usize) -> () {
            ::core::alloc::GlobalAlloc::dealloc(
                &_GLOBAL_ALLOCATOR,
                arg0 as *mut u8,
                ::core::alloc::Layout::from_size_align_unchecked(arg1, arg2),
            )
        }
        #[rustc_std_internal_symbol]
        unsafe fn __rg_realloc(arg0: *mut u8, arg1: usize, arg2: usize, arg3: usize) -> *mut u8 {
            ::core::alloc::GlobalAlloc::realloc(
                &_GLOBAL_ALLOCATOR,
                arg0 as *mut u8,
                ::core::alloc::Layout::from_size_align_unchecked(arg1, arg2),
                arg3,
            ) as *mut u8
        }
        #[rustc_std_internal_symbol]
        unsafe fn __rg_alloc_zeroed(arg0: usize, arg1: usize) -> *mut u8 {
            ::core::alloc::GlobalAlloc::alloc_zeroed(
                &_GLOBAL_ALLOCATOR,
                ::core::alloc::Layout::from_size_align_unchecked(arg0, arg1),
            ) as *mut u8
        }
    };
    #[panic_handler]
    fn panic(_panic_info: &::core::panic::PanicInfo) -> ! {
        unsafe { nvptx::trap() }
    }
    #[alloc_error_handler]
    fn alloc_error_handler(_: core::alloc::Layout) -> ! {
        unsafe { nvptx::trap() }
    }
}
