use proc_macro2::TokenStream;

pub(in super::super) fn quote_arch_checks() -> TokenStream {
    quote! {
        #[cfg(not(target_endian = "little"))]
        compile_error!("Both the CPU and NVIDIA GPU must use little-endian representation.");

        #[cfg(not(target_pointer_width = "64"))]
        compile_error!("Both the CPU and NVIDIA GPU must have 64bit pointers.");

        #[allow(dead_code, non_camel_case_types)]
        enum Both_the_CPU_and_NVIDIA_GPU_must_have_8b_aligned_u8 {}
        const _: [Both_the_CPU_and_NVIDIA_GPU_must_have_8b_aligned_u8; 1 - {
            const ASSERT: bool = (::core::mem::align_of::<u8>() == 1); ASSERT
        } as usize] = [];

        #[allow(dead_code, non_camel_case_types)]
        enum Both_the_CPU_and_NVIDIA_GPU_must_have_16b_aligned_u16 {}
        const _: [Both_the_CPU_and_NVIDIA_GPU_must_have_16b_aligned_u16; 1 - {
            const ASSERT: bool = (::core::mem::align_of::<u16>() == 2); ASSERT
        } as usize] = [];

        #[allow(dead_code, non_camel_case_types)]
        enum Both_the_CPU_and_NVIDIA_GPU_must_have_32b_aligned_u32 {}
        const _: [Both_the_CPU_and_NVIDIA_GPU_must_have_32b_aligned_u32; 1 - {
            const ASSERT: bool = (::core::mem::align_of::<u32>() == 4); ASSERT
        } as usize] = [];

        #[allow(dead_code, non_camel_case_types)]
        enum Both_the_CPU_and_NVIDIA_GPU_must_have_64b_aligned_u64 {}
        const _: [Both_the_CPU_and_NVIDIA_GPU_must_have_64b_aligned_u64; 1 - {
            const ASSERT: bool = (::core::mem::align_of::<u64>() == 8); ASSERT
        } as usize] = [];

        // i128 / u128 are not yet FFI safe:
        //   https://github.com/rust-lang/unsafe-code-guidelines/issues/119
        //
        // enum Both_the_CPU_and_NVIDIA_GPU_must_have_128b_aligned_u128 {}
        // const _: [Both_the_CPU_and_NVIDIA_GPU_must_have_128b_aligned_u128; 1 - {
        //     const ASSERT: bool = (::core::mem::align_of::<u128>() == 16); ASSERT
        // } as usize] = [];

        #[allow(dead_code, non_camel_case_types)]
        enum Both_the_CPU_and_NVIDIA_GPU_must_have_8b_aligned_u8_vectors {}
        const _: [Both_the_CPU_and_NVIDIA_GPU_must_have_8b_aligned_u8_vectors; 1 - {
            const ASSERT: bool = (::core::mem::align_of::<(u8, u8)>() == 1); ASSERT
        } as usize] = [];

        #[allow(dead_code, non_camel_case_types)]
        enum Both_the_CPU_and_NVIDIA_GPU_must_have_16b_aligned_u16_vectors {}
        const _: [Both_the_CPU_and_NVIDIA_GPU_must_have_16b_aligned_u16_vectors; 1 - {
            const ASSERT: bool = (::core::mem::align_of::<(u16, u16)>() == 2); ASSERT
        } as usize] = [];

        #[allow(dead_code, non_camel_case_types)]
        enum Both_the_CPU_and_NVIDIA_GPU_must_have_32b_aligned_u32_vectors {}
        const _: [Both_the_CPU_and_NVIDIA_GPU_must_have_32b_aligned_u32_vectors; 1 - {
            const ASSERT: bool = (::core::mem::align_of::<(u32, u32)>() == 4); ASSERT
        } as usize] = [];

        #[allow(dead_code, non_camel_case_types)]
        enum Both_the_CPU_and_NVIDIA_GPU_must_have_64b_aligned_u64_vectors {}
        const _: [Both_the_CPU_and_NVIDIA_GPU_must_have_64b_aligned_u64_vectors; 1 - {
            const ASSERT: bool = (::core::mem::align_of::<(u64, u64)>() == 8); ASSERT
        } as usize] = [];
    }
}
