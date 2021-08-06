use proc_macro2::TokenStream;

/// Check that the CPU and NVIDIA GPU architecture are sufficiently
///  aligned such that FFI-safe types have the same bit representation
///  on both sides.
///
/// ```
/// #[deny(improper_ctypes)]
/// extern "C" { static BOOL: bool; }
/// ```
///
/// ```compile_fail
/// #[deny(improper_ctypes)]
/// extern "C" { static CHAR: char; }
/// ```
///
/// ```compile_fail
/// #[deny(improper_ctypes)]
/// extern "C" { static STR: &str; }
/// ```
///
/// ```
/// #[deny(improper_ctypes)]
/// extern "C" { static F32: f32; }
/// ```
///
/// ```
/// #[deny(improper_ctypes)]
/// extern "C" { static F64: f64; }
/// ```
///
/// ```
/// #[deny(improper_ctypes)]
/// extern "C" { static I8: i8; }
/// ```
///
/// ```
/// #[deny(improper_ctypes)]
/// extern "C" { static I16: i16; }
/// ```
///
/// ```
/// #[deny(improper_ctypes)]
/// extern "C" { static I32: i32; }
/// ```
///
/// ```
/// #[deny(improper_ctypes)]
/// extern "C" { static I64: i64; }
/// ```
///
/// ```compile_fail
/// #[deny(improper_ctypes)]
/// extern "C" { static I128: i128; }
/// ```
///
/// ```
/// #[deny(improper_ctypes)]
/// extern "C" { static ISIZE: isize; }
/// ```
///
/// ```
/// #[deny(improper_ctypes)]
/// extern "C" { static U8: u8; }
/// ```
///
/// ```
/// #[deny(improper_ctypes)]
/// extern "C" { static U16: u16; }
/// ```
///
/// ```
/// #[deny(improper_ctypes)]
/// extern "C" { static U32: u32; }
/// ```
///
/// ```
/// #[deny(improper_ctypes)]
/// extern "C" { static U64: u64; }
/// ```
///
/// ```compile_fail
/// #[deny(improper_ctypes)]
/// extern "C" { static U128: u128; }
/// ```
///
/// ```
/// #[deny(improper_ctypes)]
/// extern "C" { static USIZE: usize; }
/// ```
///
/// ```
/// #[deny(improper_ctypes)]
/// extern "C" { static CONST_PTR: *const u8; }
/// ```
///
/// ```
/// #[deny(improper_ctypes)]
/// extern "C" { static MUT_PTR: *mut u8; }
/// ```
pub(in super::super) fn quote_arch_checks() -> TokenStream {
    let types: syn::punctuated::Punctuated<syn::Ident, syn::token::Comma> = syn::parse_quote! {
        bool,f32,f64,i8,i16,i32,i64,isize,u8,u16,u32,u64,usize
    };
    let sizes = [1, 4, 8, 1, 2, 4, 8, 8, 1, 2, 4, 8, 8];
    let aligns = [1, 4, 8, 1, 2, 4, 8, 8, 1, 2, 4, 8, 8];

    let mut size_align_checks = Vec::new();

    for ((ty, size), align) in types.iter().zip(sizes.iter()).zip(aligns.iter()) {
        size_align_checks.push(quote_size_check(ty, *size, false));
        size_align_checks.push(quote_align_check(ty, *align, false));

        size_align_checks.push(quote_size_check(ty, *size, true));
        size_align_checks.push(quote_align_check(ty, *size, true));
    }

    quote! {
        #[cfg(not(target_endian = "little"))]
        compile_error!("Both the CPU and NVIDIA GPU must use little-endian representation.");

        #[cfg(not(target_pointer_width = "64"))]
        compile_error!("Both the CPU and NVIDIA GPU must have 64bit pointers.");

        #(#size_align_checks)*
    }
}

fn quote_size_check(ty: &syn::Ident, bytes: usize, vector: bool) -> TokenStream {
    let error = quote::format_ident!(
        "Both_the_CPU_and_NVIDIA_GPU_must_have_{}b_sized_{}{}",
        bytes * 8,
        ty,
        if vector { "vectors" } else { "" },
    );

    let (ty, bytes) = if vector {
        (quote! { (#ty, #ty) }, bytes * 2)
    } else {
        (quote! { #ty }, bytes)
    };

    quote! {
        #[allow(dead_code, non_camel_case_types)]
        enum #error {}
        const _: [#error; 1 - {
            const ASSERT: bool = (::core::mem::size_of::<#ty>() == #bytes); ASSERT
        } as usize] = [];
    }
}

fn quote_align_check(ty: &syn::Ident, bytes: usize, vector: bool) -> TokenStream {
    let error = quote::format_ident!(
        "Both_the_CPU_and_NVIDIA_GPU_must_have_{}b_aligned_{}{}",
        bytes * 8,
        ty,
        if vector { "vectors" } else { "" },
    );

    let ty = if vector {
        quote! { (#ty, #ty) }
    } else {
        quote! { #ty }
    };

    quote! {
        #[allow(dead_code, non_camel_case_types)]
        enum #error {}
        const _: [#error; 1 - {
            const ASSERT: bool = (::core::mem::align_of::<#ty>() == #bytes); ASSERT
        } as usize] = [];
    }
}
