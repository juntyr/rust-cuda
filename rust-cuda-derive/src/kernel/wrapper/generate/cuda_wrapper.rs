use proc_macro2::TokenStream;
use syn::spanned::Spanned;

use super::super::{
    super::{KERNEL_TYPE_USE_END_CANARY, KERNEL_TYPE_USE_START_CANARY},
    FuncIdent, FunctionInputs, ImplGenerics,
};

#[allow(clippy::too_many_lines)]
pub(in super::super) fn quote_cuda_wrapper(
    crate_path: &syn::Path,
    inputs @ FunctionInputs {
        func_inputs,
        func_input_cuda_types,
    }: &FunctionInputs,
    func @ FuncIdent {
        func_ident,
        func_ident_hash,
        ..
    }: &FuncIdent,
    impl_generics @ ImplGenerics {
        impl_generics: generics,
        ..
    }: &ImplGenerics,
    func_attrs: &[syn::Attribute],
    func_params: &[syn::Ident],
) -> TokenStream {
    let (ffi_inputs, ffi_types) =
        specialise_ffi_input_types(crate_path, inputs, func, impl_generics);

    let func_layout_params = func_params
        .iter()
        .map(|ident| {
            syn::Ident::new(
                &format!("__{func_ident_hash}_{ident}_layout").to_uppercase(),
                ident.span(),
            )
        })
        .collect::<Vec<_>>();

    let ffi_param_ptx_jit_wrap = func_inputs
        .iter().zip(func_input_cuda_types.iter()).enumerate()
        .rev()
        .fold(quote! {
            #func_ident(#(#func_params),*)
        }, |inner, (_i, (arg, _ptx_jit))| match arg {
            syn::FnArg::Typed(syn::PatType {
                pat,
                ty,
                ..
            }) => {
                // Emit PTX JIT load markers
                // let ptx_jit_load = if ptx_jit.0 {
                //     quote! {
                //         #crate_path::ptx_jit::PtxJITConstLoad!([#i] => #pat.as_ref())
                //     }
                // } else { quote! {} };

                let specialised_ty = quote::quote_spanned! { ty.span()=>
                    #crate_path::device::specialise_kernel_type!(#ty for #generics in #func_ident)
                };

                quote::quote_spanned! { ty.span()=>
                    <#specialised_ty as #crate_path::common::CudaKernelParameter>::with_ffi_as_device(
                        #pat, |#pat| { #inner }
                    )
                }
            },
            syn::FnArg::Receiver(_) => unreachable!(),
        });

    quote! {
        #[cfg(target_os = "cuda")]
        #[#crate_path::device::specialise_kernel_function(#func_ident)]
        #[no_mangle]
        #(#func_attrs)*
        pub unsafe extern "ptx-kernel" fn #func_ident_hash(#(#ffi_inputs),*) {
            unsafe {
                ::core::arch::asm!(#KERNEL_TYPE_USE_START_CANARY);
            }
            #(
                #[no_mangle]
                static #func_layout_params: [
                    u8; #crate_path::const_type_layout::serialised_type_graph_len::<#ffi_types>()
                ] = #crate_path::const_type_layout::serialise_type_graph::<#ffi_types>();

                unsafe { ::core::ptr::read_volatile(&#func_layout_params[0]) };
            )*
            unsafe {
                ::core::arch::asm!(#KERNEL_TYPE_USE_END_CANARY);
            }

            #[deny(improper_ctypes)]
            mod __rust_cuda_ffi_safe_assert {
                #[allow(unused_imports)]
                use super::*;

                extern "C" { #(
                    #[allow(dead_code)]
                    static #func_params: #ffi_types;
                )* }
            }

            #ffi_param_ptx_jit_wrap
        }
    }
}

fn specialise_ffi_input_types(
    crate_path: &syn::Path,
    FunctionInputs { func_inputs, .. }: &FunctionInputs,
    FuncIdent { func_ident, .. }: &FuncIdent,
    ImplGenerics { impl_generics, .. }: &ImplGenerics,
) -> (Vec<syn::FnArg>, Vec<syn::Type>) {
    func_inputs
        .iter()
        .map(|arg| match arg {
            syn::FnArg::Typed(
                syn::PatType {
                    attrs,
                    pat,
                    colon_token,
                    ty,
                },
            ) => {
                let specialised_ty = quote::quote_spanned! { ty.span()=>
                    #crate_path::device::specialise_kernel_type!(#ty for #impl_generics in #func_ident)
                };

                let ffi_ty: syn::Type = syn::parse_quote_spanned! { ty.span()=>
                    <#specialised_ty as #crate_path::common::CudaKernelParameter>::FfiType<'static, 'static>
                };

                let ffi_param = syn::FnArg::Typed(syn::PatType {
                    attrs: attrs.clone(),
                    ty: Box::new(ffi_ty.clone()),
                    pat: pat.clone(),
                    colon_token: *colon_token,
                });

                (ffi_param, ffi_ty)
            },
            syn::FnArg::Receiver(_) => unreachable!(),
        })
        .unzip()
}
