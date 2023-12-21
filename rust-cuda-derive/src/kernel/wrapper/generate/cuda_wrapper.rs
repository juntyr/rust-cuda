use proc_macro2::TokenStream;
use quote::quote_spanned;
use syn::spanned::Spanned;

use super::super::{
    super::{KERNEL_TYPE_USE_END_CANARY, KERNEL_TYPE_USE_START_CANARY},
    FuncIdent, FunctionInputs, ImplGenerics, InputCudaType,
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
    let (ptx_func_inputs, ptx_func_types) =
        specialise_ptx_func_inputs(crate_path, inputs, func, impl_generics);
    let ptx_func_unboxed_types =
        specialise_ptx_unboxed_types(crate_path, inputs, func, impl_generics);

    let func_layout_params = func_params
        .iter()
        .map(|ident| {
            syn::Ident::new(
                &format!("__{func_ident_hash}_{ident}_layout").to_uppercase(),
                ident.span(),
            )
        })
        .collect::<Vec<_>>();

    let ptx_func_input_unwrap = func_inputs
        .iter().zip(func_input_cuda_types.iter()).enumerate()
        .rev()
        .fold(quote! {
            #func_ident(#(#func_params),*)
        }, |inner, (i, (arg, (cuda_mode, ptx_jit)))| match arg {
            syn::FnArg::Typed(syn::PatType {
                pat,
                ty,
                ..
            }) => {
                // Emit PTX JIT load markers
                let ptx_jit_load = if ptx_jit.0 {
                    quote! {
                        #crate_path::ptx_jit::PtxJITConstLoad!([#i] => #pat.as_ref())
                    }
                } else { quote! {} };

                let arg_type = match &**ty {
                    syn::Type::Reference(syn::TypeReference { elem, .. }) => elem,
                    other => other,
                };
                let syn_type = quote::quote_spanned! { ty.span()=>
                    #crate_path::device::specialise_kernel_type!(#arg_type for #generics in #func_ident)
                };

                match cuda_mode {
                    InputCudaType::SafeDeviceCopy => if let syn::Type::Reference(
                        syn::TypeReference { and_token, .. }
                    ) = &**ty {
                        // DeviceCopy mode only supports immutable references
                        // TODO: ptx_jit_load should be here, not there
                        // also ptx_jit_load should not be enabled for interior mutability
                        quote! { { let #pat: #and_token #syn_type = #pat.as_ref().into_ref(); #inner } }
                    } else {
                        quote! { #ptx_jit_load; { let #pat: #syn_type = #pat.into_inner(); #inner } }
                    },
                    InputCudaType::LendRustToCuda => if let syn::Type::Reference(
                        syn::TypeReference { and_token, mutability, ..}
                    ) = &**ty {
                        if mutability.is_some() {
                            quote! {
                                #ptx_jit_load;
                                #crate_path::device::BorrowFromRust::with_borrow_from_rust_mut(
                                    #pat, |#pat: #and_token #mutability #crate_path::device::ShallowCopy<#syn_type>| { #inner },
                                )
                            }
                        } else {
                            quote! {
                                #ptx_jit_load;
                                #crate_path::device::BorrowFromRust::with_borrow_from_rust(
                                    #pat, |#pat: #and_token #crate_path::device::ShallowCopy<#syn_type>| { #inner },
                                )
                            }
                        }
                    } else {
                        quote! {
                            #ptx_jit_load;
                            #crate_path::device::BorrowFromRust::with_moved_from_rust(
                                #pat, |#pat: #syn_type| { #inner },
                            )
                        }
                    }
                }
            },
            syn::FnArg::Receiver(_) => unreachable!(),
        });

    quote! {
        #[cfg(target_os = "cuda")]
        #[#crate_path::device::specialise_kernel_function(#func_ident)]
        #[no_mangle]
        #(#func_attrs)*
        pub unsafe extern "ptx-kernel" fn #func_ident_hash(#(#ptx_func_inputs),*) {
            unsafe {
                ::core::arch::asm!(#KERNEL_TYPE_USE_START_CANARY);
            }
            #(
                #[no_mangle]
                static #func_layout_params: [
                    u8; #crate_path::const_type_layout::serialised_type_graph_len::<#ptx_func_types>()
                ] = #crate_path::const_type_layout::serialise_type_graph::<#ptx_func_types>();

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
                    static #func_params: #ptx_func_types;
                )* }
            }

            if false {
                #[allow(dead_code)]
                fn assert_impl_devicecopy<T: #crate_path::rustacuda_core::DeviceCopy>(_val: &T) {}

                #[allow(dead_code)]
                fn assert_impl_no_safe_aliasing<T: #crate_path::safety::NoSafeAliasing>() {}

                #(assert_impl_devicecopy(&#func_params);)*
                #(assert_impl_no_safe_aliasing::<#ptx_func_unboxed_types>();)*
            }

            #ptx_func_input_unwrap
        }
    }
}

fn specialise_ptx_func_inputs(
    crate_path: &syn::Path,
    FunctionInputs {
        func_inputs,
        func_input_cuda_types,
    }: &FunctionInputs,
    FuncIdent { func_ident, .. }: &FuncIdent,
    ImplGenerics { impl_generics, .. }: &ImplGenerics,
) -> (Vec<TokenStream>, Vec<TokenStream>) {
    func_inputs
        .iter()
        .zip(func_input_cuda_types.iter())
        .map(|(arg, (cuda_mode, _ptx_jit))| match arg {
            syn::FnArg::Typed(
                fn_arg @ syn::PatType {
                    attrs,
                    pat,
                    colon_token,
                    ty,
                },
            ) => {
                let arg_type = match &**ty {
                    syn::Type::Reference(syn::TypeReference { elem, .. }) => elem,
                    other => other,
                };
                let syn_type = quote::quote_spanned! { ty.span()=>
                    #crate_path::device::specialise_kernel_type!(#arg_type for #impl_generics in #func_ident)
                };

                let cuda_type = match cuda_mode {
                    InputCudaType::SafeDeviceCopy => quote::quote_spanned! { ty.span()=>
                        #crate_path::utils::device_copy::SafeDeviceCopyWrapper<#syn_type>
                    },
                    InputCudaType::LendRustToCuda => quote::quote_spanned! { ty.span()=>
                        #crate_path::common::DeviceAccessible<
                            <#syn_type as #crate_path::common::RustToCuda>::CudaRepresentation
                        >
                    },
                };

                let ty = if let syn::Type::Reference(syn::TypeReference {
                    lifetime,
                    mutability,
                    ..
                }) = &**ty
                {
                    let lifetime = quote_spanned! { lifetime.span()=>
                        'static
                    };

                    if mutability.is_some() {
                        quote::quote_spanned! { ty.span()=>
                            #crate_path::common::DeviceMutRef<#lifetime, #cuda_type>
                        }
                    } else {
                        quote::quote_spanned! { ty.span()=>
                            #crate_path::common::DeviceConstRef<#lifetime, #cuda_type>
                        }
                    }
                } else if matches!(cuda_mode, InputCudaType::LendRustToCuda) {
                    let lifetime = quote_spanned! { ty.span()=>
                        'static
                    };

                    quote::quote_spanned! { ty.span()=>
                        #crate_path::common::DeviceMutRef<#lifetime, #cuda_type>
                    }
                } else {
                    cuda_type
                };

                let fn_arg = quote::quote_spanned! { fn_arg.span()=>
                    #(#attrs)* #pat #colon_token #ty
                };

                (fn_arg, ty)
            },
            syn::FnArg::Receiver(_) => unreachable!(),
        })
        .unzip()
}

fn specialise_ptx_unboxed_types(
    crate_path: &syn::Path,
    FunctionInputs { func_inputs, .. }: &FunctionInputs,
    FuncIdent { func_ident, .. }: &FuncIdent,
    ImplGenerics { impl_generics, .. }: &ImplGenerics,
) -> Vec<TokenStream> {
    func_inputs
        .iter()
        .map(|arg| match arg {
            syn::FnArg::Typed(syn::PatType { ty, .. }) => {
                let arg_type = match &**ty {
                    syn::Type::Reference(syn::TypeReference { elem, .. }) => elem,
                    other => other,
                };

                quote::quote_spanned! { ty.span()=>
                    #crate_path::device::specialise_kernel_type!(#arg_type for #impl_generics in #func_ident)
                }
            },
            syn::FnArg::Receiver(_) => unreachable!(),
        })
        .collect()
}
