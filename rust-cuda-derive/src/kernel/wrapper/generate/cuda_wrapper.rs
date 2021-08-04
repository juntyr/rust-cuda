use proc_macro2::TokenStream;
use syn::spanned::Spanned;

use super::super::{FuncIdent, FunctionInputs, InputCudaType, KernelConfig};

pub(in super::super) fn generate_cuda_wrapper(
    config @ KernelConfig { args, .. }: &KernelConfig,
    inputs @ FunctionInputs {
        func_inputs,
        func_input_cuda_types,
    }: &FunctionInputs,
    FuncIdent { func_ident, .. }: &FuncIdent,
    func_attrs: &[syn::Attribute],
    func_params: &[syn::Pat],
    func_type_errors: &[syn::Ident],
) -> TokenStream {
    let ptx_func_inputs = specialise_ptx_func_inputs(config, inputs);

    let ptx_func_types: Vec<&syn::Type> = ptx_func_inputs
        .iter()
        .map(|arg| match arg {
            syn::FnArg::Typed(syn::PatType { ty, .. }) => &**ty,
            syn::FnArg::Receiver(_) => unreachable!(),
        })
        .collect();

    let ptx_func_input_unwrap = func_inputs
        .iter().zip(func_input_cuda_types.iter()).enumerate()
        .rev()
        .fold(quote! {
            #func_ident(#(#func_params),*)
        }, |inner, (i, (arg, (cuda_mode, ptx_jit)))| match arg {
            syn::FnArg::Typed(syn::PatType {
                attrs: _attrs,
                pat,
                colon_token: _colon_token,
                ty,
            }) => {
                // Emit PTX JIT load markers
                let ptx_jit_load = if ptx_jit.0 {
                    quote! {
                        rust_cuda::ptx_jit::PtxJITConstLoad!([#i] => #pat.as_ref())
                    }
                } else { quote! {} };

                match cuda_mode {
                    InputCudaType::DeviceCopy => if let syn::Type::Reference(
                        syn::TypeReference { mutability, .. }
                    ) = &**ty {
                        if mutability.is_some() {
                            quote! { #ptx_jit_load; let #pat = #pat.as_mut(); #inner }
                        } else {
                            quote! { #ptx_jit_load; let #pat = #pat.as_ref(); #inner }
                        }
                    } else {
                        inner
                    },
                    InputCudaType::LendRustBorrowToCuda => if let syn::Type::Reference(
                        syn::TypeReference { mutability, .. }
                    ) = &**ty {
                        if mutability.is_some() {
                            quote! {
                                #ptx_jit_load;
                                rust_cuda::device::BorrowFromRust::with_borrow_from_rust_mut(
                                    #pat, |#pat| { #inner },
                                )
                            }
                        } else {
                            quote! {
                                #ptx_jit_load;
                                rust_cuda::device::BorrowFromRust::with_borrow_from_rust(
                                    #pat, |#pat| { #inner },
                                )
                            }
                        }
                    } else { unreachable!() }
                }
            },
            syn::FnArg::Receiver(_) => unreachable!(),
        });

    quote! {
        #[cfg(target_os = "cuda")]
        #[rust_cuda::device::specialise_kernel_entry(#args)]
        #[no_mangle]
        #(#func_attrs)*
        pub unsafe extern "ptx-kernel" fn #func_ident(#ptx_func_inputs) {
            #(
                #[allow(non_camel_case_types, dead_code)]
                struct #func_type_errors;
                const _: [#func_type_errors; 1 - {
                    const ASSERT: bool = (::core::mem::size_of::<#ptx_func_types>() <= 8); ASSERT
                } as usize] = [];
            )*

            if false {
                fn assert_impl_devicecopy<T: rust_cuda::rustacuda_core::DeviceCopy>(_val: &T) {}

                #(assert_impl_devicecopy(&#func_params);)*
            }

            #ptx_func_input_unwrap
        }
    }
}

fn specialise_ptx_func_inputs(
    KernelConfig { args, .. }: &KernelConfig,
    FunctionInputs {
        func_inputs,
        func_input_cuda_types,
    }: &FunctionInputs,
) -> syn::punctuated::Punctuated<syn::FnArg, syn::token::Comma> {
    func_inputs
        .iter()
        .zip(func_input_cuda_types.iter())
        .enumerate()
        .map(|(i, (arg, (cuda_mode, _ptx_jit)))| match arg {
            syn::FnArg::Typed(syn::PatType {
                attrs,
                pat,
                colon_token,
                ty,
            }) => syn::FnArg::Typed(syn::PatType {
                attrs: attrs.clone(),
                pat: {
                    if let syn::Type::Reference(syn::TypeReference { mutability, .. }) = &**ty {
                        if matches!(cuda_mode, InputCudaType::DeviceCopy) && mutability.is_some() {
                            syn::parse_quote!(mut #pat)
                        } else {
                            pat.clone()
                        }
                    } else {
                        pat.clone()
                    }
                },
                colon_token: *colon_token,
                ty: {
                    let type_ident = quote::format_ident!("__T_{}", i);
                    let syn_type = syn::parse_quote!(
                        rust_cuda::device::specialise_kernel_type!(#args :: #type_ident)
                    );

                    let cuda_type = match cuda_mode {
                        InputCudaType::DeviceCopy => syn_type,
                        InputCudaType::LendRustBorrowToCuda => syn::parse_quote!(
                            <#syn_type as rust_cuda::common::RustToCuda>::CudaRepresentation
                        ),
                    };

                    if let syn::Type::Reference(syn::TypeReference {
                        and_token: _and_token,
                        lifetime,
                        mutability,
                        elem: _elem,
                    }) = &**ty
                    {
                        if lifetime.is_some() {
                            abort!(lifetime.span(), "Kernel parameters cannot have lifetimes.");
                        }

                        if mutability.is_some() {
                            syn::parse_quote!(
                                rust_cuda::common::DeviceBoxMut<#cuda_type>
                            )
                        } else {
                            syn::parse_quote!(
                                rust_cuda::common::DeviceBoxConst<#cuda_type>
                            )
                        }
                    } else if matches!(cuda_mode, InputCudaType::LendRustBorrowToCuda) {
                        abort!(
                            ty.span(),
                            "Kernel parameters transferred using `LendRustBorrowToCuda` must be \
                             references."
                        );
                    } else {
                        cuda_type
                    }
                },
            }),
            syn::FnArg::Receiver(_) => unreachable!(),
        })
        .collect()
}
