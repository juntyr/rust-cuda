use proc_macro2::TokenStream;
use syn::spanned::Spanned;

use super::super::{FuncIdent, FunctionInputs, InputCudaType, KernelConfig};

pub(in super::super) fn quote_cuda_wrapper(
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
    let arch_checks = super::arch_checks::quote_arch_checks();

    let (ptx_func_inputs, ptx_func_types) = specialise_ptx_func_inputs(config, inputs);
    let ptx_func_unboxed_types = specialise_ptx_unboxed_types(config, inputs);

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
                        rust_cuda::ptx_jit::PtxJITConstLoad!([#i] => #pat.as_ref())
                    }
                } else { quote! {} };

                match cuda_mode {
                    InputCudaType::DeviceCopy => if let syn::Type::Reference(
                        syn::TypeReference { mutability, .. }
                    ) = &**ty {
                        if mutability.is_some() {
                            quote! { #ptx_jit_load; { let #pat = #pat.as_mut(); #inner } }
                        } else {
                            quote! { #ptx_jit_load; { let #pat = #pat.as_ref(); #inner } }
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
        pub unsafe extern "ptx-kernel" fn #func_ident(#(#ptx_func_inputs),*) {
            #arch_checks

            #(
                #[allow(dead_code, non_camel_case_types)]
                enum #func_type_errors {}
                const _: [#func_type_errors; 1 - {
                    const ASSERT: bool = (
                        ::core::mem::size_of::<#ptx_func_types>() <= 8
                    ); ASSERT
                } as usize] = [];
            )*

            #[deny(improper_ctypes)]
            mod __rust_cuda_ffi_safe_assert {
                use super::#args;

                extern "C" { #(
                    #[allow(dead_code)]
                    static #func_params: #ptx_func_types;
                )* }
            }

            if false {
                #[allow(dead_code)]
                fn assert_impl_devicecopy<T: rust_cuda::rustacuda_core::DeviceCopy>(_val: &T) {}

                #[allow(dead_code)]
                fn assert_impl_no_aliasing<
                    T: rust_cuda::utils::aliasing::NoAliasing
                >() {}

                #(assert_impl_devicecopy(&#func_params);)*
                #(assert_impl_no_aliasing::<#ptx_func_unboxed_types>();)*
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
) -> (Vec<TokenStream>, Vec<TokenStream>) {
    func_inputs
        .iter()
        .zip(func_input_cuda_types.iter())
        .enumerate()
        .map(|(i, (arg, (cuda_mode, _ptx_jit)))| match arg {
            syn::FnArg::Typed(
                fn_arg
                @
                syn::PatType {
                    attrs,
                    pat,
                    colon_token,
                    ty,
                },
            ) => {
                let pat = if let syn::Type::Reference(syn::TypeReference { mutability, .. }) = &**ty
                {
                    if matches!(cuda_mode, InputCudaType::DeviceCopy) && mutability.is_some() {
                        quote::quote_spanned! { pat.span()=> mut #pat }
                    } else {
                        quote::quote_spanned! { pat.span()=> #pat }
                    }
                } else {
                    quote::quote_spanned! { pat.span()=> #pat }
                };

                let type_ident = quote::format_ident!("__T_{}", i);
                let syn_type = quote::quote_spanned! { ty.span()=>
                    rust_cuda::device::specialise_kernel_type!(#args :: #type_ident)
                };

                let cuda_type = match cuda_mode {
                    InputCudaType::DeviceCopy => syn_type,
                    InputCudaType::LendRustBorrowToCuda => quote::quote_spanned! { ty.span()=>
                        rust_cuda::common::DeviceAccessible<
                            <#syn_type as rust_cuda::common::RustToCuda>::CudaRepresentation
                        >
                    },
                };

                let ty = if let syn::Type::Reference(syn::TypeReference {
                    lifetime,
                    mutability,
                    ..
                }) = &**ty
                {
                    if lifetime.is_some() {
                        abort!(lifetime.span(), "Kernel parameters cannot have lifetimes.");
                    }

                    if mutability.is_some() {
                        quote::quote_spanned! { ty.span()=>
                            rust_cuda::common::DevicePointerMut<#cuda_type>
                        }
                    } else {
                        quote::quote_spanned! { ty.span()=>
                            rust_cuda::common::DevicePointerConst<#cuda_type>
                        }
                    }
                } else if matches!(cuda_mode, InputCudaType::LendRustBorrowToCuda) {
                    abort!(
                        ty.span(),
                        "Kernel parameters transferred using `LendRustBorrowToCuda` must be \
                         references."
                    );
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
    KernelConfig { args, .. }: &KernelConfig,
    FunctionInputs {
        func_inputs,
        func_input_cuda_types,
    }: &FunctionInputs,
) -> Vec<TokenStream> {
    func_inputs
        .iter()
        .zip(func_input_cuda_types.iter())
        .enumerate()
        .map(|(i, (arg, (cuda_mode, _ptx_jit)))| match arg {
            syn::FnArg::Typed(syn::PatType { ty, .. }) => {
                let type_ident = quote::format_ident!("__T_{}", i);
                let syn_type = quote::quote_spanned! { ty.span()=>
                    rust_cuda::device::specialise_kernel_type!(#args :: #type_ident)
                };

                match cuda_mode {
                    InputCudaType::DeviceCopy => syn_type,
                    InputCudaType::LendRustBorrowToCuda => quote::quote_spanned! { ty.span()=>
                        <#syn_type as rust_cuda::common::RustToCuda>::CudaRepresentation
                    },
                }
            },
            syn::FnArg::Receiver(_) => unreachable!(),
        })
        .collect()
}
