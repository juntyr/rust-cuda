use proc_macro2::TokenStream;
use syn::spanned::Spanned;

use crate::kernel::utils::r2c_move_lifetime;

use super::super::super::super::{DeclGenerics, FunctionInputs, InputCudaType, KernelConfig};

pub(super) fn generate_launch_types(
    KernelConfig { args, .. }: &KernelConfig,
    DeclGenerics {
        generic_start_token,
        generic_close_token,
        ..
    }: &DeclGenerics,
    FunctionInputs {
        func_inputs,
        func_input_cuda_types,
    }: &FunctionInputs,
    macro_type_ids: &[syn::Ident],
) -> (Vec<TokenStream>, Vec<TokenStream>, Vec<TokenStream>) {
    let mut cpu_func_types_launch = Vec::with_capacity(func_inputs.len());
    let mut cpu_func_lifetime_erased_types = Vec::with_capacity(func_inputs.len());
    let mut cpu_func_unboxed_types = Vec::with_capacity(func_inputs.len());

    func_inputs
        .iter()
        .zip(func_input_cuda_types.iter())
        .enumerate()
        .for_each(|(i, (arg, (cuda_mode, _ptx_jit)))| match arg {
            syn::FnArg::Typed(syn::PatType { ty, .. }) => {
                let type_ident = quote::format_ident!("__T_{}", i);
                let syn_type = quote::quote_spanned! { ty.span()=>
                    <() as #args #generic_start_token
                        #($#macro_type_ids),*
                    #generic_close_token>::#type_ident
                };

                cpu_func_unboxed_types.push(syn_type.clone());

                let cuda_type = match cuda_mode {
                    InputCudaType::SafeDeviceCopy => quote::quote_spanned! { ty.span()=>
                        rust_cuda::utils::device_copy::SafeDeviceCopyWrapper<#syn_type>
                    },
                    InputCudaType::LendRustToCuda => quote::quote_spanned! { ty.span()=>
                        rust_cuda::common::DeviceAccessible<
                            <#syn_type as rust_cuda::common::RustToCuda>::CudaRepresentation
                        >
                    },
                };

                cpu_func_types_launch.push(
                    if let syn::Type::Reference(syn::TypeReference {
                        mutability,
                        lifetime,
                        ..
                    }) = &**ty
                    {
                        if mutability.is_some() {
                            quote::quote_spanned! { ty.span()=>
                                rust_cuda::common::DeviceMutRef<#lifetime, #cuda_type>
                            }
                        } else {
                            quote::quote_spanned! { ty.span()=>
                                rust_cuda::common::DeviceConstRef<#lifetime, #cuda_type>
                            }
                        }
                    } else if matches!(cuda_mode, InputCudaType::LendRustToCuda) {
                        let lifetime = r2c_move_lifetime(i, ty);

                        quote::quote_spanned! { ty.span()=>
                            rust_cuda::common::DeviceMutRef<#lifetime, #cuda_type>
                        }
                    } else {
                        quote! { #cuda_type }
                    },
                );

                cpu_func_lifetime_erased_types.push(
                    if let syn::Type::Reference(syn::TypeReference { mutability, .. }) = &**ty {
                        if mutability.is_some() {
                            quote::quote_spanned! { ty.span()=>
                                rust_cuda::common::DeviceMutRef<'static, #cuda_type>
                            }
                        } else {
                            quote::quote_spanned! { ty.span()=>
                                rust_cuda::common::DeviceConstRef<'static, #cuda_type>
                            }
                        }
                    } else if matches!(cuda_mode, InputCudaType::LendRustToCuda) {
                        quote::quote_spanned! { ty.span()=>
                            rust_cuda::common::DeviceMutRef<'static, #cuda_type>
                        }
                    } else {
                        cuda_type
                    },
                );
            },
            syn::FnArg::Receiver(_) => unreachable!(),
        });

    (
        cpu_func_types_launch,
        cpu_func_lifetime_erased_types,
        cpu_func_unboxed_types,
    )
}
