use proc_macro2::TokenStream;
use syn::spanned::Spanned;

use super::super::super::super::{FunctionInputs, InputCudaType};

pub(in super::super) fn generate_launch_types(
    crate_path: &syn::Path,
    FunctionInputs {
        func_inputs,
        func_input_cuda_types,
    }: &FunctionInputs,
) -> (Vec<TokenStream>, Vec<syn::Type>) {
    let mut cpu_func_types_launch = Vec::with_capacity(func_inputs.len());
    let mut cpu_func_unboxed_types = Vec::with_capacity(func_inputs.len());

    func_inputs
        .iter()
        .zip(func_input_cuda_types.iter())
        .for_each(|(arg, (cuda_mode, _ptx_jit))| match arg {
            syn::FnArg::Typed(syn::PatType { ty, .. }) => {
                let syn_type = match &**ty {
                    syn::Type::Reference(syn::TypeReference { elem, .. }) => elem,
                    other => other,
                };

                cpu_func_unboxed_types.push(syn_type.clone());

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

                cpu_func_types_launch.push(
                    if let syn::Type::Reference(syn::TypeReference {
                        mutability,
                        lifetime,
                        ..
                    }) = &**ty
                    {
                        let comma: Option<syn::token::Comma> =
                            lifetime.as_ref().map(|_| syn::parse_quote!(,));

                        if mutability.is_some() {
                            quote::quote_spanned! { ty.span()=>
                                #crate_path::common::DeviceMutRef<#lifetime #comma #cuda_type>
                            }
                        } else {
                            quote::quote_spanned! { ty.span()=>
                                #crate_path::common::DeviceConstRef<#lifetime #comma #cuda_type>
                            }
                        }
                    } else if matches!(cuda_mode, InputCudaType::LendRustToCuda) {
                        quote::quote_spanned! { ty.span()=>
                            #crate_path::common::DeviceMutRef<#cuda_type>
                        }
                    } else {
                        quote! { #cuda_type }
                    },
                );
            },
            syn::FnArg::Receiver(_) => unreachable!(),
        });

    (cpu_func_types_launch, cpu_func_unboxed_types)
}
