use proc_macro2::TokenStream;
use syn::spanned::Spanned;

use crate::kernel::utils::r2c_move_lifetime;

use super::super::super::super::{FunctionInputs, ImplGenerics, InputCudaType, KernelConfig};

pub(in super::super) fn generate_launch_types(
    crate_path: &syn::Path,
    KernelConfig { args, .. }: &KernelConfig,
    ImplGenerics { ty_generics, .. }: &ImplGenerics,
    FunctionInputs {
        func_inputs,
        func_input_cuda_types,
    }: &FunctionInputs,
) -> (Vec<TokenStream>, Vec<TokenStream>) {
    let mut cpu_func_types_launch = Vec::with_capacity(func_inputs.len());
    let mut cpu_func_unboxed_types = Vec::with_capacity(func_inputs.len());

    func_inputs
        .iter()
        .zip(func_input_cuda_types.iter())
        .enumerate()
        .for_each(|(i, (arg, (cuda_mode, _ptx_jit)))| match arg {
            syn::FnArg::Typed(syn::PatType { ty, .. }) => {
                let type_ident = quote::format_ident!("__T_{}", i);
                let syn_type = quote::quote_spanned! { ty.span()=>
                    <() as #args #ty_generics>::#type_ident
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
                        let lifetime = r2c_move_lifetime(i, ty);

                        quote::quote_spanned! { ty.span()=>
                            #crate_path::common::DeviceMutRef<#lifetime, #cuda_type>
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
