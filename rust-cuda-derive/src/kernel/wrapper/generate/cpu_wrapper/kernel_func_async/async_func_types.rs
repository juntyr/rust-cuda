use proc_macro2::TokenStream;
use syn::spanned::Spanned;

use super::super::super::super::{FunctionInputs, InputCudaType};

pub(super) fn generate_async_func_types(
    crate_path: &syn::Path,
    FunctionInputs {
        func_inputs,
        func_input_cuda_types,
    }: &FunctionInputs,
    stream: &syn::Lifetime,
) -> Vec<TokenStream> {
    func_inputs
        .iter()
        .zip(func_input_cuda_types.iter())
        .map(|(arg, (cuda_mode, _ptx_jit))| match arg {
            syn::FnArg::Typed(syn::PatType {
                attrs,
                pat,
                colon_token,
                ty,
            }) => {
                let syn_type = match &**ty {
                    syn::Type::Reference(syn::TypeReference { elem, .. }) => elem,
                    other => other,
                };

                let cuda_type = match cuda_mode {
                    InputCudaType::SafeDeviceCopy => quote! {
                        #crate_path::utils::device_copy::SafeDeviceCopyWrapper<#syn_type>
                    },
                    InputCudaType::LendRustToCuda => quote! {
                        #crate_path::common::DeviceAccessible<
                            <#syn_type as #crate_path::common::RustToCuda>::CudaRepresentation
                        >
                    },
                };

                if let syn::Type::Reference(syn::TypeReference {
                    lifetime,
                    mutability,
                    ..
                }) = &**ty
                {
                    let lifetime = lifetime.clone().unwrap_or(syn::parse_quote!('_));

                    let wrapped_type = if mutability.is_some() {
                        if matches!(cuda_mode, InputCudaType::SafeDeviceCopy) {
                            abort!(
                                mutability.span(),
                                "Cannot mutably alias a `SafeDeviceCopy` kernel parameter."
                            );
                        }

                        quote!(
                            #crate_path::host::HostAndDeviceMutRefAsync<#stream, #lifetime, #cuda_type>
                        )
                    } else {
                        quote!(
                            #crate_path::host::HostAndDeviceConstRefAsync<#stream, #lifetime, #cuda_type>
                        )
                    };

                    quote! {
                        #(#attrs)* #mutability #pat #colon_token #wrapped_type
                    }
                } else if matches!(cuda_mode, InputCudaType::LendRustToCuda) {
                    let wrapped_type = quote! {
                        #crate_path::host::HostAndDeviceOwnedAsync<#stream, '_, #cuda_type>
                    };

                    quote! {
                        #(#attrs)* #pat #colon_token #wrapped_type
                    }
                } else {
                    quote! { #(#attrs)* #pat #colon_token #cuda_type }
                }
            },
            syn::FnArg::Receiver(_) => unreachable!(),
        })
        .collect()
}
