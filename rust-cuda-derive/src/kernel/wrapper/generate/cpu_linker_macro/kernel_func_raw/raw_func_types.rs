use proc_macro2::TokenStream;
use syn::spanned::Spanned;

use super::super::super::super::{DeclGenerics, FunctionInputs, InputCudaType, KernelConfig};

pub(super) fn generate_raw_func_types(
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
) -> Vec<TokenStream> {
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
            }) => {
                let type_ident = quote::format_ident!("__T_{}", i);
                let syn_type = quote! {
                    <() as #args #generic_start_token
                        #($#macro_type_ids),*
                    #generic_close_token>::#type_ident
                };

                let cuda_type = match cuda_mode {
                    InputCudaType::DeviceCopy => syn_type,
                    InputCudaType::RustToCuda => quote!(
                        rust_cuda::common::DeviceAccessible<
                            <#syn_type as rust_cuda::common::RustToCuda>::CudaRepresentation
                        >
                    ),
                };

                if let syn::Type::Reference(syn::TypeReference {
                    lifetime,
                    mutability,
                    ..
                }) = &**ty
                {
                    let wrapped_type = if mutability.is_some() {
                        if matches!(cuda_mode, InputCudaType::DeviceCopy) {
                            abort!(
                                mutability.span(),
                                "Cannot mutably alias a `DeviceCopy` kernel parameter."
                            );
                        }

                        quote!(
                            rust_cuda::host::HostAndDeviceMutRef<#lifetime, #cuda_type>
                        )
                    } else {
                        quote!(
                            rust_cuda::host::HostAndDeviceConstRef<#lifetime, #cuda_type>
                        )
                    };

                    quote! {
                        #(#attrs)* #mutability #pat #colon_token #wrapped_type
                    }
                } else if matches!(cuda_mode, InputCudaType::RustToCuda) {
                    abort!(
                        ty.span(),
                        "Kernel parameters transferred using `RustToCuda` must be references."
                    );
                } else {
                    quote! { #(#attrs)* #pat #colon_token #cuda_type }
                }
            },
            syn::FnArg::Receiver(_) => unreachable!(),
        })
        .collect()
}
