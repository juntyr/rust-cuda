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
                    InputCudaType::LendRustBorrowToCuda => quote!(
                        rust_cuda::common::DeviceAccessible<
                            <#syn_type as rust_cuda::common::RustToCuda>::CudaRepresentation
                        >
                    ),
                };

                if let syn::Type::Reference(syn::TypeReference {
                    and_token,
                    lifetime,
                    mutability,
                    ..
                }) = &**ty
                {
                    if lifetime.is_some() {
                        abort!(lifetime.span(), "Kernel parameters cannot have lifetimes.");
                    }

                    let wrapped_type = if mutability.is_some() {
                        quote!(
                            rust_cuda::host::HostDevicePointerMut<#cuda_type>
                        )
                    } else {
                        quote!(
                            rust_cuda::host::HostDevicePointerConst<#cuda_type>
                        )
                    };

                    quote! {
                        #(#attrs)* #pat #colon_token #and_token #lifetime #mutability #wrapped_type
                    }
                } else if matches!(cuda_mode, InputCudaType::LendRustBorrowToCuda) {
                    abort!(
                        ty.span(),
                        "Kernel parameters transferred using `LendRustBorrowToCuda` must be \
                         references."
                    );
                } else {
                    quote! { #(#attrs)* #pat #colon_token #cuda_type }
                }
            },
            syn::FnArg::Receiver(_) => unreachable!(),
        })
        .collect()
}
