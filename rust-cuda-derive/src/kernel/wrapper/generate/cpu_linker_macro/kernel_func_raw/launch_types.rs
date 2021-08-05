use proc_macro2::TokenStream;
use syn::spanned::Spanned;

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
) -> Vec<TokenStream> {
    func_inputs
        .iter()
        .zip(func_input_cuda_types.iter())
        .enumerate()
        .map(|(i, (arg, (cuda_mode, _ptx_jit)))| match arg {
            syn::FnArg::Typed(syn::PatType { ty, .. }) => {
                let type_ident = quote::format_ident!("__T_{}", i);
                let syn_type = quote::quote_spanned! { ty.span()=>
                    <() as #args #generic_start_token
                        #($#macro_type_ids),*
                    #generic_close_token>::#type_ident
                };

                let cuda_type = match cuda_mode {
                    InputCudaType::DeviceCopy => syn_type,
                    InputCudaType::LendRustBorrowToCuda => quote::quote_spanned! { ty.span()=>
                        <#syn_type as rust_cuda::common::RustToCuda>::CudaRepresentation
                    },
                };

                if let syn::Type::Reference(syn::TypeReference { mutability, .. }) = &**ty {
                    if mutability.is_some() {
                        quote::quote_spanned! { ty.span()=>
                            rust_cuda::common::DeviceBoxMut<#cuda_type>
                        }
                    } else {
                        quote::quote_spanned! { ty.span()=>
                            rust_cuda::common::DeviceBoxConst<#cuda_type>
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
                }
            },
            syn::FnArg::Receiver(_) => unreachable!(),
        })
        .collect()
}
