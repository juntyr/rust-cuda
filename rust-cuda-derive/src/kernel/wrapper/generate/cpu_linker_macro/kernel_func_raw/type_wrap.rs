use proc_macro2::TokenStream;
use quote::quote;

use crate::kernel::wrapper::InputCudaType;

use super::super::super::super::FunctionInputs;

pub(super) fn generate_func_input_and_ptx_jit_wraps(
    FunctionInputs {
        func_inputs,
        func_input_cuda_types,
    }: &FunctionInputs,
) -> (Vec<TokenStream>, Vec<TokenStream>) {
    func_inputs
        .iter()
        .zip(func_input_cuda_types.iter())
        .map(|(arg, (cuda_mode, ptx_jit))| match arg {
            syn::FnArg::Typed(syn::PatType { pat, ty, .. }) => {
                #[allow(clippy::if_same_then_else)]
                let func_input = if let syn::Type::Reference(_) = &**ty {
                    quote! { #pat.for_device() }
                } else if matches!(cuda_mode, InputCudaType::LendRustToCuda) {
                    quote! { #pat.for_device() }
                } else {
                    quote! { #pat }
                };

                let ptx_load = if ptx_jit.0 {
                    quote! { ConstLoad[#pat.for_host()] }
                } else {
                    quote! { Ignore[#pat] }
                };

                (func_input, ptx_load)
            },
            syn::FnArg::Receiver(_) => unreachable!(),
        })
        .unzip()
}
