use proc_macro2::TokenStream;

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
        .map(|(arg, (_cuda_mode, ptx_jit))| match arg {
            syn::FnArg::Typed(syn::PatType {
                attrs: _attrs,
                pat,
                colon_token: _colon_token,
                ty,
            }) => {
                let func_input = if let syn::Type::Reference(_) = &**ty {
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
