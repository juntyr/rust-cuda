use proc_macro2::TokenStream;

use crate::kernel::wrapper::InputCudaType;

use super::super::super::super::FunctionInputs;

pub(super) fn generate_func_input_and_ptx_jit_wraps(
    crate_path: &syn::Path,
    FunctionInputs {
        func_inputs,
        func_input_cuda_types,
    }: &FunctionInputs,
) -> (Vec<TokenStream>, TokenStream) {
    let mut any_ptx_jit = false;

    let (func_input_wrap, func_cpu_ptx_jit_wrap): (Vec<TokenStream>, Vec<TokenStream>) =
        func_inputs
            .iter()
            .zip(func_input_cuda_types.iter())
            .map(|(arg, (cuda_mode, ptx_jit))| match arg {
                syn::FnArg::Typed(syn::PatType { pat, ty, .. }) => {
                    #[allow(clippy::if_same_then_else)]
                    let func_input = if let syn::Type::Reference(_) = &**ty {
                        quote! { unsafe { #pat.for_device_async() } }
                    } else if matches!(cuda_mode, InputCudaType::LendRustToCuda) {
                        quote! { unsafe { #pat.for_device_async() } }
                    } else {
                        quote! { #pat }
                    };

                    let ptx_load = if ptx_jit.0 {
                        any_ptx_jit = true;

                        quote! { Some(#crate_path::ptx_jit::arg_as_raw_bytes(#pat.for_host())) }
                    } else {
                        quote! { None }
                    };

                    (func_input, ptx_load)
                },
                syn::FnArg::Receiver(_) => unreachable!(),
            })
            .unzip();

    if any_ptx_jit {
        (
            func_input_wrap,
            quote!(Some(&[#(#func_cpu_ptx_jit_wrap),*])),
        )
    } else {
        (func_input_wrap, quote!(None))
    }
}
