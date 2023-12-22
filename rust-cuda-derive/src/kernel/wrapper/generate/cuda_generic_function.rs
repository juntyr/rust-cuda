use proc_macro2::TokenStream;
use syn::spanned::Spanned;

use super::super::{DeclGenerics, FuncIdent};

pub(in super::super) fn quote_cuda_generic_function(
    crate_path: &syn::Path,
    DeclGenerics {
        generic_start_token,
        generic_kernel_params: generic_params,
        generic_close_token,
        ..
    }: &DeclGenerics,
    func_inputs: &syn::punctuated::Punctuated<syn::FnArg, syn::token::Comma>,
    FuncIdent { func_ident, .. }: &FuncIdent,
    func_attrs: &[syn::Attribute],
    func_block: &syn::Block,
) -> TokenStream {
    let kernel_func_inputs = func_inputs
        .iter()
        .map(|arg| match arg {
            syn::FnArg::Typed(syn::PatType {
                attrs,
                ty,
                pat,
                colon_token,
            }) => {
                let ty: syn::Type = syn::parse_quote_spanned! { ty.span()=>
                    <#ty as #crate_path::common::CudaKernelParameter>::DeviceType<'_>
                };

                syn::FnArg::Typed(syn::PatType {
                    attrs: attrs.clone(),
                    ty: Box::new(ty),
                    pat: pat.clone(),
                    colon_token: *colon_token,
                })
            },
            syn::FnArg::Receiver(_) => unreachable!(),
        })
        .collect::<Vec<_>>();

    quote! {
        #[cfg(target_os = "cuda")]
        #(#func_attrs)*
        fn #func_ident #generic_start_token #generic_params #generic_close_token (
            #(#kernel_func_inputs),*
        )
        #func_block
    }
}
