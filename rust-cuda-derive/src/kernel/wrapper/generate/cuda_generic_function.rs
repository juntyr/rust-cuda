use proc_macro2::TokenStream;
use quote::quote;

use super::super::{DeclGenerics, FuncIdent};

pub(in super::super) fn quote_cuda_generic_function(
    DeclGenerics {
        generic_start_token,
        generic_kernel_params: generic_params,
        generic_close_token,
        generic_kernel_where_clause: generic_where_clause,
        ..
    }: &DeclGenerics,
    func_inputs: &syn::punctuated::Punctuated<syn::FnArg, syn::token::Comma>,
    FuncIdent { func_ident, .. }: &FuncIdent,
    func_attrs: &[syn::Attribute],
    func_block: &syn::Block,
) -> TokenStream {
    quote! {
        #[cfg(target_os = "cuda")]
        #(#func_attrs)*
        fn #func_ident #generic_start_token #generic_params #generic_close_token (#func_inputs)
            #generic_where_clause
        #func_block
    }
}
