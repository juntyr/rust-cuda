use proc_macro2::TokenStream;

use super::super::{DeclGenerics, FuncIdent, FunctionInputs};

pub(in super::super) fn quote_cuda_generic_function(
    DeclGenerics {
        generic_start_token,
        generic_params,
        generic_close_token,
        generic_where_clause,
    }: &DeclGenerics,
    FunctionInputs { func_inputs, .. }: &FunctionInputs,
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
