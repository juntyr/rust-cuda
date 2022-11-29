use proc_macro2::TokenStream;

use super::super::super::{DeclGenerics, FuncIdent, KernelConfig};

pub(super) fn quote_new_kernel(
    crate_path: &syn::Path,
    KernelConfig { kernel, .. }: &KernelConfig,
    DeclGenerics {
        generic_start_token,
        generic_close_token,
        ..
    }: &DeclGenerics,
    FuncIdent {
        func_ident_hash, ..
    }: &FuncIdent,
    macro_type_ids: &[syn::Ident],
) -> TokenStream {
    quote! {
        fn new_kernel() -> #crate_path::rustacuda::error::CudaResult<
            #crate_path::host::TypedKernel<dyn #kernel #generic_start_token
                #($#macro_type_ids),*
            #generic_close_token>
        > {
            let ptx = Self::get_ptx_str();
            let entry_point = #crate_path::host::specialise_kernel_call!(
                #func_ident_hash #generic_start_token
                    #($#macro_type_ids),*
                #generic_close_token
            );

            #crate_path::host::TypedKernel::new(ptx, entry_point)
        }
    }
}
