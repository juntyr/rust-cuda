use proc_macro2::TokenStream;
use quote::quote;

use super::super::super::{DeclGenerics, FuncIdent, KernelConfig};

pub(super) fn quote_new_kernel(
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
        fn new_kernel() -> rust_cuda::rustacuda::error::CudaResult<
            rust_cuda::host::TypedKernel<dyn #kernel #generic_start_token
                #($#macro_type_ids),*
            #generic_close_token>
        > {
            let ptx = Self::get_ptx_str();
            let entry_point = rust_cuda::host::specialise_kernel_call!(
                #func_ident_hash #generic_start_token
                    #($#macro_type_ids),*
                #generic_close_token
            );

            rust_cuda::host::TypedKernel::new(ptx, entry_point)
        }
    }
}
