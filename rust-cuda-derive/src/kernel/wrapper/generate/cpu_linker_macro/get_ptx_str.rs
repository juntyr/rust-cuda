use proc_macro2::TokenStream;

use super::super::super::{DeclGenerics, KernelConfig};

pub(super) fn quote_get_ptx_str(
    KernelConfig { args, .. }: &KernelConfig,
    DeclGenerics {
        generic_start_token,
        generic_close_token,
        ..
    }: &DeclGenerics,
    macro_type_ids: &[syn::Ident],
) -> TokenStream {
    let crate_name = match proc_macro::tracked_env::var("CARGO_CRATE_NAME") {
        Ok(crate_name) => crate_name.to_uppercase(),
        Err(err) => abort_call_site!("Failed to read crate name: {:?}.", err),
    };

    let crate_manifest_dir = proc_macro::tracked_env::var("CARGO_MANIFEST_DIR")
        .unwrap_or_else(|err| abort_call_site!("Failed to read crate path: {:?}.", err));

    quote! {
        fn get_ptx_str() -> &'static str {
            rust_cuda::host::link_kernel!(
                #args #crate_name #crate_manifest_dir #generic_start_token
                    #($#macro_type_ids),*
                #generic_close_token
            )
        }
    }
}
