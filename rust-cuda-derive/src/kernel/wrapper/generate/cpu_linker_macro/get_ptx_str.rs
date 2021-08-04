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
    let crate_name = match std::env::var("CARGO_CRATE_NAME") {
        Ok(crate_name) => crate_name.to_uppercase(),
        Err(err) => abort_call_site!("Failed to read crate name: {:?}.", err),
    };

    let crate_manifest_dir = match std::env::var_os("CARGO_MANIFEST_DIR") {
        Some(crate_manifest_dir) => {
            let crate_manifest_dir = format!("{:?}", crate_manifest_dir);

            crate_manifest_dir
                .strip_prefix('"')
                .unwrap()
                .strip_suffix('"')
                .unwrap()
                .to_owned()
        },
        None => abort_call_site!("Failed to read crate path: NotPresent."),
    };

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
