use proc_macro2::TokenStream;
use syn::spanned::Spanned;

use crate::kernel::utils::skip_kernel_compilation;

use super::super::super::{DeclGenerics, FuncIdent, FunctionInputs, KernelConfig};

pub(super) fn quote_get_ptx_str(
    FuncIdent { func_ident, .. }: &FuncIdent,
    config @ KernelConfig { args, .. }: &KernelConfig,
    generics
    @ DeclGenerics {
        generic_start_token,
        generic_close_token,
        ..
    }: &DeclGenerics,
    inputs: &FunctionInputs,
    func_params: &[syn::Ident],
    macro_type_ids: &[syn::Ident],
) -> TokenStream {
    let crate_name = match proc_macro::tracked_env::var("CARGO_CRATE_NAME") {
        Ok(crate_name) => crate_name.to_uppercase(),
        Err(err) => abort_call_site!("Failed to read crate name: {:?}.", err),
    };

    let crate_manifest_dir = proc_macro::tracked_env::var("CARGO_MANIFEST_DIR")
        .unwrap_or_else(|err| abort_call_site!("Failed to read crate path: {:?}.", err));

    let cpu_func_lifetime_erased_types =
        super::kernel_func_raw::generate_launch_types(config, generics, inputs, macro_type_ids).1;

    let type_layout_asserts = if skip_kernel_compilation() {
        Vec::new()
    } else {
        cpu_func_lifetime_erased_types
            .iter()
            .zip(func_params.iter())
            .map(|(ty, param)| {
                let layout_param =
                    syn::Ident::new(&format!("__{}_layout", param).to_uppercase(), param.span());

                quote::quote_spanned! { ty.span()=>
                    const _: ::rust_cuda::memory::type_layout::Assert<{
                        ::rust_cuda::memory::type_layout::CpuAndGpuTypeLayouts::Match
                    }> = ::rust_cuda::memory::type_layout::Assert::<{
                        ::rust_cuda::memory::type_layout::check::<#ty>(#layout_param)
                    }>;
                }
            })
            .collect::<Vec<_>>()
    };

    quote! {
        fn get_ptx_str() -> &'static str {
            rust_cuda::host::link_kernel!{
                #func_ident #args #crate_name #crate_manifest_dir #generic_start_token
                    #($#macro_type_ids),*
                #generic_close_token
            }

            #(#type_layout_asserts)*

            PTX_STR
        }
    }
}
