use proc_macro2::TokenStream;

use super::super::{DeclGenerics, FuncIdent, FunctionInputs, KernelConfig};

mod get_ptx_str;
mod new_kernel;

use get_ptx_str::quote_get_ptx_str;
use new_kernel::quote_new_kernel;

pub(in super::super) fn quote_cpu_linker_macro(
    crate_path: &syn::Path,
    config @ KernelConfig {
        visibility,
        linker,
        launcher,
        ptx,
        ..
    }: &KernelConfig,
    decl_generics @ DeclGenerics {
        generic_start_token,
        generic_trait_params: generic_params,
        generic_close_token,
        ..
    }: &DeclGenerics,
    func_inputs: &FunctionInputs,
    func_ident: &FuncIdent,
    func_params: &[syn::Ident],
    ptx_lint_levels: &TokenStream,
) -> TokenStream {
    let macro_types = generic_params
        .iter()
        .enumerate()
        .map(|(i, generic)| {
            let generic_ident = quote::format_ident!("__g_{}", i);

            match generic {
                syn::GenericParam::Type(_) => quote!($#generic_ident:ty),
                syn::GenericParam::Const(_) => quote!($#generic_ident:expr),
                syn::GenericParam::Lifetime(_) => unreachable!(),
            }
        })
        .collect::<Vec<_>>();

    let macro_type_ids = (0..generic_params.len())
        .map(|i| quote::format_ident!("__g_{}", i))
        .collect::<Vec<_>>();

    let cpu_linker_macro_visibility = if visibility.is_some() {
        quote! { #[macro_export] }
    } else {
        quote! {}
    };

    let get_ptx_str = quote_get_ptx_str(
        crate_path,
        func_ident,
        config,
        decl_generics,
        func_inputs,
        func_params,
        &macro_type_ids,
        ptx_lint_levels,
    );
    let new_kernel = quote_new_kernel(
        crate_path,
        config,
        decl_generics,
        func_ident,
        &macro_type_ids,
    );

    quote! {
        #[cfg(not(target_os = "cuda"))]
        #cpu_linker_macro_visibility
        macro_rules! #linker {
            (#(#macro_types),* $(,)?) => {
                unsafe impl #ptx #generic_start_token #($#macro_type_ids),* #generic_close_token
                    for #launcher #generic_start_token #($#macro_type_ids),* #generic_close_token
                {
                    #get_ptx_str

                    #new_kernel
                }
            };
        }
    }
}
