use proc_macro2::TokenStream;

use super::super::{DeclGenerics, FuncIdent, FunctionInputs, KernelConfig};

mod get_ptx;

use get_ptx::quote_get_ptx;

pub(in super::super) fn quote_cpu_linker_macro(
    crate_path: &syn::Path,
    config @ KernelConfig {
        visibility,
        linker,
        kernel,
        launcher,
        ..
    }: &KernelConfig,
    decl_generics @ DeclGenerics {
        generic_start_token,
        generic_trait_params: generic_params,
        generic_close_token,
        generic_kernel_params,
        ..
    }: &DeclGenerics,
    func_inputs: &FunctionInputs,
    func_ident @ FuncIdent {
        func_ident: func_ident_name,
        func_ident_hash, ..
    }: &FuncIdent,
    func_params: &[syn::Ident],
    ptx_lint_levels: &TokenStream,
) -> TokenStream {
    let macro_generics = generic_kernel_params//generic_params
        .iter()
        .enumerate()
        .map(|(i, generic)| {
            let generic_ident = quote::format_ident!("__g_{}", i);

            match generic {
                syn::GenericParam::Type(_) => quote!($#generic_ident:ty),
                syn::GenericParam::Const(_) => quote!($#generic_ident:expr),
                syn::GenericParam::Lifetime(_) => quote!($#generic_ident:lifetime),//unreachable!(),
            }
        })
        .collect::<Vec<_>>();

    let macro_generic_ids = (0..generic_kernel_params.len())
        .map(|i| quote::format_ident!("__g_{}", i))
        .collect::<Vec<_>>();

    let macro_only_lt_generic_ids = generic_kernel_params//generic_params
        .iter()
        .enumerate()
        .filter_map(|(i, generic)| {
            let generic_ident = quote::format_ident!("__g_{}", i);

            match generic {
                syn::GenericParam::Type(_) | syn::GenericParam::Const(_) => None,
                syn::GenericParam::Lifetime(_) => Some(generic_ident),
            }
        })
        .collect::<Vec<_>>();

    let macro_non_lt_generic_ids = generic_kernel_params//generic_params
        .iter()
        .enumerate()
        .filter_map(|(i, generic)| {
            let generic_ident = quote::format_ident!("__g_{}", i);

            match generic {
                syn::GenericParam::Type(_) | syn::GenericParam::Const(_) => Some(generic_ident),
                syn::GenericParam::Lifetime(_) => None,
            }
        })
        .collect::<Vec<_>>();

    let cpu_linker_macro_visibility = if visibility.is_some() {
        quote! { #[macro_export] }
    } else {
        quote! {}
    };

    let get_ptx = quote_get_ptx(
        crate_path,
        func_ident,
        config,
        decl_generics,
        func_inputs,
        func_params,
        &macro_non_lt_generic_ids,
        ptx_lint_levels,
    );

    quote! {
        #[cfg(not(target_os = "cuda"))]
        #cpu_linker_macro_visibility
        macro_rules! #linker {
            (impl #func_ident_name #generic_start_token #(#macro_generics),* $(,)? #generic_close_token for $ptx:ident) => {
                unsafe impl<#($#macro_only_lt_generic_ids),*> #crate_path::host::CompiledKernelPtx<
                    #func_ident_name #generic_start_token #($#macro_generic_ids),* #generic_close_token
                    //dyn #kernel #generic_start_token #($#macro_type_ids),* #generic_close_token
                > for $ptx #generic_start_token #($#macro_generic_ids),* #generic_close_token // #launcher #generic_start_token #($#macro_type_ids),* #generic_close_token
                {
                    #get_ptx

                    fn get_entry_point() -> &'static ::core::ffi::CStr {
                        #crate_path::host::specialise_kernel_call!(
                            #func_ident_hash #generic_start_token
                                #($#macro_non_lt_generic_ids),*
                            #generic_close_token
                        )
                    }
                }
            };
        }
    }
}
