use proc_macro2::TokenStream;
use quote::quote;

use crate::kernel::wrapper::{DeclGenerics, FuncIdent, FunctionInputs, ImplGenerics, KernelConfig};

mod args_trait;
mod get_ptx;

use get_ptx::quote_get_ptx;

#[expect(clippy::too_many_arguments)] // FIXME
pub(in super::super) fn quote_host_link_macro(
    crate_path: &syn::Path,
    KernelConfig {
        visibility, link, ..
    }: &KernelConfig,
    decl_generics @ DeclGenerics {
        generic_start_token,
        generic_close_token,
        generic_kernel_params,
        ..
    }: &DeclGenerics,
    impl_generics: &ImplGenerics,
    func_inputs: &FunctionInputs,
    func_ident @ FuncIdent {
        func_ident: func_ident_name,
        func_ident_hash,
        ..
    }: &FuncIdent,
    func_params: &[syn::Ident],
    ptx_lint_levels: &TokenStream,
) -> TokenStream {
    let macro_generics = generic_kernel_params
        .iter()
        .enumerate()
        .map(|(i, generic)| {
            let generic_ident = quote::format_ident!("__g_{}", i);

            match generic {
                syn::GenericParam::Type(_) => quote!($#generic_ident:ty),
                syn::GenericParam::Const(_) => quote!($#generic_ident:expr),
                syn::GenericParam::Lifetime(_) => quote!($#generic_ident:lifetime),
            }
        })
        .collect::<Vec<_>>();

    let macro_generic_ids = (0..generic_kernel_params.len())
        .map(|i| quote::format_ident!("__g_{}", i))
        .collect::<Vec<_>>();

    let macro_only_lt_generic_ids = generic_kernel_params
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

    let macro_non_lt_generic_ids = generic_kernel_params
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

    let get_ptx = quote_get_ptx(
        crate_path,
        func_ident,
        decl_generics,
        impl_generics,
        func_inputs,
        func_params,
        &macro_non_lt_generic_ids,
        ptx_lint_levels,
    );

    quote! {
        #[cfg(not(target_os = "cuda"))]
        #visibility macro #link(
            impl #func_ident_name #generic_start_token
                #(#macro_generics),* $(,)?
            #generic_close_token for $ptx:ident
        ) {
            unsafe impl<#($#macro_only_lt_generic_ids),*> #crate_path::kernel::CompiledKernelPtx<
                #func_ident_name #generic_start_token #($#macro_generic_ids),* #generic_close_token
            > for $ptx #generic_start_token #($#macro_generic_ids),* #generic_close_token
            {
                #get_ptx

                fn get_entry_point() -> &'static ::core::ffi::CStr {
                    #crate_path::kernel::specialise_kernel_entry_point!(
                        #func_ident_hash #generic_start_token
                            #($#macro_non_lt_generic_ids),*
                        #generic_close_token
                    )
                }
            }
        }
    }
}
