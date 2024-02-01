use proc_macro2::TokenStream;
use syn::spanned::Spanned;
use quote::quote;

use crate::kernel::{
    utils::skip_kernel_compilation,
    wrapper::{DeclGenerics, FuncIdent, FunctionInputs, ImplGenerics},
    KERNEL_TYPE_LAYOUT_IDENT, PTX_CSTR_IDENT,
};

#[allow(clippy::too_many_arguments)]
pub(super) fn quote_get_ptx(
    crate_path: &syn::Path,
    FuncIdent {
        func_ident,
        func_ident_hash,
        ..
    }: &FuncIdent,
    generics @ DeclGenerics {
        generic_start_token,
        generic_close_token,
        ..
    }: &DeclGenerics,
    impl_generics: &ImplGenerics,
    inputs: &FunctionInputs,
    func_params: &[syn::Ident],
    macro_type_ids: &[syn::Ident],
    ptx_lint_levels: &TokenStream,
) -> TokenStream {
    let crate_name = match proc_macro::tracked_env::var("CARGO_CRATE_NAME") {
        Ok(crate_name) => crate_name.to_uppercase(),
        Err(err) => abort_call_site!("Failed to read crate name: {:?}.", err),
    };

    let crate_manifest_dir = proc_macro::tracked_env::var("CARGO_MANIFEST_DIR")
        .unwrap_or_else(|err| abort_call_site!("Failed to read crate path: {:?}.", err));

    let args = syn::Ident::new("KernelArgs", proc_macro::Span::def_site().into());
    let args_trait = super::args_trait::quote_args_trait(&args, impl_generics, inputs);

    let cpu_func_lifetime_erased_types =
        generate_lifetime_erased_types(crate_path, &args, generics, inputs, macro_type_ids);

    let ptx_cstr_ident = syn::Ident::new(PTX_CSTR_IDENT, func_ident.span());

    let matching_kernel_assert = if skip_kernel_compilation() {
        quote!()
    } else {
        quote::quote_spanned! { func_ident.span()=>
            const _: #crate_path::safety::ptx_entry_point::Assert<{
                #crate_path::safety::ptx_entry_point::HostAndDeviceKernelEntryPoint::Match
            }> = #crate_path::safety::ptx_entry_point::Assert::<{
                #crate_path::safety::ptx_entry_point::check(
                    #ptx_cstr_ident.to_bytes(),
                    #crate_path::kernel::specialise_kernel_entry_point!(
                        #func_ident_hash #generic_start_token
                            #($#macro_type_ids),*
                        #generic_close_token
                    ).to_bytes(),
                )
            }>;
        }
    };

    let signature_layout_assert = if skip_kernel_compilation() {
        quote!()
    } else {
        let ffi_signature_ident = syn::Ident::new(KERNEL_TYPE_LAYOUT_IDENT, func_ident.span());
        let ffi_signature_ty = quote! { extern "C" fn(#(#cpu_func_lifetime_erased_types),*) };

        quote::quote_spanned! { func_ident.span()=>
            const _: #crate_path::safety::ptx_kernel_signature::Assert<{
                #crate_path::safety::ptx_kernel_signature::HostAndDeviceKernelSignatureTypeLayout::Match
            }> = #crate_path::safety::ptx_kernel_signature::Assert::<{
                #crate_path::safety::ptx_kernel_signature::check::<#ffi_signature_ty>(#ffi_signature_ident)
            }>;
        }
    };

    let private_func_params = func_params
        .iter()
        .map(|param| {
            let mut private = syn::Ident::clone(param);
            private.set_span(proc_macro::Span::def_site().into());
            private
        })
        .collect::<Vec<_>>();

    quote! {
        fn get_ptx() -> &'static ::core::ffi::CStr {
            #args_trait

            extern "C" { #(
                #[allow(dead_code)]
                #[deny(improper_ctypes)]
                static #private_func_params: #cpu_func_lifetime_erased_types;
            )* }

            #crate_path::kernel::compile_kernel!{
                #func_ident #func_ident_hash #crate_name #crate_manifest_dir #generic_start_token
                    #($#macro_type_ids),*
                #generic_close_token #ptx_lint_levels
            }

            #matching_kernel_assert

            #signature_layout_assert

            #ptx_cstr_ident
        }
    }
}

fn generate_lifetime_erased_types(
    crate_path: &syn::Path,
    args: &syn::Ident,
    DeclGenerics {
        generic_start_token,
        generic_close_token,
        ..
    }: &DeclGenerics,
    FunctionInputs { func_inputs }: &FunctionInputs,
    macro_type_ids: &[syn::Ident],
) -> Vec<proc_macro2::TokenStream> {
    func_inputs
        .iter()
        .enumerate()
        .map(|(i, syn::PatType { ty, .. })| {
            let type_ident = quote::format_ident!("__T_{}", i);

            let mut specialised_ty = quote::quote_spanned! { ty.span()=>
                <() as #args #generic_start_token
                    #($#macro_type_ids),*
                #generic_close_token>::#type_ident
            };
            // the args trait has to unbox outer lifetimes, so we need to add them back in here
            if let syn::Type::Reference(syn::TypeReference { and_token, lifetime, mutability, .. }) = &**ty {
                let lifetime = quote::quote_spanned! { lifetime.span()=> 'static };

                specialised_ty = quote! { #and_token #lifetime #mutability #specialised_ty };
            }

            quote::quote_spanned! { ty.span()=>
                <#specialised_ty as #crate_path::kernel::CudaKernelParameter>::FfiType<'static, 'static>
            }
        }).collect()
}
