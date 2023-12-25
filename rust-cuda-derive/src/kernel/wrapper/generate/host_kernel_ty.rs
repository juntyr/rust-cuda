use proc_macro2::TokenStream;

use super::super::{DeclGenerics, FuncIdent, FunctionInputs, ImplGenerics};

pub(in super::super) fn quote_host_kernel_ty(
    crate_path: &syn::Path,
    DeclGenerics {
        generic_kernel_params,
        generic_start_token,
        generic_close_token,
        ..
    }: &DeclGenerics,
    ImplGenerics { ty_generics, .. }: &ImplGenerics,
    FunctionInputs { func_inputs }: &FunctionInputs,
    FuncIdent { func_ident, .. }: &FuncIdent,
    func_params: &[syn::Ident],
    func_attrs: &[syn::Attribute],
) -> TokenStream {
    let cuda_kernel_param_tys = func_inputs
        .iter()
        .map(|syn::PatType { ty, .. }| &**ty)
        .collect::<Vec<_>>();

    let launcher = syn::Ident::new("launcher", proc_macro2::Span::mixed_site());

    let full_generics = generic_kernel_params
        .iter()
        .map(|param| match param {
            syn::GenericParam::Type(syn::TypeParam { ident, .. })
            | syn::GenericParam::Const(syn::ConstParam { ident, .. }) => quote!(#ident),
            syn::GenericParam::Lifetime(syn::LifetimeDef { lifetime, .. }) => quote!(#lifetime),
        })
        .collect::<Vec<_>>();

    let mut private_func_ident = syn::Ident::clone(func_ident);
    private_func_ident.set_span(proc_macro::Span::def_site().into());

    let ty_turbofish = ty_generics.as_turbofish();

    quote! {
        #[cfg(not(target_os = "cuda"))]
        #[allow(non_camel_case_types)]
        pub type #func_ident #generic_start_token
            #generic_kernel_params
        #generic_close_token = impl Fn(
            &mut #crate_path::kernel::Launcher<#func_ident #generic_start_token
                #(#full_generics),*
            #generic_close_token>,
            #(#cuda_kernel_param_tys),*
        );

        #[cfg(not(target_os = "cuda"))]
        #(#func_attrs)*
        #[allow(clippy::too_many_arguments)]
        #[allow(clippy::used_underscore_binding)]
        fn #private_func_ident #generic_start_token
            #generic_kernel_params
        #generic_close_token (
            #launcher: &mut #crate_path::kernel::Launcher<#func_ident #generic_start_token
                #(#full_generics),*
            #generic_close_token>,
            #func_inputs
        ) {
            let _: #func_ident <#(#full_generics),*> = #private_func_ident #ty_turbofish;

            #(
                let _ = #func_params;
            )*
        }
    }
}
