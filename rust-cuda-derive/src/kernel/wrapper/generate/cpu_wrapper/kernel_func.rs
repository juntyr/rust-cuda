use proc_macro2::TokenStream;
use syn::spanned::Spanned;

use super::super::super::{DeclGenerics, FuncIdent, FunctionInputs, ImplGenerics};

pub(super) fn quote_kernel_func_inputs(
    crate_path: &syn::Path,
    ImplGenerics { ty_generics, .. }: &ImplGenerics,
    DeclGenerics {
        generic_kernel_params,
        generic_start_token,
        generic_close_token,
        ..
    }: &DeclGenerics,
    FunctionInputs { func_inputs }: &FunctionInputs,
    FuncIdent { func_ident, .. }: &FuncIdent,
    func_params: &[syn::Ident],
    func_attrs: &[syn::Attribute],
) -> TokenStream {
    let (kernel_func_inputs, kernel_func_input_tys): (Vec<_>, Vec<_>) = func_inputs
        .iter()
        .map(
            |syn::PatType {
                 attrs,
                 ty,
                 pat,
                 colon_token,
             }| {
                let ty: syn::Type = syn::parse_quote_spanned! { ty.span()=>
                    <#ty as #crate_path::common::CudaKernelParameter>::SyncHostType
                };

                (
                    syn::FnArg::Typed(syn::PatType {
                        attrs: attrs.clone(),
                        ty: Box::new(ty.clone()),
                        pat: pat.clone(),
                        colon_token: *colon_token,
                    }),
                    ty,
                )
            },
        )
        .unzip();

    let cuda_kernel_param_tys = func_inputs
        .iter()
        .map(|syn::PatType { ty, .. }| &**ty)
        .collect::<Vec<_>>();

    let launcher = syn::Ident::new("launcher", proc_macro2::Span::mixed_site());

    let launch = quote::format_ident!("launch{}", func_inputs.len());

    let full_generics = generic_kernel_params
        .iter()
        .map(|param| match param {
            syn::GenericParam::Type(syn::TypeParam { ident, .. })
            | syn::GenericParam::Const(syn::ConstParam { ident, .. }) => quote!(#ident),
            syn::GenericParam::Lifetime(syn::LifetimeDef { lifetime, .. }) => quote!(#lifetime),
        })
        .collect::<Vec<_>>();

    let ty_turbofish = ty_generics.as_turbofish();

    quote! {
        #[cfg(not(target_os = "cuda"))]
        #[allow(non_camel_case_types)]
        pub type #func_ident #generic_start_token
            #generic_kernel_params
        #generic_close_token = impl Copy + Fn(
            &mut #crate_path::host::Launcher<#func_ident #generic_start_token
                #(#full_generics),*
            #generic_close_token>,
            #(#kernel_func_input_tys),*
        ) -> #crate_path::rustacuda::error::CudaResult<()>;

        #[cfg(not(target_os = "cuda"))]
        #(#func_attrs)*
        #[allow(clippy::too_many_arguments)]
        #[allow(clippy::used_underscore_binding)]
        pub fn #func_ident <#generic_kernel_params>(
            #launcher: &mut #crate_path::host::Launcher<#func_ident #generic_start_token
                #(#full_generics),*
            #generic_close_token>,
            #(#kernel_func_inputs),*
        ) -> #crate_path::rustacuda::error::CudaResult<()> {
            let _: #func_ident <#(#full_generics),*> = #func_ident #ty_turbofish;

            #launcher.#launch::<
                #(#cuda_kernel_param_tys),*
            >(#(#func_params),*)
        }
    }
}
