use proc_macro2::TokenStream;
use syn::spanned::Spanned;
use quote::quote;

use crate::kernel::wrapper::{DeclGenerics, FuncIdent};

pub(in super::super) fn quote_cuda_generic_function(
    crate_path: &syn::Path,
    DeclGenerics {
        generic_start_token,
        generic_kernel_params: generic_params,
        generic_close_token,
        ..
    }: &DeclGenerics,
    func_inputs: &syn::punctuated::Punctuated<syn::PatType, syn::token::Comma>,
    FuncIdent { func_ident, .. }: &FuncIdent,
    func_attrs: &[syn::Attribute],
    func_block: &syn::Block,
) -> TokenStream {
    let mut generic_params = (*generic_params).clone();

    let kernel_func_inputs = func_inputs
        .iter()
        .enumerate()
        .map(
            |(
                i,
                syn::PatType {
                    attrs,
                    ty,
                    pat,
                    colon_token,
                },
            )| {
                let (ty, lt) = if let syn::Type::Reference(syn::TypeReference {
                    and_token,
                    lifetime,
                    mutability,
                    elem,
                }) = &**ty
                {
                    let lifetime = lifetime.clone().unwrap_or_else(|| {
                        let lifetime =
                            syn::Lifetime::new(&format!("'__rust_cuda_lt_{i}"), ty.span());
                        generic_params.insert(
                            0,
                            syn::GenericParam::Lifetime(syn::LifetimeDef {
                                attrs: Vec::new(),
                                colon_token: None,
                                lifetime: lifetime.clone(),
                                bounds: syn::punctuated::Punctuated::new(),
                            }),
                        );
                        lifetime
                    });
                    let lt = quote!(#lifetime);
                    (
                        syn::Type::Reference(syn::TypeReference {
                            and_token: *and_token,
                            lifetime: Some(lifetime),
                            mutability: *mutability,
                            elem: elem.clone(),
                        }),
                        lt,
                    )
                } else {
                    (syn::Type::clone(ty), quote!('_))
                };

                let ty: syn::Type = syn::parse_quote_spanned! { ty.span()=>
                    <#ty as #crate_path::kernel::CudaKernelParameter>::DeviceType<#lt>
                };

                syn::FnArg::Typed(syn::PatType {
                    attrs: attrs.clone(),
                    ty: Box::new(ty),
                    pat: pat.clone(),
                    colon_token: *colon_token,
                })
            },
        )
        .collect::<Vec<_>>();

    let generic_start_token = generic_start_token.unwrap_or_default();
    let generic_close_token = generic_close_token.unwrap_or_default();

    quote! {
        #[cfg(target_os = "cuda")]
        #(#func_attrs)*
        fn #func_ident #generic_start_token #generic_params #generic_close_token (
            #(#kernel_func_inputs),*
        )
        #func_block
    }
}
