use quote::quote_spanned;
use syn::spanned::Spanned;

#[expect(clippy::too_many_lines)]
pub fn expand_cuda_struct_generics_where_requested_in_attrs(
    ast: &syn::DeriveInput,
) -> (
    Vec<syn::Attribute>,
    syn::Generics,
    syn::Generics,
    Vec<syn::Attribute>,
    bool,
    syn::Path,
) {
    let mut type_params = ast
        .generics
        .type_params()
        .map(|param| &param.ident)
        .collect::<Vec<_>>();

    let mut struct_attrs_cuda = ast.attrs.clone();
    let mut struct_generics_cuda = ast.generics.clone();
    let mut struct_generics_cuda_async = ast.generics.clone();
    let mut struct_layout_attrs = Vec::new();

    for ty in &type_params {
        let ty_str = ty.to_string();
        struct_layout_attrs.push(syn::parse_quote_spanned! { ty.span() =>
            #[layout(free = #ty_str)]
        });
    }

    let mut r2c_ignore = false;
    let mut r2c_async_impl = None;
    let mut crate_path = None;

    struct_attrs_cuda.retain(|attr| {
        if attr.path.is_ident("cuda") {
            if let Ok(syn::Meta::List(list)) = attr.parse_meta() {
                for meta in &list.nested {
                    match meta {
                        syn::NestedMeta::Meta(syn::Meta::Path(path)) if path.is_ident("ignore") => {
                            r2c_ignore = true;
                        },
                        syn::NestedMeta::Meta(syn::Meta::NameValue(syn::MetaNameValue {
                            path,
                            lit: syn::Lit::Str(s),
                            ..
                        })) if path.is_ident("bound") => match syn::parse_str::<syn::WherePredicate>(&s.value()) {
                            Ok(bound) => {
                                struct_generics_cuda
                                    .make_where_clause()
                                    .predicates
                                    .push(bound.clone());
                                struct_generics_cuda_async
                                    .make_where_clause()
                                    .predicates
                                    .push(bound);
                            },
                            Err(err) => emit_error!(
                                s.span(),
                                "[rust-cuda]: Invalid #[cuda(bound = \"<where-predicate>\")] \
                                 struct attribute: {}.",
                                err
                            ),
                        },
                        syn::NestedMeta::Meta(syn::Meta::NameValue(syn::MetaNameValue {
                            path,
                            lit: syn::Lit::Str(s),
                            ..
                        })) if path.is_ident("free") => {
                            match syn::parse_str::<syn::Ident>(&s.value()) {
                                Ok(param) => {
                                    if let Some(i) = type_params.iter().position(|ty| **ty == param)
                                    {
                                        type_params.swap_remove(i);
                                    } else {
                                        emit_error!(
                                            s.span(),
                                            "[rust-cuda]: Invalid #[cuda(free = \"{}\")] \
                                             attribute: \"{}\" is either not a type parameter or \
                                             has already been freed (duplicate attribute).",
                                            param,
                                            param,
                                        );
                                    }
                                },
                                Err(err) => emit_error!(
                                    s.span(),
                                    "[rust-cuda]: Invalid #[cuda(free = \"<type>\")] attribute: \
                                     {}.",
                                    err
                                ),
                            }
                        },
                        syn::NestedMeta::Meta(syn::Meta::NameValue(syn::MetaNameValue {
                            path,
                            lit: syn::Lit::Bool(b),
                            ..
                        })) if path.is_ident("async") => if r2c_async_impl.is_none() {
                            r2c_async_impl = Some(b.value());
                        } else {
                            emit_error!(
                                b.span(),
                                "[rust-cuda]: Duplicate #[cuda(async)] attribute.",
                            );
                        },
                        syn::NestedMeta::Meta(syn::Meta::NameValue(syn::MetaNameValue {
                            path,
                            lit: syn::Lit::Str(s),
                            ..
                        })) if path.is_ident("crate") => match syn::parse_str::<syn::Path>(&s.value()) {
                            Ok(new_crate_path) => {
                                if crate_path.is_none() {
                                    crate_path = Some(
                                        syn::parse_quote_spanned! { s.span() => #new_crate_path },
                                    );
                                } else {
                                    emit_error!(
                                        s.span(),
                                        "[rust-cuda]: Duplicate #[cuda(crate)] attribute.",
                                    );
                                }
                            },
                            Err(err) => emit_error!(
                                s.span(),
                                "[rust-cuda]: Invalid #[cuda(crate = \
                                 \"<crate-path>\")] attribute: {}.",
                                err
                            ),
                        },
                        syn::NestedMeta::Meta(syn::Meta::NameValue(syn::MetaNameValue {
                            path: syn::Path {
                                leading_colon: None,
                                segments,
                            },
                            lit: syn::Lit::Str(s),
                            ..
                        })) if segments.len() == 2
                            && let syn::PathSegment {
                                ident: layout_ident,
                                arguments: syn::PathArguments::None,
                            } = &segments[0]
                            && let syn::PathSegment {
                                ident: attr_ident,
                                arguments: syn::PathArguments::None,
                            } = &segments[1]
                            && layout_ident == "layout"
                            && !segments.trailing_punct() =>
                        {
                            struct_layout_attrs.push(syn::Attribute {
                                pound_token: attr.pound_token,
                                style: attr.style,
                                bracket_token: attr.bracket_token,
                                path: proc_macro2::Ident::new("layout", attr.path.span()).into(),
                                tokens: quote_spanned!(s.span() => (#attr_ident = #s)),
                            });
                        },
                        _ => {
                            emit_error!(
                                meta.span(),
                                "[rust-cuda]: Expected #[cuda(crate = \"<crate-path>\")] / #[cuda(bound = \"<where-predicate>\")] / #[cuda(free = \"<type>\")] / #[cuda(async = <bool>)] / #[cuda(layout::ATTR = \"VALUE\")] / #[cuda(ignore)] struct attribute."
                            );
                        },
                    }
                }
            } else {
                emit_error!(
                    attr.span(),
                    "[rust-cuda]: Expected #[cuda(crate = \"<crate-path>\")] / #[cuda(bound = \"<where-predicate>\")] / #[cuda(free = \"<type>\")] / #[cuda(async = <bool>)] / #[cuda(layout::ATTR = \"VALUE\")] / #[cuda(ignore)] struct attribute."
                );
            }

            false
        } else {
            !r2c_ignore
        }
    });

    let crate_path = crate_path.unwrap_or_else(|| syn::parse_quote!(::rust_cuda));

    for ty in &type_params {
        struct_generics_cuda
            .make_where_clause()
            .predicates
            .push(syn::parse_quote! {
                #ty: #crate_path::lend::RustToCuda
            });
        struct_generics_cuda_async
            .make_where_clause()
            .predicates
            .push(syn::parse_quote! {
                #ty: #crate_path::lend::RustToCudaAsync
            });
    }

    (
        struct_attrs_cuda,
        struct_generics_cuda,
        struct_generics_cuda_async,
        struct_layout_attrs,
        r2c_async_impl.unwrap_or(true),
        crate_path,
    )
}
