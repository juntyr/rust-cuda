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
        if attr.path().is_ident("cuda") {
            if attr
                .parse_nested_meta(|meta| {
                    if meta.path.is_ident("ignore") {
                        r2c_ignore = true;
                        return Ok(());
                    }

                    if meta.path.is_ident("bound") {
                        match meta
                            .value()
                            .and_then(<syn::LitStr as syn::parse::Parse>::parse)
                            .and_then(|s| syn::parse_str::<syn::WherePredicate>(&s.value()))
                        {
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
                                meta.path.span(),
                                "[rust-cuda]: Invalid #[cuda(bound = \"<where-predicate>\")] \
                                 struct attribute: {}.",
                                err
                            ),
                        }

                        return Ok(());
                    }

                    if meta.path.is_ident("free") {
                        match meta
                            .value()
                            .and_then(<syn::LitStr as syn::parse::Parse>::parse)
                            .and_then(|s| syn::parse_str::<syn::Ident>(&s.value()))
                        {
                            Ok(param) => {
                                if let Some(i) = type_params.iter().position(|ty| **ty == param) {
                                    type_params.swap_remove(i);
                                } else {
                                    emit_error!(
                                        param.span(),
                                        "[rust-cuda]: Invalid #[cuda(free = \"{}\")] attribute: \
                                         \"{}\" is either not a type parameter or has already \
                                         been freed (duplicate attribute).",
                                        param,
                                        param,
                                    );
                                }
                            },
                            Err(err) => emit_error!(
                                meta.path.span(),
                                "[rust-cuda]: Invalid #[cuda(free = \"<type>\")] attribute: {}.",
                                err
                            ),
                        }

                        return Ok(());
                    }

                    if meta.path.is_ident("async") {
                        match meta
                            .value()
                            .and_then(<syn::LitBool as syn::parse::Parse>::parse)
                        {
                            Ok(b) => {
                                if r2c_async_impl.is_none() {
                                    r2c_async_impl = Some(b.value());
                                } else {
                                    emit_error!(
                                        b.span(),
                                        "[rust-cuda]: Duplicate #[cuda(async)] attribute.",
                                    );
                                }
                            },
                            Err(err) => emit_error!(
                                meta.path.span(),
                                "[rust-cuda]: Invalid #[cuda(async = <bool>)] attribute: {}.",
                                err
                            ),
                        }

                        return Ok(());
                    }

                    if meta.path.is_ident("crate") {
                        match meta
                            .value()
                            .and_then(<syn::LitStr as syn::parse::Parse>::parse)
                            .and_then(|s| syn::parse_str(&s.value()))
                        {
                            Ok(new_crate_path) => {
                                if crate_path.is_none() {
                                    crate_path = Some(new_crate_path);
                                } else {
                                    emit_error!(
                                        meta.path.span(),
                                        "[rust-cuda]: Duplicate #[cuda(crate)] attribute.",
                                    );
                                }
                            },
                            Err(err) => emit_error!(
                                meta.path.span(),
                                "[rust-cuda]: Invalid #[cuda(crate = \"<crate-path>\")] \
                                 attribute: {}.",
                                err
                            ),
                        }

                        return Ok(());
                    }

                    if meta.path.leading_colon.is_none()
                        && meta.path.segments.len() == 2
                        && let Some(syn::PathSegment {
                            ident: layout_ident,
                            arguments: syn::PathArguments::None,
                        }) = meta.path.segments.get(0)
                        && let Some(syn::PathSegment {
                            ident: attr_ident,
                            arguments: syn::PathArguments::None,
                        }) = meta.path.segments.get(1)
                        && layout_ident == "layout"
                        && !meta.path.segments.trailing_punct()
                    {
                        match meta
                            .value()
                            .and_then(<syn::LitStr as syn::parse::Parse>::parse)
                        {
                            Ok(s) => struct_layout_attrs.push(syn::Attribute {
                                pound_token: attr.pound_token,
                                style: attr.style,
                                bracket_token: attr.bracket_token,
                                meta: syn::Meta::List(syn::MetaList {
                                    path: proc_macro2::Ident::new("layout", layout_ident.span())
                                        .into(),
                                    delimiter: syn::MacroDelimiter::Brace(syn::token::Brace(
                                        attr.path().span(),
                                    )),
                                    tokens: quote_spanned!(s.span() => #attr_ident = #s),
                                }),
                            }),
                            Err(err) => emit_error!(
                                meta.path.span(),
                                "[rust-cuda]: Invalid #[cuda(layout::ATTR = \"VALUE\")] \
                                 attribute: {}.",
                                err
                            ),
                        }

                        return Ok(());
                    }

                    emit_error!(
                        meta.path.span(),
                        "[rust-cuda]: Expected #[cuda(crate = \"<crate-path>\")] / #[cuda(bound = \
                         \"<where-predicate>\")] / #[cuda(free = \"<type>\")] / #[cuda(async = \
                         <bool>)] / #[cuda(layout::ATTR = \"VALUE\")] / #[cuda(ignore)] struct \
                         attribute."
                    );

                    Ok(())
                })
                .is_err()
            {
                emit_error!(
                    attr.span(),
                    "[rust-cuda]: Expected #[cuda(crate = \"<crate-path>\")] / #[cuda(bound = \
                     \"<where-predicate>\")] / #[cuda(free = \"<type>\")] / #[cuda(async = \
                     <bool>)] / #[cuda(layout::ATTR = \"VALUE\")] / #[cuda(ignore)] struct \
                     attribute."
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
