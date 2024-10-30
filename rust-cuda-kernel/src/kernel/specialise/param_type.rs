use proc_macro::TokenStream;
use quote::ToTokens;

#[expect(clippy::module_name_repetitions)]
pub fn specialise_kernel_param_type(tokens: TokenStream) -> TokenStream {
    let SpecialiseTypeConfig {
        mut ty,
        generics,
        kernel,
    } = match syn::parse(tokens) {
        Ok(config) => config,
        Err(err) => {
            abort_call_site!(
                "specialise_kernel_param_type!(TY for GENERICS in KERNEL) expects TY type, \
                 GENERICS generics, and KERNEL identifier: {:?}",
                err
            )
        },
    };

    let crate_name = proc_macro::tracked_env::var("CARGO_CRATE_NAME")
        .unwrap_or_else(|err| abort_call_site!("Failed to read crate name: {:?}", err));

    let specialisation_var = format!(
        "RUST_CUDA_DERIVE_SPECIALISE_{}_{}",
        crate_name.to_uppercase(),
        kernel.to_string().to_uppercase()
    );

    let specialisation = match proc_macro::tracked_env::var(&specialisation_var) {
        Ok(specialisation) => specialisation,
        Err(err) => abort_call_site!(
            "Failed to read specialisation from {:?}: {:?}",
            &specialisation_var,
            err
        ),
    };
    let specialisation = match syn::parse_str(&specialisation) {
        _ if specialisation.is_empty() => syn::PathArguments::None,
        Ok(specialisation) => syn::PathArguments::AngleBracketed(specialisation),
        Err(err) => abort_call_site!("Failed to parse specialisation: {:?}", err),
    };

    if let syn::PathArguments::AngleBracketed(syn::AngleBracketedGenericArguments {
        args, ..
    }) = specialisation
    {
        if generics.params.len() != args.len() {
            abort_call_site!(
                "Mismatch specialising {} with {}",
                generics.split_for_impl().1.to_token_stream(),
                args.to_token_stream()
            );
        }

        // replace all lifetimes with 'static
        ty = syn::fold::Fold::fold_type(
            &mut FoldLifetimeAllStatic {
                r#static: syn::parse_quote!('static),
            },
            ty,
        );

        for (generic, arg) in generics.params.into_iter().zip(args.into_iter()) {
            match (generic, arg) {
                (
                    syn::GenericParam::Lifetime(syn::LifetimeParam {
                        lifetime: _generic, ..
                    }),
                    syn::GenericArgument::Lifetime(_arg),
                ) => {
                    // all lifetimes are already replaced with 'static above
                },
                (
                    syn::GenericParam::Const(syn::ConstParam { ident: generic, .. }),
                    syn::GenericArgument::Const(arg),
                ) => {
                    ty = syn::fold::Fold::fold_type(&mut FoldConstGeneric { generic, arg }, ty);
                },
                (
                    syn::GenericParam::Type(syn::TypeParam { ident: generic, .. }),
                    syn::GenericArgument::Type(arg),
                ) => {
                    ty = syn::fold::Fold::fold_type(&mut FoldTypeGeneric { generic, arg }, ty);
                },
                (generic, arg) => abort_call_site!(
                    "Mismatch specialising {} with {}",
                    generic.to_token_stream(),
                    arg.to_token_stream()
                ),
            }
        }
    } else if !generics.params.is_empty() {
        abort_call_site!(
            "Missing specialisation for {}",
            generics.split_for_impl().1.to_token_stream()
        );
    }

    ty.into_token_stream().into()
}

struct SpecialiseTypeConfig {
    ty: syn::Type,
    generics: syn::Generics,
    kernel: syn::Ident,
}

impl syn::parse::Parse for SpecialiseTypeConfig {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let ty: syn::Type = input.parse()?;
        let _for: syn::token::For = input.parse()?;
        let generics: syn::Generics = input.parse()?;
        let _in: syn::token::In = input.parse()?;
        let kernel: syn::Ident = input.parse()?;

        Ok(Self {
            ty,
            generics,
            kernel,
        })
    }
}

struct FoldLifetimeAllStatic {
    r#static: syn::Lifetime,
}

impl syn::fold::Fold for FoldLifetimeAllStatic {
    fn fold_type_reference(&mut self, r#ref: syn::TypeReference) -> syn::TypeReference {
        let syn::TypeReference {
            and_token,
            lifetime: _,
            mutability,
            elem,
        } = r#ref;

        syn::fold::fold_type_reference(
            self,
            syn::TypeReference {
                and_token,
                lifetime: Some(self.r#static.clone()),
                mutability,
                elem,
            },
        )
    }

    fn fold_lifetime(&mut self, lt: syn::Lifetime) -> syn::Lifetime {
        let mut r#static = self.r#static.clone();
        r#static.set_span(lt.span());
        r#static
    }
}

struct FoldConstGeneric {
    generic: syn::Ident,
    arg: syn::Expr,
}

impl syn::fold::Fold for FoldConstGeneric {
    fn fold_generic_argument(&mut self, arg: syn::GenericArgument) -> syn::GenericArgument {
        let syn::GenericArgument::Type(syn::Type::Path(syn::TypePath {
            qself: None,
            path:
                syn::Path {
                    leading_colon: None,
                    segments,
                },
        })) = arg
        else {
            return syn::fold::fold_generic_argument(self, arg);
        };

        if let Some(syn::PathSegment {
            ident,
            arguments: syn::PathArguments::None,
        }) = segments.first()
            && segments.len() == 1
            && ident == &self.generic
        {
            return syn::GenericArgument::Const(self.arg.clone());
        }

        syn::fold::fold_generic_argument(
            self,
            syn::GenericArgument::Type(syn::Type::Path(syn::TypePath {
                qself: None,
                path: syn::Path {
                    leading_colon: None,
                    segments,
                },
            })),
        )
    }

    fn fold_expr(&mut self, expr: syn::Expr) -> syn::Expr {
        let syn::Expr::Path(syn::ExprPath {
            qself: None,
            path:
                syn::Path {
                    leading_colon: None,
                    segments,
                },
            attrs,
        }) = expr
        else {
            return syn::fold::fold_expr(self, expr);
        };

        if let Some(syn::PathSegment {
            ident,
            arguments: syn::PathArguments::None,
        }) = segments.first()
            && segments.len() == 1
            && ident == &self.generic
        {
            return self.arg.clone();
        }

        syn::fold::fold_expr(
            self,
            syn::Expr::Path(syn::ExprPath {
                qself: None,
                path: syn::Path {
                    leading_colon: None,
                    segments,
                },
                attrs,
            }),
        )
    }
}

struct FoldTypeGeneric {
    generic: syn::Ident,
    arg: syn::Type,
}

impl syn::fold::Fold for FoldTypeGeneric {
    fn fold_type(&mut self, ty: syn::Type) -> syn::Type {
        let syn::Type::Path(syn::TypePath {
            qself: None,
            path:
                syn::Path {
                    leading_colon: None,
                    segments,
                },
        }) = ty
        else {
            return syn::fold::fold_type(self, ty);
        };

        if let Some(syn::PathSegment {
            ident,
            arguments: syn::PathArguments::None,
        }) = segments.first()
            && ident == &self.generic
        {
            return if segments.len() > 1 {
                syn::Type::Path(syn::TypePath {
                    qself: Some(syn::QSelf {
                        lt_token: syn::parse_quote!(<),
                        ty: Box::new(self.arg.clone()),
                        position: 0,
                        as_token: None,
                        gt_token: syn::parse_quote!(>),
                    }),
                    path: syn::Path {
                        leading_colon: syn::parse_quote!(::),
                        segments: segments.into_iter().skip(1).collect(),
                    },
                })
            } else {
                self.arg.clone()
            };
        }

        syn::fold::fold_type(
            self,
            syn::Type::Path(syn::TypePath {
                qself: None,
                path: syn::Path {
                    leading_colon: None,
                    segments,
                },
            }),
        )
    }
}
