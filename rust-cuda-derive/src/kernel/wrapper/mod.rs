use std::{
    collections::HashMap,
    hash::{Hash, Hasher},
};

use proc_macro::TokenStream;

mod config;
mod generate;
mod inputs;
mod parse;

use super::lints::{parse_ptx_lint_level, LintLevel, PtxLint};

use config::KernelConfig;
use generate::{
    args_trait::quote_args_trait, cpu_linker_macro::quote_cpu_linker_macro,
    cpu_wrapper::quote_cpu_wrapper, cuda_generic_function::quote_cuda_generic_function,
    cuda_wrapper::quote_cuda_wrapper,
};
use inputs::{parse_function_inputs, FunctionInputs};
use parse::parse_kernel_fn;
use proc_macro2::{Ident, Span};
use syn::spanned::Spanned;

#[allow(clippy::too_many_lines)]
pub fn kernel(attr: TokenStream, func: TokenStream) -> TokenStream {
    let mut hasher = seahash::SeaHasher::new();

    attr.to_string().hash(&mut hasher);
    func.to_string().hash(&mut hasher);

    let kernel_hash = hasher.finish();

    let config: KernelConfig = match syn::parse_macro_input::parse(attr) {
        Ok(config) => config,
        Err(err) => {
            abort_call_site!(
                "#[kernel(pub? use LINKER! as impl KERNEL<ARGS, PTX> for LAUNCHER)] expects \
                 LINKER, KERNEL, ARGS, PTX, and LAUNCHER identifiers: {:?}",
                err
            )
        },
    };

    let mut func = parse_kernel_fn(func);

    let mut crate_path = None;
    let mut ptx_lint_levels = HashMap::new();

    func.attrs.retain(|attr| {
        if attr.path.is_ident("kernel") {
            if let Ok(syn::Meta::List(list)) = attr.parse_meta() {
                for meta in &list.nested {
                    match meta {
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

                                    continue;
                                }

                                emit_error!(
                                    s.span(),
                                    "[rust-cuda]: Duplicate #[kernel(crate)] attribute.",
                                );
                            },
                            Err(err) => emit_error!(
                                s.span(),
                                "[rust-cuda]: Invalid #[kernel(crate = \
                                 \"<crate-path>\")] attribute: {}.",
                                err
                            ),
                        },
                        syn::NestedMeta::Meta(syn::Meta::List(syn::MetaList {
                            path,
                            nested,
                            ..
                        })) if path.is_ident("allow") || path.is_ident("warn") || path.is_ident("deny") || path.is_ident("forbid") => {
                            parse_ptx_lint_level(path, nested, &mut ptx_lint_levels);
                        },
                        _ => {
                            emit_error!(
                                meta.span(),
                                "[rust-cuda]: Expected #[kernel(crate = \"<crate-path>\")] or #[kernel(allow/warn/deny/forbid(<lint>))] function attribute."
                            );
                        }
                    }
                }
            } else {
                emit_error!(
                    attr.span(),
                    "[rust-cuda]: Expected #[kernel(crate = \"<crate-path>\")] or or #[kernel(allow/warn/deny/forbid(<lint>))] function attribute."
                );
            }

            false
        } else {
            true
        }
    });

    let crate_path = crate_path.unwrap_or_else(|| syn::parse_quote!(::rust_cuda));

    let _ = ptx_lint_levels.try_insert(PtxLint::Verbose, LintLevel::Allow);
    let _ = ptx_lint_levels.try_insert(PtxLint::DoublePrecisionUse, LintLevel::Warn);
    let _ = ptx_lint_levels.try_insert(PtxLint::LocalMemoryUsage, LintLevel::Warn);
    let _ = ptx_lint_levels.try_insert(PtxLint::RegisterSpills, LintLevel::Warn);
    let _ = ptx_lint_levels.try_insert(PtxLint::DumpBinary, LintLevel::Allow);

    let ptx_lint_levels = {
        let (lints, levels): (Vec<Ident>, Vec<Ident>) = ptx_lint_levels
            .into_iter()
            .map(|(lint, level)| {
                (
                    Ident::new(&lint.to_string(), Span::call_site()),
                    Ident::new(&level.to_string(), Span::call_site()),
                )
            })
            .unzip();

        quote! {
            #(#levels(ptx::#lints)),*
        }
    };

    let mut generic_kernel_params = func.sig.generics.params.clone();
    let mut func_inputs = parse_function_inputs(&func, &mut generic_kernel_params);

    let (generic_start_token, generic_close_token) = if generic_kernel_params.is_empty() {
        (None, None)
    } else if let (Some(start), Some(close)) =
        (func.sig.generics.lt_token, func.sig.generics.gt_token)
    {
        (Some(start), Some(close))
    } else {
        (Some(syn::parse_quote!(<)), Some(syn::parse_quote!(>)))
    };

    let generic_trait_params = generic_kernel_params
        .iter()
        .filter(|generic_param| !matches!(generic_param, syn::GenericParam::Lifetime(_)))
        .cloned()
        .collect();
    let generic_wrapper_params = generic_kernel_params
        .iter()
        .filter(|generic_param| matches!(generic_param, syn::GenericParam::Lifetime(_)))
        .cloned()
        .collect();

    let generic_kernel_where_clause = &func.sig.generics.where_clause;
    let generic_trait_where_clause = generic_kernel_where_clause.as_ref().map(
        |syn::WhereClause {
             where_token,
             predicates,
         }: &syn::WhereClause| {
            let predicates = predicates
                .iter()
                .filter(|predicate| !matches!(predicate, syn::WherePredicate::Lifetime(_)))
                .cloned()
                .collect();

            syn::WhereClause {
                where_token: *where_token,
                predicates,
            }
        },
    );
    let generic_wrapper_where_clause = generic_kernel_where_clause.as_ref().map(
        |syn::WhereClause {
             where_token,
             predicates,
         }: &syn::WhereClause| {
            let predicates = predicates
                .iter()
                .filter(|predicate| matches!(predicate, syn::WherePredicate::Lifetime(_)))
                .cloned()
                .collect();

            syn::WhereClause {
                where_token: *where_token,
                predicates,
            }
        },
    );

    let decl_generics = DeclGenerics {
        generic_start_token: &generic_start_token,
        generic_trait_params: &generic_trait_params,
        generic_close_token: &generic_close_token,
        generic_trait_where_clause: &generic_trait_where_clause,
        generic_wrapper_params: &generic_wrapper_params,
        generic_wrapper_where_clause: &generic_wrapper_where_clause,
        generic_kernel_params: &generic_kernel_params,
        generic_kernel_where_clause,
    };
    let trait_generics = syn::Generics {
        lt_token: generic_start_token,
        params: generic_trait_params.clone(),
        gt_token: generic_close_token,
        where_clause: generic_trait_where_clause.clone(),
    };
    let (impl_generics, ty_generics, where_clause) = trait_generics.split_for_impl();
    let blanket_ty = syn::Ident::new("K", Span::mixed_site());
    let mut blanket_params = generic_trait_params.clone();
    let ptx = &config.ptx;
    blanket_params.push(syn::GenericParam::Type(syn::TypeParam {
        attrs: Vec::new(),
        ident: blanket_ty.clone(),
        colon_token: syn::parse_quote!(:),
        bounds: syn::parse_quote!(#ptx #ty_generics),
        eq_token: None,
        default: None,
    }));
    let trait_blanket_generics = syn::Generics {
        lt_token: Some(generic_start_token.unwrap_or(syn::parse_quote!(<))),
        params: blanket_params,
        gt_token: Some(generic_close_token.unwrap_or(syn::parse_quote!(>))),
        where_clause: generic_trait_where_clause.clone(),
    };
    let (blanket_impl_generics, _, blanket_where_clause) = trait_blanket_generics.split_for_impl();
    let blanket_generics = BlanketGenerics {
        blanket_ty,
        impl_generics: blanket_impl_generics,
        where_clause: blanket_where_clause,
    };
    let impl_generics = ImplGenerics {
        impl_generics,
        ty_generics,
        where_clause,
    };

    let func_ident = FuncIdent {
        func_ident: &func.sig.ident,
        func_ident_async: quote::format_ident!("{}_async", &func.sig.ident),
        func_ident_hash: quote::format_ident!("{}_{:016x}", &func.sig.ident, kernel_hash),
    };

    let func_params = func_inputs
        .func_inputs
        .iter()
        .enumerate()
        .map(|(i, arg)| match arg {
            syn::FnArg::Typed(syn::PatType { pat, .. }) => match ident_from_pat(pat) {
                Some(ident) => ident,
                None => syn::Ident::new(&format!("{}_arg_{i}", func_ident.func_ident), pat.span()),
            },
            syn::FnArg::Receiver(_) => unreachable!(),
        })
        .collect::<Vec<_>>();

    let pat_func_inputs = func_inputs
        .func_inputs
        .iter_mut()
        .zip(&func_params)
        .map(|(arg, ident)| match arg {
            syn::FnArg::Typed(syn::PatType {
                attrs,
                colon_token,
                ty,
                ..
            }) => {
                let ident_fn_arg = syn::FnArg::Typed(syn::PatType {
                    attrs: attrs.clone(),
                    pat: Box::new(syn::Pat::Ident(syn::PatIdent {
                        attrs: Vec::new(),
                        by_ref: None,
                        mutability: None,
                        ident: ident.clone(),
                        subpat: None,
                    })),
                    colon_token: *colon_token,
                    ty: ty.clone(),
                });

                std::mem::replace(arg, ident_fn_arg)
            },
            syn::FnArg::Receiver(_) => unreachable!(),
        })
        .collect();

    let args_trait = quote_args_trait(&config, &decl_generics, &impl_generics, &func_inputs);
    let cpu_wrapper = quote_cpu_wrapper(
        &crate_path,
        &config,
        &decl_generics,
        &impl_generics,
        &blanket_generics,
        &func_inputs,
        &func_ident,
        &func_params,
        &func.attrs,
    );
    let cpu_cuda_check = quote_generic_check(&crate_path, &func_ident, &config);
    let cpu_linker_macro = quote_cpu_linker_macro(
        &crate_path,
        &config,
        &decl_generics,
        &func_inputs,
        &func_ident,
        &func_params,
        &ptx_lint_levels,
    );
    let cuda_wrapper = quote_cuda_wrapper(
        &crate_path,
        &config,
        &func_inputs,
        &func_ident,
        &func.attrs,
        &func_params,
    );
    let cuda_generic_function = quote_cuda_generic_function(
        &decl_generics,
        &pat_func_inputs,
        &func_ident,
        &func.attrs,
        &func.block,
    );

    (quote! {
        #args_trait
        #cpu_wrapper

        #cpu_cuda_check

        #cpu_linker_macro

        #cuda_wrapper
        #cuda_generic_function
    })
    .into()
}

enum InputCudaType {
    SafeDeviceCopy,
    LendRustToCuda,
}

struct InputPtxJit(bool);

#[allow(clippy::struct_field_names)]
struct DeclGenerics<'f> {
    generic_start_token: &'f Option<syn::token::Lt>,
    generic_trait_params: &'f syn::punctuated::Punctuated<syn::GenericParam, syn::token::Comma>,
    generic_close_token: &'f Option<syn::token::Gt>,
    generic_trait_where_clause: &'f Option<syn::WhereClause>,
    generic_wrapper_params: &'f syn::punctuated::Punctuated<syn::GenericParam, syn::token::Comma>,
    generic_wrapper_where_clause: &'f Option<syn::WhereClause>,
    generic_kernel_params: &'f syn::punctuated::Punctuated<syn::GenericParam, syn::token::Comma>,
    generic_kernel_where_clause: &'f Option<syn::WhereClause>,
}

struct ImplGenerics<'f> {
    #[allow(clippy::struct_field_names)]
    impl_generics: syn::ImplGenerics<'f>,
    ty_generics: syn::TypeGenerics<'f>,
    where_clause: Option<&'f syn::WhereClause>,
}

struct BlanketGenerics<'f> {
    blanket_ty: syn::Ident,
    impl_generics: syn::ImplGenerics<'f>,
    where_clause: Option<&'f syn::WhereClause>,
}

#[allow(clippy::struct_field_names)]
struct FuncIdent<'f> {
    func_ident: &'f syn::Ident,
    func_ident_async: syn::Ident,
    func_ident_hash: syn::Ident,
}

fn ident_from_pat(pat: &syn::Pat) -> Option<syn::Ident> {
    match pat {
        syn::Pat::Lit(_)
        | syn::Pat::Macro(_)
        | syn::Pat::Path(_)
        | syn::Pat::Range(_)
        | syn::Pat::Rest(_)
        | syn::Pat::Verbatim(_)
        | syn::Pat::Wild(_) => None,
        syn::Pat::Ident(syn::PatIdent { ident, .. }) => Some(ident.clone()),
        syn::Pat::Box(syn::PatBox { pat, .. })
        | syn::Pat::Reference(syn::PatReference { pat, .. })
        | syn::Pat::Type(syn::PatType { pat, .. }) => ident_from_pat(pat),
        syn::Pat::Or(syn::PatOr { cases, .. }) => ident_from_pat_iter(cases.iter()),
        syn::Pat::Slice(syn::PatSlice { elems, .. })
        | syn::Pat::TupleStruct(syn::PatTupleStruct {
            pat: syn::PatTuple { elems, .. },
            ..
        })
        | syn::Pat::Tuple(syn::PatTuple { elems, .. }) => ident_from_pat_iter(elems.iter()),
        syn::Pat::Struct(syn::PatStruct { fields, .. }) => {
            ident_from_pat_iter(fields.iter().map(|field| &*field.pat))
        },
        _ => Err(()).ok(),
    }
}

fn ident_from_pat_iter<'p, I: Iterator<Item = &'p syn::Pat>>(iter: I) -> Option<syn::Ident> {
    iter.filter_map(ident_from_pat)
        .fold(None, |acc: Option<(String, Span)>, ident| {
            if let Some((mut str_acc, span_acc)) = acc {
                str_acc.push('_');
                str_acc.push_str(ident.to_string().trim_matches('_'));

                Some((str_acc, span_acc.join(ident.span()).unwrap()))
            } else {
                Some((ident.to_string(), ident.span()))
            }
        })
        .map(|(string, span)| syn::Ident::new(&string, span))
}

fn quote_generic_check(
    crate_path: &syn::Path,
    FuncIdent {
        func_ident_hash, ..
    }: &FuncIdent,
    KernelConfig { args, .. }: &KernelConfig,
) -> proc_macro2::TokenStream {
    let crate_name = match proc_macro::tracked_env::var("CARGO_CRATE_NAME") {
        Ok(crate_name) => crate_name.to_uppercase(),
        Err(err) => abort_call_site!("Failed to read crate name: {:?}.", err),
    };

    let crate_manifest_dir = proc_macro::tracked_env::var("CARGO_MANIFEST_DIR")
        .unwrap_or_else(|err| abort_call_site!("Failed to read crate path: {:?}.", err));

    quote::quote_spanned! { func_ident_hash.span()=>
        #[cfg(not(target_os = "cuda"))]
        const _: ::core::result::Result<(), ()> = #crate_path::host::check_kernel!(
            #func_ident_hash #args #crate_name #crate_manifest_dir
        );
    }
}
