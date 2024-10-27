use std::{
    collections::HashMap,
    hash::{Hash, Hasher},
};

use proc_macro::TokenStream;

mod config;
mod generate;
mod parse;

use crate::kernel::lints::{parse_ptx_lint_level, LintLevel, PtxLint};

use config::KernelConfig;
use generate::{
    cuda_generic_function::quote_cuda_generic_function, cuda_wrapper::quote_cuda_wrapper,
    host_kernel_ty::quote_host_kernel_ty, host_link_macro::quote_host_link_macro,
};
use parse::parse_kernel_fn;
use proc_macro2::{Ident, Span};
use quote::quote;
use syn::spanned::Spanned;

#[expect(clippy::too_many_lines)]
pub fn kernel(attr: TokenStream, func: TokenStream) -> TokenStream {
    let mut hasher = seahash::SeaHasher::new();

    attr.to_string().hash(&mut hasher);
    func.to_string().hash(&mut hasher);

    let kernel_hash = hasher.finish();

    let config: KernelConfig = match syn::parse(attr) {
        Ok(config) => config,
        Err(err) => {
            abort_call_site!(
                "#[kernel(pub? use LINK! for impl)] expects LINK macro identifier: {:?}",
                err
            )
        },
    };

    let mut func = parse_kernel_fn(func);

    let mut crate_path = None;
    let mut ptx_lint_levels = HashMap::new();

    func.attrs.retain(|attr| {
        if attr.path().is_ident("kernel") {
            if attr
                .parse_nested_meta(|meta| {
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
                                        "[rust-cuda]: Duplicate #[kernel(crate)] attribute.",
                                    );
                                }
                            },
                            Err(err) => emit_error!(
                                meta.path.span(),
                                "[rust-cuda]: Invalid #[kernel(crate = \"<crate-path>\")] \
                                 attribute: {}.",
                                err
                            ),
                        }

                        return Ok(());
                    }

                    if meta.path.is_ident("allow")
                        || meta.path.is_ident("warn")
                        || meta.path.is_ident("deny")
                        || meta.path.is_ident("forbid")
                    {
                        parse_ptx_lint_level(&meta, &mut ptx_lint_levels);
                        return Ok(());
                    }

                    emit_error!(
                        meta.path.span(),
                        "[rust-cuda]: Expected #[kernel(crate = \"<crate-path>\")] or \
                         #[kernel(allow/warn/deny/forbid(<lint>))] function attribute."
                    );

                    Ok(())
                })
                .is_err()
            {
                emit_error!(
                    attr.span(),
                    "[rust-cuda]: Expected #[kernel(crate = \"<crate-path>\")] or or \
                     #[kernel(allow/warn/deny/forbid(<lint>))] function attribute."
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
    let _ = ptx_lint_levels.try_insert(PtxLint::LocalMemoryUse, LintLevel::Warn);
    let _ = ptx_lint_levels.try_insert(PtxLint::RegisterSpills, LintLevel::Warn);
    let _ = ptx_lint_levels.try_insert(PtxLint::DumpAssembly, LintLevel::Allow);
    let _ = ptx_lint_levels.try_insert(PtxLint::DynamicStackSize, LintLevel::Warn);

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

    let mut func_inputs = FunctionInputs {
        func_inputs: func
            .sig
            .inputs
            .into_iter()
            .map(|arg| match arg {
                syn::FnArg::Typed(arg) => arg,
                syn::FnArg::Receiver(_) => {
                    unreachable!("already checked that no receiver arg exists")
                },
            })
            .collect(),
    };

    let generic_kernel_params = func.sig.generics.params.clone();
    let (generic_start_token, generic_close_token) =
        (func.sig.generics.lt_token, func.sig.generics.gt_token);

    let generic_trait_params = generic_kernel_params
        .iter()
        .filter(|generic_param| !matches!(generic_param, syn::GenericParam::Lifetime(_)))
        .cloned()
        .collect();

    let decl_generics = DeclGenerics {
        generic_start_token: &generic_start_token,
        generic_close_token: &generic_close_token,
        generic_kernel_params: &generic_kernel_params,
    };
    let trait_generics = syn::Generics {
        lt_token: generic_start_token,
        params: generic_trait_params,
        gt_token: generic_close_token,
        where_clause: None,
    };
    let (impl_generics, ty_generics, _where_clause) = trait_generics.split_for_impl();
    let impl_generics = ImplGenerics {
        impl_generics,
        ty_generics,
    };

    let func_ident = FuncIdent {
        func_ident: &func.sig.ident,
        func_ident_hash: quote::format_ident!("{}_{:016x}", &func.sig.ident, kernel_hash),
    };

    let func_params = func_inputs
        .func_inputs
        .iter()
        .enumerate()
        .map(|(i, syn::PatType { pat, .. })| match ident_from_pat(pat) {
            Some(ident) => ident,
            None => syn::Ident::new(&format!("{}_arg_{i}", func_ident.func_ident), pat.span()),
        })
        .collect::<Vec<_>>();

    let pat_func_inputs = func_inputs
        .func_inputs
        .iter_mut()
        .zip(&func_params)
        .map(|(arg, ident)| {
            let syn::PatType {
                attrs,
                colon_token,
                ty,
                ..
            } = arg;

            let ident_fn_arg = syn::PatType {
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
            };

            std::mem::replace(arg, ident_fn_arg)
        })
        .collect();

    let host_kernel_ty = quote_host_kernel_ty(
        &crate_path,
        &decl_generics,
        &impl_generics,
        &func_inputs,
        &func_ident,
        &func_params,
        &func.attrs,
    );
    let host_generic_kernel_check = quote_generic_check(&crate_path, &func_ident);
    let host_link_macro = quote_host_link_macro(
        &crate_path,
        &config,
        &decl_generics,
        &impl_generics,
        &func_inputs,
        &func_ident,
        &func_params,
        &ptx_lint_levels,
    );
    let cuda_wrapper = quote_cuda_wrapper(
        &crate_path,
        &func_inputs,
        &func_ident,
        &impl_generics,
        &func.attrs,
        &func_params,
    );
    let cuda_generic_function = quote_cuda_generic_function(
        &crate_path,
        &decl_generics,
        &pat_func_inputs,
        &func_ident,
        &func.attrs,
        &func.block,
    );

    (quote! {
        #host_kernel_ty

        #host_generic_kernel_check

        #host_link_macro

        #cuda_wrapper
        #cuda_generic_function
    })
    .into()
}

struct FunctionInputs {
    func_inputs: syn::punctuated::Punctuated<syn::PatType, syn::token::Comma>,
}

#[expect(clippy::struct_field_names)]
struct DeclGenerics<'f> {
    generic_start_token: &'f Option<syn::token::Lt>,
    generic_close_token: &'f Option<syn::token::Gt>,
    generic_kernel_params: &'f syn::punctuated::Punctuated<syn::GenericParam, syn::token::Comma>,
}

struct ImplGenerics<'f> {
    impl_generics: syn::ImplGenerics<'f>,
    ty_generics: syn::TypeGenerics<'f>,
}

struct FuncIdent<'f> {
    func_ident: &'f syn::Ident,
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
        syn::Pat::Reference(syn::PatReference { pat, .. })
        | syn::Pat::Type(syn::PatType { pat, .. }) => ident_from_pat(pat),
        syn::Pat::Or(syn::PatOr { cases, .. }) => ident_from_pat_iter(cases.iter()),
        syn::Pat::Slice(syn::PatSlice { elems, .. })
        | syn::Pat::TupleStruct(syn::PatTupleStruct { elems, .. })
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
        func_ident,
        func_ident_hash,
        ..
    }: &FuncIdent,
) -> proc_macro2::TokenStream {
    let crate_name = proc_macro::tracked_env::var("CARGO_CRATE_NAME")
        .unwrap_or_else(|err| abort_call_site!("Failed to read crate name: {:?}.", err));

    let crate_manifest_dir = proc_macro::tracked_env::var("CARGO_MANIFEST_DIR")
        .unwrap_or_else(|err| abort_call_site!("Failed to read crate path: {:?}.", err));

    quote::quote_spanned! { func_ident_hash.span()=>
        #[cfg(not(target_os = "cuda"))]
        #crate_path::kernel::check_kernel! {
            #func_ident #func_ident_hash #crate_name #crate_manifest_dir
        }
    }
}
