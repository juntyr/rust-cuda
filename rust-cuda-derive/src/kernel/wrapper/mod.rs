use proc_macro::TokenStream;

mod config;
mod generate;
mod inputs;
mod parse;

use config::KernelConfig;
use generate::{
    args_trait::quote_args_trait, cpu_linker_macro::quote_cpu_linker_macro,
    cpu_wrapper::quote_cpu_wrapper, cuda_generic_function::quote_cuda_generic_function,
    cuda_wrapper::quote_cuda_wrapper,
};
use inputs::{parse_function_inputs, FunctionInputs};
use parse::parse_kernel_fn;
use proc_macro2::Span;
use syn::spanned::Spanned;

#[allow(clippy::too_many_lines)]
pub fn kernel(attr: TokenStream, func: TokenStream) -> TokenStream {
    let config: KernelConfig = match syn::parse_macro_input::parse(attr) {
        Ok(config) => config,
        Err(err) => {
            abort_call_site!(
                "#[kernel(pub? use LINKER! as impl KERNEL<ARGS> for LAUNCHER)] expects LINKER, \
                 KERNEL, ARGS and LAUNCHER identifiers: {:?}",
                err
            )
        },
    };

    let func = parse_kernel_fn(func);

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
    let impl_generics = {
        let (impl_generics, ty_generics, where_clause) = trait_generics.split_for_impl();

        ImplGenerics {
            impl_generics,
            ty_generics,
            where_clause,
        }
    };

    let func_ident = FuncIdent {
        func_ident: &func.sig.ident,
        func_ident_raw: quote::format_ident!("{}_raw", &func.sig.ident),
    };

    let func_params = func_inputs
        .func_inputs
        .iter()
        .enumerate()
        .map(|(i, arg)| match arg {
            syn::FnArg::Typed(syn::PatType { pat, .. }) => match ident_from_pat(pat) {
                Some(ident) => ident,
                None => {
                    syn::Ident::new(&format!("{}_arg_{}", func_ident.func_ident, i), pat.span())
                },
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

    let func_type_errors: Vec<syn::Ident> = func_params
        .iter()
        .map(|arg| {
            quote::format_ident!(
                "Kernel_parameter_{}_must_fit_into_64b_or_be_a_reference",
                arg,
            )
        })
        .collect();

    let args_trait = quote_args_trait(&config, &decl_generics, &impl_generics, &func_inputs);
    let cpu_wrapper = quote_cpu_wrapper(
        &config,
        &decl_generics,
        &impl_generics,
        &func_inputs,
        &func_ident,
        &func.attrs,
    );
    let cpu_cuda_check = quote_generic_check(&config);
    let cpu_linker_macro = quote_cpu_linker_macro(
        &config,
        &decl_generics,
        &func_inputs,
        &func_ident,
        &func_params,
        &func.attrs,
        &func_type_errors,
    );
    let cuda_wrapper = quote_cuda_wrapper(
        &config,
        &func_inputs,
        &func_ident,
        &func.attrs,
        &func_params,
        &func_type_errors,
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
    DeviceCopy,
    RustToCuda,
}

struct InputPtxJit(bool);

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
    impl_generics: syn::ImplGenerics<'f>,
    ty_generics: syn::TypeGenerics<'f>,
    where_clause: Option<&'f syn::WhereClause>,
}

struct FuncIdent<'f> {
    func_ident: &'f syn::Ident,
    func_ident_raw: syn::Ident,
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
        #[cfg(test)]
        syn::Pat::__TestExhaustive(_) => unimplemented!(),
        #[cfg(not(test))]
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

fn quote_generic_check(KernelConfig { args, .. }: &KernelConfig) -> proc_macro2::TokenStream {
    let crate_name = match proc_macro::tracked_env::var("CARGO_CRATE_NAME") {
        Ok(crate_name) => crate_name.to_uppercase(),
        Err(err) => abort_call_site!("Failed to read crate name: {:?}.", err),
    };

    let crate_manifest_dir = proc_macro::tracked_env::var("CARGO_MANIFEST_DIR")
        .unwrap_or_else(|err| abort_call_site!("Failed to read crate path: {:?}.", err));

    quote! {
        #[cfg(not(target_os = "cuda"))]
        rust_cuda::host::check_kernel! { #args #crate_name #crate_manifest_dir }
    }
}
