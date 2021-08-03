use proc_macro::TokenStream;

mod config;
mod generate;
mod inputs;
mod parse;

use config::KernelConfig;
use generate::{
    args_trait::quote_args_trait, cpu_linker_macro::generate_cpu_linker_macro,
    cpu_wrapper::quote_cpu_wrapper, cuda_generic_function::generate_cuda_generic_function,
    cuda_wrapper::generate_cuda_wrapper,
};
use inputs::{parse_function_inputs, FunctionInputs};
use parse::parse_kernel_fn;

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

    let decl_generics = DeclGenerics {
        generic_start_token: &func.sig.generics.lt_token,
        generic_params: &func.sig.generics.params,
        generic_close_token: &func.sig.generics.gt_token,
        generic_where_clause: &func.sig.generics.where_clause,
    };
    let impl_generics = {
        let (impl_generics, ty_generics, where_clause) = func.sig.generics.split_for_impl();

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

    let func_inputs = parse_function_inputs(&func);
    let func_params = func_inputs
        .func_inputs
        .iter()
        .map(|arg| match arg {
            syn::FnArg::Typed(syn::PatType { pat, .. }) => (&**pat).clone(),
            syn::FnArg::Receiver(_) => unreachable!(),
        })
        .collect::<Vec<_>>();

    let func_type_errors: Vec<syn::Ident> = func_inputs
        .func_inputs
        .iter()
        .map(|arg| match arg {
            syn::FnArg::Typed(syn::PatType { pat, .. }) => {
                quote::format_ident!(
                    "CudaParameter_{}_MustFitInto64BitsOrBeAReference",
                    quote::ToTokens::to_token_stream(pat)
                        .to_string()
                        .replace(' ', "_")
                )
            },
            syn::FnArg::Receiver(_) => unreachable!(),
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
    let cpu_linker_macro = generate_cpu_linker_macro(
        &config,
        &decl_generics,
        &func_inputs,
        &func_ident,
        &func_params,
        &func.attrs,
        &func_type_errors,
    );
    let cuda_wrapper = generate_cuda_wrapper(
        &config,
        &func_inputs,
        &func_ident,
        &func.attrs,
        &func_params,
        &func_type_errors,
    );
    let cuda_generic_function = generate_cuda_generic_function(
        &decl_generics,
        &func_inputs,
        &func_ident,
        &func.attrs,
        &func.block,
    );

    (quote! {
        #args_trait
        #cpu_wrapper

        #cpu_linker_macro

        #cuda_wrapper
        #cuda_generic_function
    })
    .into()
}

enum InputCudaType {
    DeviceCopy,
    LendRustBorrowToCuda,
}

struct InputPtxJit(bool);

struct DeclGenerics<'f> {
    generic_start_token: &'f Option<syn::token::Lt>,
    generic_params: &'f syn::punctuated::Punctuated<syn::GenericParam, syn::token::Comma>,
    generic_close_token: &'f Option<syn::token::Gt>,
    generic_where_clause: &'f Option<syn::WhereClause>,
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
