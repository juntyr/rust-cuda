use proc_macro2::TokenStream;
use quote::quote;

use super::super::{DeclGenerics, FunctionInputs, ImplGenerics, KernelConfig};

pub(in super::super) fn quote_args_trait(
    KernelConfig {
        visibility, args, ..
    }: &KernelConfig,
    DeclGenerics {
        generic_start_token,
        generic_trait_params: generic_params,
        generic_close_token,
        generic_trait_where_clause: generic_where_clause,
        ..
    }: &DeclGenerics,
    ImplGenerics {
        impl_generics,
        ty_generics,
        where_clause,
    }: &ImplGenerics,
    FunctionInputs { func_inputs, .. }: &FunctionInputs,
) -> TokenStream {
    let func_input_typedefs = (0..func_inputs.len())
        .map(|i| {
            let type_ident = quote::format_ident!("__T_{}", i);

            quote! {
                type #type_ident;
            }
        })
        .collect::<Vec<_>>();

    let func_input_types = func_inputs
        .iter()
        .enumerate()
        .map(|(i, arg)| {
            let pat_type = match arg {
                syn::FnArg::Typed(pat_type) => pat_type,
                syn::FnArg::Receiver(_) => unreachable!(),
            };

            let type_ident = quote::format_ident!("__T_{}", i);
            let arg_type = match &*pat_type.ty {
                syn::Type::Reference(syn::TypeReference { elem, .. }) => elem,
                other => other,
            };

            quote! {
                type #type_ident = #arg_type;
            }
        })
        .collect::<Vec<_>>();

    quote! {
        #[cfg(not(target_os = "cuda"))]
        #[allow(clippy::missing_safety_doc)]
        #visibility unsafe trait #args #generic_start_token #generic_params #generic_close_token
            #generic_where_clause
        {
            #(#func_input_typedefs)*
        }

        // #args must always be pub in CUDA kernel as it is used to define the
        //  public kernel entry point signature
        #[cfg(target_os = "cuda")]
        #[allow(clippy::missing_safety_doc)]
        pub unsafe trait #args #generic_start_token #generic_params #generic_close_token
            #generic_where_clause
        {
            #(#func_input_typedefs)*
        }

        unsafe impl #impl_generics #args #ty_generics for () #where_clause {
            #(#func_input_types)*
        }
    }
}
