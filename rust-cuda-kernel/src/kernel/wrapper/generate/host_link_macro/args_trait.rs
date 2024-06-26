use proc_macro2::TokenStream;
use quote::quote;

use crate::kernel::wrapper::{FunctionInputs, ImplGenerics};

pub(in super::super) fn quote_args_trait(
    args: &syn::Ident,
    ImplGenerics {
        impl_generics,
        ty_generics,
    }: &ImplGenerics,
    FunctionInputs { func_inputs }: &FunctionInputs,
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
        .map(|(i, pat_type)| {
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
        #[allow(non_camel_case_types)]
        pub trait #args #impl_generics {
            #(#func_input_typedefs)*
        }

        impl #impl_generics #args #ty_generics for () {
            #(#func_input_types)*
        }
    }
}
