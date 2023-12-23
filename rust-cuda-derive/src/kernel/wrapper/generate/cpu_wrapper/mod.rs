use proc_macro2::TokenStream;

use super::super::{DeclGenerics, FuncIdent, FunctionInputs, ImplGenerics};

mod kernel_func;
use kernel_func::quote_kernel_func_inputs;

pub(in super::super) fn quote_cpu_wrapper(
    crate_path: &syn::Path,
    decl: &DeclGenerics,
    impl_generics: &ImplGenerics,
    func_inputs: &FunctionInputs,
    fn_ident: &FuncIdent,
    func_params: &[syn::Ident],
    func_attrs: &[syn::Attribute],
) -> TokenStream {
    let kernel_func = quote_kernel_func_inputs(
        crate_path,
        impl_generics,
        decl,
        func_inputs,
        fn_ident,
        func_params,
        func_attrs,
    );

    quote! {
        #kernel_func
    }
}
