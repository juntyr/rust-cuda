use proc_macro2::TokenStream;

use super::super::{
    BlanketGenerics, DeclGenerics, FuncIdent, FunctionInputs, ImplGenerics, KernelConfig,
};

mod kernel_func;
mod kernel_func_async;

use kernel_func::quote_kernel_func_inputs;
use kernel_func_async::quote_kernel_func_async;

#[allow(clippy::too_many_arguments)]
pub(in super::super) fn quote_cpu_wrapper(
    crate_path: &syn::Path,
    config @ KernelConfig {
        visibility, kernel, ..
    }: &KernelConfig,
    decl @ DeclGenerics {
        generic_start_token,
        generic_trait_params,
        generic_close_token,
        generic_trait_where_clause,
        ..
    }: &DeclGenerics,
    impl_generics @ ImplGenerics { ty_generics, .. }: &ImplGenerics,
    BlanketGenerics {
        blanket_ty,
        impl_generics: blanket_impl_generics,
        where_clause: blanket_where_clause,
    }: &BlanketGenerics,
    func_inputs: &FunctionInputs,
    fn_ident: &FuncIdent,
    func_params: &[syn::Ident],
    func_attrs: &[syn::Attribute],
) -> TokenStream {
    let kernel_func = quote_kernel_func_inputs(
        crate_path,
        config,
        impl_generics,
        decl,
        func_inputs,
        fn_ident,
        func_params,
        func_attrs,
    );
    let kernel_func_async = quote_kernel_func_async(
        crate_path,
        config,
        impl_generics,
        decl,
        func_inputs,
        fn_ident,
        func_params,
        func_attrs,
    );

    quote! {
        // #[cfg(not(target_os = "cuda"))]
        // #[allow(clippy::missing_safety_doc)]
        // #visibility unsafe trait #kernel #generic_start_token
        //     #generic_trait_params
        // #generic_close_token: #crate_path::host::CompiledKernelPtx<
        //     dyn #kernel #ty_generics
        // > #generic_trait_where_clause
        // {
        //     #kernel_func

        //     #kernel_func_async
        // }

        // #[cfg(not(target_os = "cuda"))]
        // #[allow(clippy::missing_safety_doc)]
        // unsafe impl #blanket_impl_generics #kernel #ty_generics for #blanket_ty
        //     #blanket_where_clause
        // {}

        #kernel_func

        // #kernel_func_async
    }
}
