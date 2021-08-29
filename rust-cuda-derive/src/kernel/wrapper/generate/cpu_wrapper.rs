use proc_macro2::TokenStream;
use syn::spanned::Spanned;

use super::super::{
    DeclGenerics, FuncIdent, FunctionInputs, ImplGenerics, InputCudaType, KernelConfig,
};

pub(in super::super) fn quote_cpu_wrapper(
    config @ KernelConfig {
        visibility, kernel, ..
    }: &KernelConfig,
    DeclGenerics {
        generic_start_token,
        generic_trait_params,
        generic_close_token,
        generic_trait_where_clause,
        generic_wrapper_params,
        generic_wrapper_where_clause,
        ..
    }: &DeclGenerics,
    impl_generics @ ImplGenerics { ty_generics, .. }: &ImplGenerics,
    func_inputs: &FunctionInputs,
    FuncIdent {
        func_ident,
        func_ident_raw,
    }: &FuncIdent,
    func_attrs: &[syn::Attribute],
) -> TokenStream {
    let launcher_predicate = quote! {
        Self: Sized + rust_cuda::host::Launcher<
            KernelTraitObject = dyn #kernel #ty_generics
        >
    };

    let generic_wrapper_where_clause =
        if let Some(generic_wrapper_where_clause) = generic_wrapper_where_clause {
            let comma = if generic_wrapper_where_clause.predicates.empty_or_trailing() {
                quote!()
            } else {
                quote!(,)
            };

            quote! {
                #generic_wrapper_where_clause #comma #launcher_predicate
            }
        } else {
            quote! {
                where #launcher_predicate
            }
        };

    let (new_func_inputs_decl, new_func_inputs_raw_decl) =
        generate_new_func_inputs_decl(config, impl_generics, func_inputs);

    quote! {
        #[cfg(not(target_os = "cuda"))]
        #visibility unsafe trait #kernel #generic_start_token #generic_trait_params #generic_close_token
            #generic_trait_where_clause
        {
            fn get_ptx_str() -> &'static str where #launcher_predicate;

            fn new_kernel() -> rust_cuda::rustacuda::error::CudaResult<
                rust_cuda::host::TypedKernel<dyn #kernel #ty_generics>
            > where #launcher_predicate;

            #(#func_attrs)*
            fn #func_ident #generic_start_token #generic_wrapper_params #generic_close_token (
                &mut self, #(#new_func_inputs_decl),*
            ) -> rust_cuda::rustacuda::error::CudaResult<()>
                #generic_wrapper_where_clause;

            #(#func_attrs)*
            fn #func_ident_raw #generic_start_token #generic_wrapper_params #generic_close_token (
                &mut self, #(#new_func_inputs_raw_decl),*
            ) -> rust_cuda::rustacuda::error::CudaResult<()>
                #generic_wrapper_where_clause;
        }
    }
}

fn generate_new_func_inputs_decl(
    KernelConfig { args, .. }: &KernelConfig,
    ImplGenerics { ty_generics, .. }: &ImplGenerics,
    FunctionInputs {
        func_inputs,
        func_input_cuda_types,
    }: &FunctionInputs,
) -> (Vec<syn::FnArg>, Vec<syn::FnArg>) {
    func_inputs
        .iter()
        .zip(func_input_cuda_types.iter())
        .enumerate()
        .map(|(i, (arg, (cuda_mode, _ptx_jit)))| match arg {
            syn::FnArg::Typed(syn::PatType {
                attrs,
                pat,
                colon_token,
                ty,
            }) => (
                syn::FnArg::Typed(syn::PatType {
                    attrs: attrs.clone(),
                    pat: pat.clone(),
                    colon_token: *colon_token,
                    ty: {
                        let type_ident = quote::format_ident!("__T_{}", i);
                        let syn_type = syn::parse_quote!(<() as #args #ty_generics>::#type_ident);

                        if let syn::Type::Reference(syn::TypeReference {
                            and_token,
                            lifetime,
                            mutability,
                            ..
                        }) = &**ty
                        {
                            Box::new(syn::Type::Reference(syn::TypeReference {
                                and_token: *and_token,
                                lifetime: lifetime.clone(),
                                mutability: *mutability,
                                elem: syn_type,
                            }))
                        } else {
                            syn_type
                        }
                    },
                }),
                syn::FnArg::Typed(syn::PatType {
                    attrs: attrs.clone(),
                    pat: pat.clone(),
                    colon_token: *colon_token,
                    ty: {
                        let type_ident = quote::format_ident!("__T_{}", i);
                        let syn_type = syn::parse_quote!(<() as #args #ty_generics>::#type_ident);

                        let cuda_type = match cuda_mode {
                            InputCudaType::DeviceCopy => syn_type,
                            InputCudaType::RustToCuda => syn::parse_quote!(
                                rust_cuda::common::DeviceAccessible<
                                    <#syn_type as rust_cuda::common::RustToCuda>::CudaRepresentation
                                >
                            ),
                        };

                        if let syn::Type::Reference(syn::TypeReference {
                            lifetime,
                            mutability,
                            ..
                        }) = &**ty
                        {
                            let wrapped_type = if mutability.is_some() {
                                syn::parse_quote!(
                                    rust_cuda::host::HostAndDeviceMutRef<#lifetime, #cuda_type>
                                )
                            } else {
                                syn::parse_quote!(
                                    rust_cuda::host::HostAndDeviceConstRef<#lifetime, #cuda_type>
                                )
                            };

                            Box::new(wrapped_type)
                        } else if matches!(cuda_mode, InputCudaType::RustToCuda) {
                            abort!(
                                ty.span(),
                                "Kernel parameters transferred using `RustToCuda` must be \
                                 references."
                            );
                        } else {
                            cuda_type
                        }
                    },
                }),
            ),
            syn::FnArg::Receiver(_) => unreachable!(),
        })
        .unzip()
}
