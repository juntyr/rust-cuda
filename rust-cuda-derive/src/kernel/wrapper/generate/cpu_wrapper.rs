use proc_macro2::TokenStream;

use crate::kernel::utils::r2c_move_lifetime;

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
        ..
    }: &FuncIdent,
    func_attrs: &[syn::Attribute],
) -> TokenStream {
    let launcher_predicate = quote! {
        Self: Sized + rust_cuda::host::Launcher<
            KernelTraitObject = dyn #kernel #ty_generics
        >
    };

    let generic_wrapper_where_clause = match generic_wrapper_where_clause {
        Some(syn::WhereClause {
            where_token,
            predicates,
        }) if !predicates.is_empty() => {
            let comma = if predicates.empty_or_trailing() {
                quote!()
            } else {
                quote!(,)
            };

            quote! {
                #where_token #predicates #comma #launcher_predicate
            }
        },
        _ => quote! {
            where #launcher_predicate
        },
    };

    let (new_func_inputs_decl, new_func_inputs_raw_decl) =
        generate_new_func_inputs_decl(config, impl_generics, func_inputs);

    quote! {
        #[cfg(not(target_os = "cuda"))]
        #[allow(clippy::missing_safety_doc)]
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
                        let syn_type: Box<syn::Type> =
                            syn::parse_quote!(<() as #args #ty_generics>::#type_ident);

                        let cuda_type = match cuda_mode {
                            InputCudaType::SafeDeviceCopy => syn::parse_quote!(
                                rust_cuda::utils::device_copy::SafeDeviceCopyWrapper<#syn_type>
                            ),
                            InputCudaType::LendRustToCuda => syn::parse_quote!(
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
                        } else if matches!(cuda_mode, InputCudaType::LendRustToCuda) {
                            let lifetime = r2c_move_lifetime(i, ty);

                            let wrapped_type = syn::parse_quote!(
                                rust_cuda::host::HostAndDeviceOwned<#lifetime, #cuda_type>
                            );

                            Box::new(wrapped_type)
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
