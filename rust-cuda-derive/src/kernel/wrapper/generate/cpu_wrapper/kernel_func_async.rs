use proc_macro2::TokenStream;
use syn::spanned::Spanned;

use super::super::super::{DeclGenerics, FuncIdent, FunctionInputs, ImplGenerics, InputCudaType};

#[allow(clippy::too_many_arguments)]
pub(super) fn quote_kernel_func_async(
    crate_path: &syn::Path,
    ImplGenerics { ty_generics, .. }: &ImplGenerics,
    DeclGenerics {
        generic_kernel_params,
        ..
    }: &DeclGenerics,
    func_inputs: &FunctionInputs,
    FuncIdent {
        func_ident,
        func_ident_async,
        ..
    }: &FuncIdent,
    func_params: &[syn::Ident],
    func_attrs: &[syn::Attribute],
) -> TokenStream {
    let launcher = syn::Ident::new("launcher", proc_macro2::Span::mixed_site());
    let stream = syn::Lifetime::new("'stream", proc_macro2::Span::mixed_site());

    let (
        async_params,
        launch_param_types,
        unboxed_param_types,
        launch_param_wrap,
        ptx_jit_param_wrap,
    ) = generate_type_wrap(crate_path, func_inputs, &stream);

    quote! {
        #[cfg(not(target_os = "cuda"))]
        #(#func_attrs)*
        #[allow(clippy::extra_unused_type_parameters)]
        #[allow(clippy::too_many_arguments)]
        #[allow(clippy::used_underscore_binding)]
        #[allow(unused_variables)]
        pub fn #func_ident_async <#stream, #generic_kernel_params>(
            #launcher: &mut #crate_path::host::Launcher<#stream, '_, #func_ident #ty_generics>,
            #(#async_params),*
        ) -> #crate_path::rustacuda::error::CudaResult<()> {
            let kernel_jit_result = if #launcher.config.ptx_jit {
                #launcher.kernel.compile_with_ptx_jit_args(#ptx_jit_param_wrap)?
            } else {
                #launcher.kernel.compile_with_ptx_jit_args(None)?
            };
            let function = match kernel_jit_result {
                #crate_path::host::KernelJITResult::Recompiled(function)
                | #crate_path::host::KernelJITResult::Cached(function) => function,
            };

            #[allow(clippy::redundant_closure_call)]
            (|#(#func_params: #launch_param_types),*| {
                if false {
                    #[allow(dead_code)]
                    fn assert_impl_devicecopy<T: #crate_path::rustacuda_core::DeviceCopy>(_val: &T) {}

                    #[allow(dead_code)]
                    fn assert_impl_no_safe_aliasing<T: #crate_path::safety::NoSafeAliasing>() {}

                    #(assert_impl_devicecopy(&#func_params);)*
                    #(assert_impl_no_safe_aliasing::<#unboxed_param_types>();)*
                }

                let #crate_path::host::LaunchConfig {
                    grid, block, shared_memory_size, ptx_jit: _,
                } = #launcher.config.clone();

                unsafe { #launcher.stream.launch(function, grid, block, shared_memory_size,
                    &[
                        #(
                            &#func_params as *const _ as *mut ::core::ffi::c_void
                        ),*
                    ]
                ) }
            })(#(#launch_param_wrap),*)
        }
    }
}

#[allow(clippy::too_many_lines)] // FIXME
fn generate_type_wrap(
    crate_path: &syn::Path,
    FunctionInputs {
        func_inputs,
        func_input_cuda_types,
    }: &FunctionInputs,
    stream: &syn::Lifetime,
) -> (
    Vec<TokenStream>,
    Vec<TokenStream>,
    Vec<syn::Type>,
    Vec<TokenStream>,
    TokenStream,
) {
    let mut any_ptx_jit = false;

    let mut async_params = Vec::with_capacity(func_inputs.len());
    let mut launch_param_types = Vec::with_capacity(func_inputs.len());
    let mut unboxed_param_types = Vec::with_capacity(func_inputs.len());
    let mut launch_param_wrap = Vec::with_capacity(func_inputs.len());
    let mut ptx_jit_param_wrap = Vec::with_capacity(func_inputs.len());

    func_inputs
        .iter()
        .zip(func_input_cuda_types.iter())
        .for_each(|(arg, (cuda_mode, ptx_jit))| match arg {
            syn::FnArg::Typed(syn::PatType {
                attrs,
                pat,
                colon_token,
                ty,
            }) => {
                ptx_jit_param_wrap.push(if ptx_jit.0 {
                    any_ptx_jit = true;

                    quote! { Some(#crate_path::ptx_jit::arg_as_raw_bytes(#pat.for_host())) }
                } else {
                    quote! { None }
                });

                #[allow(clippy::if_same_then_else)]
                launch_param_wrap.push(if let syn::Type::Reference(_) = &**ty {
                    quote! { unsafe { #pat.for_device_async() } }
                } else if matches!(cuda_mode, InputCudaType::LendRustToCuda) {
                    quote! { unsafe { #pat.for_device_async() } }
                } else {
                    quote! { #pat }
                });

                let unboxed_param_type = match &**ty {
                    syn::Type::Reference(syn::TypeReference { elem, .. }) => elem,
                    other => other,
                };
                unboxed_param_types.push(unboxed_param_type.clone());

                let cuda_param_type = match cuda_mode {
                    InputCudaType::SafeDeviceCopy => quote::quote_spanned! { ty.span()=>
                        #crate_path::utils::device_copy::SafeDeviceCopyWrapper<#unboxed_param_type>
                    },
                    InputCudaType::LendRustToCuda => quote::quote_spanned! { ty.span()=>
                        #crate_path::common::DeviceAccessible<
                            <#unboxed_param_type as #crate_path::common::RustToCuda>::CudaRepresentation
                        >
                    },
                };

                let (async_param, launch_param_type) = if let syn::Type::Reference(syn::TypeReference {
                    mutability,
                    lifetime,
                    ..
                }) = &**ty
                {
                    let lifetime_or_default = lifetime.clone().unwrap_or(syn::parse_quote!('_));
                    let comma: Option<syn::token::Comma> =
                        lifetime.as_ref().map(|_| syn::parse_quote!(,));

                    let (async_param_type, launch_param_type) = if mutability.is_some() {
                        if matches!(cuda_mode, InputCudaType::SafeDeviceCopy) {
                            abort!(
                                mutability.span(),
                                "Cannot mutably alias a `SafeDeviceCopy` kernel parameter."
                            );
                        }

                        (
                            quote::quote_spanned! { ty.span()=>
                                #crate_path::host::HostAndDeviceMutRefAsync<#stream, #lifetime_or_default, #cuda_param_type>
                            },
                            quote::quote_spanned! { ty.span()=>
                                #crate_path::common::DeviceMutRef<#lifetime #comma #cuda_param_type>
                            },
                        )
                    } else {
                        (
                            quote::quote_spanned! { ty.span()=>
                                #crate_path::host::HostAndDeviceConstRefAsync<#stream, #lifetime_or_default, #cuda_param_type>
                            },
                            quote::quote_spanned! { ty.span()=>
                                #crate_path::common::DeviceConstRef<#lifetime #comma #cuda_param_type>
                            },
                        )
                    };

                    (quote! {
                        #(#attrs)* #mutability #pat #colon_token #async_param_type
                    }, launch_param_type)
                } else if matches!(cuda_mode, InputCudaType::LendRustToCuda) {
                    let async_param_type = quote::quote_spanned! { ty.span()=>
                        #crate_path::host::HostAndDeviceOwnedAsync<#stream, '_, #cuda_param_type>
                    };
                    let launch_param_type = quote::quote_spanned! { ty.span()=>
                        #crate_path::common::DeviceMutRef<#cuda_param_type>
                    };

                    (
                        quote! {
                            #(#attrs)* #pat #colon_token #async_param_type
                        },
                        launch_param_type
                    )
                } else {
                    (
                        quote! { #(#attrs)* #pat #colon_token #cuda_param_type },
                        quote! { #cuda_param_type },
                    )
                };

                async_params.push(async_param);
                launch_param_types.push(launch_param_type);
            },
            syn::FnArg::Receiver(_) => unreachable!(),
        });

    let ptx_jit_param_wrap = if any_ptx_jit {
        quote!(Some(&[#(#ptx_jit_param_wrap),*]))
    } else {
        quote!(None)
    };

    (
        async_params,
        launch_param_types,
        unboxed_param_types,
        launch_param_wrap,
        ptx_jit_param_wrap,
    )
}
