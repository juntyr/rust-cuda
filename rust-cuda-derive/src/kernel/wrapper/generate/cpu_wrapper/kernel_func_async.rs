use proc_macro2::TokenStream;
use syn::spanned::Spanned;

use super::super::super::{DeclGenerics, FuncIdent, FunctionInputs, ImplGenerics};

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

    let (async_params, launch_param_types, launch_param_wrap, _ptx_jit_param_wrap) =
        generate_type_wrap(crate_path, func_inputs, &stream);

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
                #launcher.kernel.compile_with_ptx_jit_args(None)? // TODO: #ptx_jit_param_wrap)?
            } else {
                #launcher.kernel.compile_with_ptx_jit_args(None)?
            };
            let function = match kernel_jit_result {
                #crate_path::host::KernelJITResult::Recompiled(function)
                | #crate_path::host::KernelJITResult::Cached(function) => function,
            };

            #[allow(clippy::redundant_closure_call)]
            (|#(#func_params: #launch_param_types),*| {
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

fn generate_type_wrap(
    crate_path: &syn::Path,
    FunctionInputs {
        func_inputs,
        func_input_cuda_types,
    }: &FunctionInputs,
    stream: &syn::Lifetime,
) -> (
    Vec<syn::FnArg>,
    Vec<syn::Type>,
    Vec<TokenStream>,
    TokenStream,
) {
    let mut any_ptx_jit = false;

    let mut async_params = Vec::with_capacity(func_inputs.len());
    let mut launch_param_types = Vec::with_capacity(func_inputs.len());
    let mut launch_param_wrap = Vec::with_capacity(func_inputs.len());
    let mut ptx_jit_param_wrap = Vec::with_capacity(func_inputs.len());

    func_inputs
        .iter()
        .zip(func_input_cuda_types.iter())
        .for_each(|(arg, ptx_jit)| match arg {
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

                let async_ty: syn::Type = syn::parse_quote_spanned! { ty.span()=>
                    <#ty as #crate_path::common::CudaKernelParameter>::AsyncHostType<#stream, '_>
                };

                let async_param = syn::FnArg::Typed(syn::PatType {
                    attrs: attrs.clone(),
                    ty: Box::new(async_ty),
                    pat: pat.clone(),
                    colon_token: *colon_token,
                });

                async_params.push(async_param);

                let launch_ty: syn::Type = syn::parse_quote_spanned! { ty.span()=>
                    <#ty as #crate_path::common::CudaKernelParameter>::FfiType<#stream, '_>
                };

                launch_param_types.push(launch_ty);

                let launch_wrap = quote::quote_spanned! { ty.span()=>
                    <#ty as #crate_path::common::CudaKernelParameter>::async_to_ffi(#pat)
                };

                launch_param_wrap.push(launch_wrap);
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
        launch_param_wrap,
        ptx_jit_param_wrap,
    )
}
