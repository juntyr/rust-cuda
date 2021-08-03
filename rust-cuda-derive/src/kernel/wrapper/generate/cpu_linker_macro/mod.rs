use proc_macro2::TokenStream;
use syn::spanned::Spanned;

use super::super::{DeclGenerics, FuncIdent, FunctionInputs, InputCudaType, KernelConfig};

pub(in super::super) fn generate_cpu_linker_macro(
    KernelConfig {
        visibility,
        kernel,
        args,
        linker,
        launcher,
    }: &KernelConfig,
    DeclGenerics {
        generic_start_token,
        generic_params,
        generic_close_token,
        ..
    }: &DeclGenerics,
    FunctionInputs {
        func_inputs,
        func_input_cuda_types,
    }: &FunctionInputs,
    FuncIdent {
        func_ident,
        func_ident_raw,
    }: &FuncIdent,
    func_params: &[syn::Pat],
    func_attrs: &[syn::Attribute],
    func_type_errors: &[syn::Ident],
) -> TokenStream {
    let macro_types = generic_params
        .iter()
        .enumerate()
        .map(|(i, generic)| {
            let generic_ident = quote::format_ident!("__g_{}", i);

            match generic {
                syn::GenericParam::Type(_) => quote!($#generic_ident:ty),
                syn::GenericParam::Lifetime(_) => quote!($#generic_ident:lifetime),
                syn::GenericParam::Const(_) => quote!($#generic_ident:expr),
            }
        })
        .collect::<Vec<_>>();

    let macro_type_ids = (0..generic_params.len())
        .map(|i| quote::format_ident!("__g_{}", i))
        .collect::<Vec<_>>();

    let new_func_inputs = func_inputs.iter().enumerate().map(|(i, arg)| {
        match arg {
            syn::FnArg::Typed(syn::PatType { attrs, pat, colon_token, ty }) => {
                let type_ident = quote::format_ident!("__T_{}", i);
                let syn_type = quote! {
                    <() as #args #generic_start_token #($#macro_type_ids),* #generic_close_token>::#type_ident
                };

                if let syn::Type::Reference(syn::TypeReference {
                    and_token, lifetime, mutability, elem: _elem,
                }) = &**ty {
                    quote! { #(#attrs)* #pat #colon_token #and_token #lifetime #mutability #syn_type }
                } else {
                    quote! { #(#attrs)* #pat #colon_token #syn_type }
                }
            },
            syn::FnArg::Receiver(_) => unreachable!(),
        }
    }).collect::<Vec<_>>();

    let new_func_inputs_raw = func_inputs
        .iter()
        .zip(func_input_cuda_types.iter())
        .enumerate()
        .map(|(i, (arg, (cuda_mode, _ptx_jit)))| match arg {
            syn::FnArg::Typed(syn::PatType {
                attrs,
                pat,
                colon_token,
                ty,
            }) => {
                let type_ident = quote::format_ident!("__T_{}", i);
                let syn_type = quote! {
                    <() as #args #generic_start_token #($#macro_type_ids),* #generic_close_token>::#type_ident
                };

                let cuda_type = match cuda_mode {
                    InputCudaType::DeviceCopy => syn_type,
                    InputCudaType::LendRustBorrowToCuda => quote!(
                        <#syn_type as rust_cuda::common::RustToCuda>::CudaRepresentation
                    ),
                };

                if let syn::Type::Reference(syn::TypeReference {
                    and_token,
                    lifetime,
                    mutability,
                    elem: _elem,
                }) = &**ty
                {
                    if lifetime.is_some() {
                        abort!(lifetime.span(), "Kernel parameters cannot have lifetimes.");
                    }

                    let wrapped_type = if mutability.is_some() {
                        quote!(
                            rust_cuda::host::HostDeviceBoxMut<#cuda_type>
                        )
                    } else {
                        quote!(
                            rust_cuda::host::HostDeviceBoxConst<#cuda_type>
                        )
                    };

                    quote! { #(#attrs)* #pat #colon_token #and_token #lifetime #mutability #wrapped_type }
                } else if matches!(cuda_mode, InputCudaType::LendRustBorrowToCuda) {
                    abort!(
                        ty.span(),
                        "Kernel parameters transferred using `LendRustBorrowToCuda` must be references."
                    );
                } else {
                    quote! { #(#attrs)* #pat #colon_token #cuda_type }
                }
            },
            syn::FnArg::Receiver(_) => unreachable!(),
        })
        .collect::<Vec<_>>();

    let crate_name = match std::env::var("CARGO_CRATE_NAME") {
        Ok(crate_name) => crate_name.to_uppercase(),
        Err(err) => abort_call_site!("Failed to read crate name: {:?}.", err),
    };

    let crate_manifest_dir = match std::env::var_os("CARGO_MANIFEST_DIR") {
        Some(crate_manifest_dir) => {
            let crate_manifest_dir = format!("{:?}", crate_manifest_dir);

            crate_manifest_dir
                .strip_prefix('"')
                .unwrap()
                .strip_suffix('"')
                .unwrap()
                .to_owned()
        },
        None => abort_call_site!("Failed to read crate path: NotPresent."),
    };

    let raw_func_input_wrap = func_inputs
        .iter().zip(func_input_cuda_types.iter())
        .rev()
        .fold(quote! { self.#func_ident_raw(#(#func_params),*) }, |inner, (arg, (cuda_mode, _ptx_jit))| match arg {
            syn::FnArg::Typed(syn::PatType {
                attrs: _attrs,
                pat,
                colon_token: _colon_token,
                ty,
            }) => {
                match cuda_mode {
                    InputCudaType::DeviceCopy => if let syn::Type::Reference(syn::TypeReference { mutability, .. }) = &**ty {
                        let pat_box = match &**pat {
                            syn::Pat::Ident(syn::PatIdent {
                                attrs,
                                by_ref: None,
                                mutability: None,
                                ident,
                                subpat: None,
                            }) => syn::Pat::Ident(syn::PatIdent {
                                attrs: attrs.clone(),
                                by_ref: None,
                                mutability: None,
                                ident: quote::format_ident!("__{}_box", ident),
                                subpat: None,
                            }),
                            _ => abort!(pat.span(), "Unexpected kernel input parameter: only identifiers are accepted."),
                        };

                        let pat_host_box = match &**pat {
                            syn::Pat::Ident(syn::PatIdent {
                                attrs,
                                by_ref: None,
                                mutability: None,
                                ident,
                                subpat: None,
                            }) => syn::Pat::Ident(syn::PatIdent {
                                attrs: attrs.clone(),
                                by_ref: None,
                                mutability: None,
                                ident: quote::format_ident!("__{}_host_box", ident),
                                subpat: None,
                            }),
                            _ => unreachable!(),
                        };

                        if mutability.is_some() {
                            quote! {
                                let mut #pat_box = rust_cuda::rustacuda::memory::DeviceBox::new(#pat)?;
                                let mut #pat_host_box = rust_cuda::host::HostDeviceBoxMut::new(&mut #pat_box, #pat);
                                let __result = {
                                    let #pat = &mut #pat_host_box;
                                    #inner
                                };
                                rust_cuda::rustacuda::memory::CopyDestination::copy_to(&#pat_box, #pat)?;
                                __result
                            }
                        } else {
                            quote! {
                                let #pat_box = rust_cuda::rustacuda::memory::DeviceBox::new(#pat)?;
                                let #pat_host_box = rust_cuda::host::HostDeviceBoxConst::new(&#pat_box, #pat);
                                {
                                    let #pat = &#pat_host_box;
                                    #inner
                                }
                            }
                        }
                    } else {
                        inner
                    },
                    InputCudaType::LendRustBorrowToCuda => if let syn::Type::Reference(syn::TypeReference { mutability, .. }) = &**ty {
                        if mutability.is_some() {
                            quote! { rust_cuda::host::LendToCuda::lend_to_cuda_mut(#pat, |mut #pat| {
                                let #pat = &mut #pat;
                                #inner
                            }) }
                        } else {
                            quote! { rust_cuda::host::LendToCuda::lend_to_cuda(#pat, |#pat| {
                                let #pat = &#pat;
                                #inner
                            }) }
                        }
                    } else { unreachable!() }
                }
            },
            syn::FnArg::Receiver(_) => unreachable!(),
        });

    let (func_input_wrap, func_cpu_ptx_jit_wrap): (Vec<_>, Vec<_>) = func_inputs
        .iter()
        .zip(func_input_cuda_types.iter())
        .map(|(arg, (_cuda_mode, ptx_jit))| match arg {
            syn::FnArg::Typed(syn::PatType {
                attrs: _attrs,
                pat,
                colon_token: _colon_token,
                ty,
            }) => {
                let func_input = if let syn::Type::Reference(_) = &**ty {
                    quote! { #pat.for_device() }
                } else {
                    quote! { #pat }
                };

                let ptx_load = if ptx_jit.0 {
                    quote! { ConstLoad[#pat.for_host()] }
                } else {
                    quote! { Ignore[#pat] }
                };

                (func_input, ptx_load)
            },
            syn::FnArg::Receiver(_) => unreachable!(),
        })
        .unzip();

    let cpu_func_types_launch = func_inputs
        .iter()
        .zip(func_input_cuda_types.iter())
        .enumerate()
        .map(|(i, (arg, (cuda_mode, _ptx_jit)))| match arg {
            syn::FnArg::Typed(syn::PatType {
                ty, ..
            }) => {
                let type_ident = quote::format_ident!("__T_{}", i);
                let syn_type = quote! {
                    <() as #args #generic_start_token #($#macro_type_ids),* #generic_close_token>::#type_ident
                };

                let cuda_type = match cuda_mode {
                    InputCudaType::DeviceCopy => syn_type,
                    InputCudaType::LendRustBorrowToCuda => quote!(
                        <#syn_type as rust_cuda::common::RustToCuda>::CudaRepresentation
                    ),
                };

                if let syn::Type::Reference(syn::TypeReference {
                    mutability, ..
                }) = &**ty
                {
                    if mutability.is_some() {
                        quote!(
                            rust_cuda::common::DeviceBoxMut<#cuda_type>
                        )
                    } else {
                        quote!(
                            rust_cuda::common::DeviceBoxConst<#cuda_type>
                        )
                    }
                } else if matches!(cuda_mode, InputCudaType::LendRustBorrowToCuda) {
                    abort!(
                        ty.span(),
                        "Kernel parameters transferred using `LendRustBorrowToCuda` must be references."
                    );
                } else {
                    cuda_type
                }
            },
            syn::FnArg::Receiver(_) => unreachable!(),
        })
        .collect::<Vec<_>>();

    let cpu_linker_macro_visibility = if visibility.is_some() {
        quote! { #[macro_export] }
    } else {
        quote! {}
    };

    quote! {
        #[cfg(not(target_os = "cuda"))]
        #cpu_linker_macro_visibility
        macro_rules! #linker {
            (#(#macro_types),* $(,)?) => {
                unsafe impl #kernel #generic_start_token #($#macro_type_ids),* #generic_close_token for #launcher #generic_start_token #($#macro_type_ids),* #generic_close_token {
                    fn get_ptx_str() -> &'static str {
                        rust_cuda::host::link_kernel!(#args #crate_name #crate_manifest_dir #generic_start_token #($#macro_type_ids),* #generic_close_token)
                    }

                    fn new_kernel() -> rust_cuda::rustacuda::error::CudaResult<rust_cuda::host::TypedKernel<dyn #kernel #generic_start_token #($#macro_type_ids),* #generic_close_token>> {
                        #[repr(C)]
                        struct TypedKernel {
                            compiler: rust_cuda::ptx_jit::host::compiler::PtxJITCompiler,
                            kernel: Option<rust_cuda::ptx_jit::host::kernel::CudaKernel>,
                            entry_point: Box<[u8]>,
                        }

                        let ptx_cstring = ::std::ffi::CString::new(Self::get_ptx_str()).map_err(|_| rust_cuda::rustacuda::error::CudaError::InvalidPtx)?;

                        let compiler = rust_cuda::ptx_jit::host::compiler::PtxJITCompiler::new(&ptx_cstring);

                        let entry_point_str = rust_cuda::host::specialise_kernel_call!(#func_ident #generic_start_token #($#macro_type_ids),* #generic_close_token);
                        let entry_point_cstring = ::std::ffi::CString::new(entry_point_str).map_err(|_| rust_cuda::rustacuda::error::CudaError::UnknownError)?;
                        let entry_point = entry_point_cstring.into_bytes_with_nul().into_boxed_slice();

                        let typed_kernel = TypedKernel { compiler, kernel: None, entry_point };

                        Ok(unsafe { ::std::mem::transmute(typed_kernel) })
                    }

                    #[allow(unused_variables)]
                    #(#func_attrs)*
                    fn #func_ident(&mut self, #(#new_func_inputs),*) -> rust_cuda::rustacuda::error::CudaResult<()> {
                        #raw_func_input_wrap
                    }

                    #[allow(unused_variables)]
                    #(#func_attrs)*
                    fn #func_ident_raw(&mut self, #(#new_func_inputs_raw),*) -> rust_cuda::rustacuda::error::CudaResult<()> {
                        #[repr(C)]
                        struct TypedKernel {
                            compiler: rust_cuda::ptx_jit::host::compiler::PtxJITCompiler,
                            kernel: Option<rust_cuda::ptx_jit::host::kernel::CudaKernel>,
                            entry_point: Box<[u8]>,
                        }

                        let kernel = rust_cuda::host::Launcher::get_kernel_mut(self);
                        let typed_kernel: &mut TypedKernel = unsafe {
                            &mut *(kernel as *mut rust_cuda::host::TypedKernel<_> as *mut TypedKernel)
                        };
                        let compiler = &mut typed_kernel.compiler;

                        let function = match (rust_cuda::ptx_jit::compilePtxJITwithArguments! {
                            compiler(
                                #(#func_cpu_ptx_jit_wrap),*
                            )
                        }, typed_kernel.kernel.as_mut()) {
                            (rust_cuda::ptx_jit::host::compiler::PtxJITResult::Cached(_), Some(kernel)) => kernel,
                            (rust_cuda::ptx_jit::host::compiler::PtxJITResult::Cached(ptx_cstr) | rust_cuda::ptx_jit::host::compiler::PtxJITResult::Recomputed(ptx_cstr), _) => {
                                // Safety: `entry_point` is created using `CString::into_bytes_with_nul`
                                let entry_point_cstr = unsafe {
                                    ::std::ffi::CStr::from_bytes_with_nul_unchecked(&typed_kernel.entry_point)
                                };

                                let kernel = rust_cuda::ptx_jit::host::kernel::CudaKernel::new(ptx_cstr, entry_point_cstr)?;

                                // Call launcher hook on kernel compilation
                                rust_cuda::host::Launcher::on_compile(self, kernel.get_function())?;

                                // Replace the existing compiled kernel, drop the old one
                                typed_kernel.kernel.insert(kernel)
                            },
                        }.get_function();

                        (|#(#func_params: #cpu_func_types_launch),*| {
                            #(
                                #[allow(non_camel_case_types, dead_code)]
                                struct #func_type_errors;
                                const _: [#func_type_errors; 1 - { const ASSERT: bool = (::std::mem::size_of::<#cpu_func_types_launch>() <= 8); ASSERT } as usize] = [];
                            )*

                            if false {
                                fn assert_impl_devicecopy<T: rust_cuda::rustacuda_core::DeviceCopy>(_val: &T) {}

                                #(assert_impl_devicecopy(&#func_params);)*
                            }

                            let stream = rust_cuda::host::Launcher::get_stream(self);
                            let rust_cuda::host::LaunchConfig {
                                grid, block, shared_memory_size
                            } = rust_cuda::host::Launcher::get_config(self);

                            unsafe { stream.launch(function, grid, block, shared_memory_size,
                                &[
                                    #(
                                        &#func_params as *const _ as *mut ::std::ffi::c_void
                                    ),*
                                ]
                            ) }?;

                            stream.synchronize()
                        })(#(#func_input_wrap),*)
                    }
                }
            };
        }
    }
}
