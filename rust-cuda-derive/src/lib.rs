#![deny(clippy::pedantic)]
#![feature(box_patterns)]
#![feature(proc_macro_tracked_env)]
#![feature(bindings_after_at)]

extern crate proc_macro;

#[macro_use]
extern crate proc_macro_error;

use std::path::PathBuf;

use proc_macro::TokenStream;
use syn::spanned::Spanned;

mod generics;
mod lend_to_cuda;
mod rust_to_cuda;

#[proc_macro_error]
#[proc_macro_derive(RustToCuda, attributes(r2cEmbed, r2cBound, r2cEval, r2cPhantom))]
pub fn rust_to_cuda_derive(input: TokenStream) -> TokenStream {
    let ast = match syn::parse(input) {
        Ok(ast) => ast,
        Err(err) => abort!(err),
    };

    // Build the implementation of the `RustToCuda` trait
    rust_to_cuda::impl_rust_to_cuda(&ast)
}

#[proc_macro_error]
#[proc_macro_derive(LendToCuda)]
pub fn lend_to_cuda_derive(input: TokenStream) -> TokenStream {
    let ast = match syn::parse(input) {
        Ok(ast) => ast,
        Err(err) => abort!(err),
    };

    // Build the implementation of the `LendToCuda` trait
    lend_to_cuda::impl_lend_to_cuda(&ast)
}

struct KernelConfig {
    linker: syn::Ident,
    kernel: syn::Ident,
    args: syn::Ident,
    launcher: syn::Path,
}

impl syn::parse::Parse for KernelConfig {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let _use: syn::Token![use] = input.parse()?;
        let linker: syn::Ident = input.parse()?;
        let _bang: syn::Token![!] = input.parse()?;
        let _as: syn::Token![as] = input.parse()?;
        let _impl: syn::Token![impl] = input.parse()?;
        let kernel: syn::Ident = input.parse()?;
        let _lt_token: syn::Token![<] = input.parse()?;
        let args: syn::Ident = input.parse()?;
        let _gt_token: syn::Token![>] = input.parse()?;
        let _for: syn::Token![for] = input.parse()?;
        let launcher: syn::Path = input.parse()?;

        Ok(Self {
            linker,
            kernel,
            args,
            launcher,
        })
    }
}

enum InputCudaType {
    DeviceCopy,
    RustToCuda,
}

enum KernelInputAttribute {
    PassType(proc_macro2::Span, InputCudaType),
    PtxJit(proc_macro2::Span, bool),
}

impl syn::parse::Parse for KernelInputAttribute {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let ident: syn::Ident = input.parse()?;

        match &*ident.to_string() {
            "pass" => {
                let eq: syn::Token![=] = input.parse()?;
                let mode: syn::Ident = input.parse()?;

                let cuda_type = match &*mode.to_string() {
                    "DeviceCopy" => InputCudaType::DeviceCopy,
                    "RustToCuda" => InputCudaType::RustToCuda,
                    _ => abort!(
                        mode.span(),
                        "Unexpected CUDA transfer mode `{:?}`: Expected `DeviceCopy` or \
                         `RustToCuda`.",
                        mode
                    ),
                };

                Ok(KernelInputAttribute::PassType(
                    ident
                        .span()
                        .join(eq.span())
                        .unwrap()
                        .join(mode.span())
                        .unwrap(),
                    cuda_type,
                ))
            },
            "jit" => {
                let eq: Option<syn::Token![=]> = input.parse()?;

                let (ptx_jit, span) = if eq.is_some() {
                    let value: syn::LitBool = input.parse()?;

                    (
                        value.value(),
                        ident
                            .span()
                            .join(eq.span())
                            .unwrap()
                            .span()
                            .join(value.span())
                            .unwrap(),
                    )
                } else {
                    (true, ident.span())
                };

                Ok(KernelInputAttribute::PtxJit(span, ptx_jit))
            },
            _ => abort!(
                ident.span(),
                "Unexpected kernel attribute `{:?}`: Expected `pass` or `jit`.",
                ident
            ),
        }
    }
}

struct KernelInputAttributes(Vec<KernelInputAttribute>);

impl syn::parse::Parse for KernelInputAttributes {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let content;
        let _parens = syn::parenthesized!(content in input);

        syn::punctuated::Punctuated::<KernelInputAttribute, syn::Token![,]>::parse_separated_nonempty(&content).map(|punctuated| {
            Self(punctuated.into_iter().collect())
        })
    }
}

#[proc_macro_error]
#[proc_macro_attribute]
pub fn kernel(attr: TokenStream, func: TokenStream) -> TokenStream {
    let KernelConfig {
        linker,
        kernel,
        args,
        launcher,
    } = match syn::parse_macro_input::parse(attr) {
        Ok(config) => config,
        Err(err) => {
            abort_call_site!(
                "#[kernel(use LINKER! as impl KERNEL<ARGS> for LAUNCHER)] expects LINKER, KERNEL, \
                 ARGS and LAUNCHER identifiers: {:?}",
                err
            )
        },
    };

    let func: syn::ItemFn = syn::parse(func).unwrap_or_else(|err| {
        abort_call_site!(
            "#[kernel(...)] must be wrapped around a function: {:?}",
            err
        )
    });

    if !matches!(func.vis, syn::Visibility::Public(_)) {
        abort!(func.vis.span(), "Kernel function must be public.");
    }

    if func.sig.constness.is_some() {
        abort!(
            func.sig.constness.span(),
            "Kernel function must not be const."
        );
    }

    if func.sig.asyncness.is_some() {
        abort!(
            func.sig.asyncness.span(),
            "Kernel function must not (yet) be async."
        );
    }

    if func.sig.abi.is_some() {
        abort!(
            func.sig.abi.span(),
            "Kernel function must not have an explicit ABI."
        );
    }

    if func.sig.variadic.is_some() {
        abort!(
            func.sig.variadic.span(),
            "Kernel function must not be variadic."
        );
    }

    match &func.sig.output {
        syn::ReturnType::Default => (),
        syn::ReturnType::Type(_, box syn::Type::Tuple(tuple)) if tuple.elems.is_empty() => (),
        syn::ReturnType::Type(_, non_unit_type) => abort!(
            non_unit_type.span(),
            "Kernel function must return the unit type."
        ),
    };

    let (func_inputs, func_input_cuda_types): (
        syn::punctuated::Punctuated<syn::FnArg, syn::Token![,]>,
        Vec<(InputCudaType, bool)>,
    ) = func
        .sig
        .inputs
        .iter()
        .map(|arg| match arg {
            receiver @ syn::FnArg::Receiver(_) => {
                abort!(receiver.span(), "Kernel function must not have a receiver.")
            },
            syn::FnArg::Typed(
                input
                @
                syn::PatType {
                    attrs,
                    pat,
                    colon_token,
                    ty,
                },
            ) => {
                let mut cuda_type: Option<InputCudaType> = None;
                let mut ptx_jit: Option<bool> = None;

                let attrs = attrs
                    .iter()
                    .filter(|attr| match attr.path.get_ident() {
                        Some(ident) if ident == "kernel" => {
                            let attrs: KernelInputAttributes =
                                match syn::parse_macro_input::parse(attr.tokens.clone().into()) {
                                    Ok(data) => data,
                                    Err(err) => abort!(attr.span(), err),
                                };

                            for attr in attrs.0 {
                                match attr {
                                    KernelInputAttribute::PassType(_span, pass_type)
                                        if cuda_type.is_none() =>
                                    {
                                        cuda_type = Some(pass_type);
                                    }
                                    KernelInputAttribute::PassType(span, _pass_type) => {
                                        abort!(span, "Duplicate CUDA transfer mode declaration.");
                                    },
                                    KernelInputAttribute::PtxJit(_span, jit)
                                        if ptx_jit.is_none() =>
                                    {
                                        ptx_jit = Some(jit);
                                    }
                                    KernelInputAttribute::PtxJit(span, _jit) => {
                                        abort!(span, "Duplicate PTX JIT declaration.");
                                    },
                                }
                            }

                            false
                        },
                        _ => true,
                    })
                    .cloned()
                    .collect();

                let cuda_type = cuda_type.unwrap_or_else(|| {
                    abort!(
                        input.span(),
                        "Kernel function input must specify its CUDA transfer mode using \
                         #[kernel(pass = ...)]."
                    );
                });

                (
                    syn::FnArg::Typed(syn::PatType {
                        attrs,
                        pat: pat.clone(),
                        colon_token: *colon_token,
                        ty: ty.clone(),
                    }),
                    (cuda_type, ptx_jit.unwrap_or(false)),
                )
            },
        })
        .unzip();

    let generic_lt_token = &func.sig.generics.lt_token;
    let generic_params = &func.sig.generics.params;
    let generic_gt_token = &func.sig.generics.gt_token;
    let generic_where_clause = &func.sig.generics.where_clause;

    let func_ident = &func.sig.ident;
    let func_attrs = &func.attrs;
    let func_block = &func.block;

    let macro_types = func
        .sig
        .generics
        .params
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

    let macro_type_ids = (0..func.sig.generics.params.len())
        .map(|i| quote::format_ident!("__g_{}", i))
        .collect::<Vec<_>>();

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

    let (impl_generics, ty_generics, where_clause) = func.sig.generics.split_for_impl();

    let new_func_inputs_decl: syn::punctuated::Punctuated<syn::FnArg, syn::token::Comma> =
        func_inputs
            .iter()
            .enumerate()
            .map(|(i, arg)| match arg {
                syn::FnArg::Receiver(receiver) => syn::FnArg::Receiver(receiver.clone()),
                syn::FnArg::Typed(syn::PatType {
                    attrs,
                    pat,
                    colon_token,
                    ty,
                }) => syn::FnArg::Typed(syn::PatType {
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
                            elem: _elem,
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
            })
            .collect();

    let new_func_inputs = func_inputs.iter().enumerate().map(|(i, arg)| {
        match arg {
            syn::FnArg::Typed(syn::PatType { attrs, pat, colon_token, ty }) => {
                let type_ident = quote::format_ident!("__T_{}", i);
                let syn_type = quote! { <() as #args #generic_lt_token #($#macro_type_ids),* #generic_gt_token>::#type_ident };

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

    let ptx_func_inputs: syn::punctuated::Punctuated<syn::FnArg, syn::token::Comma> = func_inputs
        .iter()
        .zip(func_input_cuda_types.iter())
        .enumerate()
        .map(|(i, (arg, (cuda_mode, _ptx_jit)))| match arg {
            syn::FnArg::Typed(syn::PatType {
                attrs,
                pat,
                colon_token,
                ty,
            }) => syn::FnArg::Typed(syn::PatType {
                attrs: attrs.clone(),
                pat: {
                    if let syn::Type::Reference(syn::TypeReference { mutability, .. }) = &**ty {
                        if matches!(cuda_mode, InputCudaType::DeviceCopy) && mutability.is_some() {
                            syn::parse_quote!(mut #pat)
                        } else {
                            pat.clone()
                        }
                    } else {
                        pat.clone()
                    }
                },
                colon_token: *colon_token,
                ty: {
                    let type_ident = quote::format_ident!("__T_{}", i);
                    let syn_type = syn::parse_quote!(
                        rust_cuda::device::specialise_kernel_type!(#args :: #type_ident)
                    );

                    let cuda_type = match cuda_mode {
                        InputCudaType::DeviceCopy => syn_type,
                        InputCudaType::RustToCuda => syn::parse_quote!(
                            <#syn_type as rust_cuda::common::RustToCuda>::CudaRepresentation
                        ),
                    };

                    if let syn::Type::Reference(syn::TypeReference {
                        and_token: _and_token,
                        lifetime,
                        mutability,
                        elem: _elem,
                    }) = &**ty
                    {
                        if lifetime.is_some() {
                            abort!(lifetime.span(), "Kernel parameters cannot have lifetimes.");
                        }

                        if mutability.is_some() {
                            syn::parse_quote!(
                                rust_cuda::common::DeviceBoxMut<#cuda_type>
                            )
                        } else {
                            syn::parse_quote!(
                                rust_cuda::common::DeviceBoxConst<#cuda_type>
                            )
                        }
                    } else if matches!(cuda_mode, InputCudaType::RustToCuda) {
                        abort!(
                            ty.span(),
                            "Kernel parameters transferred using `RustToCuda` must be references."
                        );
                    } else {
                        cuda_type
                    }
                },
            }),
            syn::FnArg::Receiver(_) => unreachable!(),
        })
        .collect();

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

    let args_trait = quote! {
        pub unsafe trait #args #generic_lt_token #generic_params #generic_gt_token #generic_where_clause {
            #(#func_input_typedefs)*
        }

        unsafe impl #impl_generics #args #ty_generics for () #where_clause {
            #(#func_input_types)*
        }
    };

    let cpu_wrapper = quote! {
        #[cfg(not(target_os = "cuda"))]
        pub unsafe trait #kernel #generic_lt_token #generic_params #generic_gt_token #generic_where_clause {
            #(#func_attrs)*
            fn #func_ident(&mut self, #new_func_inputs_decl);
        }
    };

    let cpu_func_types = func_inputs
        .iter()
        .enumerate()
        .map(|(i, arg)| match arg {
            syn::FnArg::Typed(syn::PatType {
                ty: _ty, ..
            }) => {
                let type_ident = quote::format_ident!("__T_{}", i);

                quote!{
                    <() as #args #generic_lt_token #($#macro_type_ids),* #generic_gt_token>::#type_ident
                }
            },
            syn::FnArg::Receiver(_) => unreachable!(),
        })
        .collect::<Vec<_>>();

    let func_type_errors: Vec<syn::Ident> = func_inputs
        .iter()
        .map(|arg| match arg {
            syn::FnArg::Typed(syn::PatType { pat, .. }) => {
                quote::format_ident!(
                    "CudaParameter_{}_MustFitInto64BitOrBeAReference",
                    quote::ToTokens::to_token_stream(pat)
                        .to_string()
                        .replace(' ', "_")
                )
            },
            syn::FnArg::Receiver(_) => unreachable!(),
        })
        .collect();

    let cpu_linker_macro = quote! {
        #[cfg(not(target_os = "cuda"))]
        macro_rules! #linker {
            (#(#macro_types),* $(,)?) => {
                unsafe impl #kernel #generic_lt_token #($#macro_type_ids),* #generic_gt_token for #launcher {
                    #[allow(unused_variables)]
                    #(#func_attrs)*
                    fn #func_ident(&mut self, #(#new_func_inputs),*) {
                        // #(
                        //     #[allow(non_camel_case_types)]
                        //     struct #func_type_errors;
                        //     const _: [#func_type_errors; 1 - { const ASSERT: bool = (::core::mem::size_of::<#cpu_func_types>() <= 8); ASSERT } as usize] = [];
                        // )*

                        const PTX_STR: &str = rust_cuda::host::link_kernel!(#args #crate_name #crate_manifest_dir #generic_lt_token #($#macro_type_ids),* #generic_gt_token);

                        unimplemented!("{:?}", PTX_STR)
                    }
                }
            };
        }
    };

    let cuda_generic_function = quote! {
        #[cfg(target_os = "cuda")]
        #(#func_attrs)*
        fn #func_ident #generic_lt_token #generic_params #generic_gt_token (#func_inputs) #generic_where_clause
        #func_block
    };

    let ptx_func_params: syn::punctuated::Punctuated<syn::Pat, syn::token::Comma> = func
        .sig
        .inputs
        .iter()
        .map(|arg| match arg {
            syn::FnArg::Typed(syn::PatType { pat, .. }) => (&**pat).clone(),
            syn::FnArg::Receiver(_) => unreachable!(),
        })
        .collect();

    let ptx_func_input_unwrap = func_inputs
        .iter().zip(func_input_cuda_types.iter())
        .rev()
        .fold(quote! { #func_ident(#ptx_func_params) }, |inner, (arg, (cuda_mode, _ptx_jit))| match arg {
            syn::FnArg::Typed(syn::PatType {
                attrs: _attrs,
                pat,
                colon_token: _colon_token,
                ty,
            }) => {
                // TODO: Also emit ptx jit markers here

                match cuda_mode {
                    InputCudaType::DeviceCopy => if let syn::Type::Reference(syn::TypeReference { mutability, .. }) = &**ty {
                        if mutability.is_some() {
                            quote! { let #pat = #pat.as_mut(); #inner }
                        } else {
                            quote! { let #pat = #pat.as_ref(); #inner }
                        }
                    } else {
                        inner
                    },
                    InputCudaType::RustToCuda => if let syn::Type::Reference(syn::TypeReference { mutability, .. }) = &**ty {
                        if mutability.is_some() {
                            quote! { rust_cuda::device::BorrowFromRust::with_borrow_from_rust_mut(#pat, |#pat| {
                                #inner
                            }) }
                        } else {
                            quote! { rust_cuda::device::BorrowFromRust::with_borrow_from_rust(#pat, |#pat| {
                                #inner
                            }) }
                        }
                    } else { unreachable!() }
                }
            },
            syn::FnArg::Receiver(_) => unreachable!(),
        });

    let cuda_func_types: Vec<&syn::Type> = ptx_func_inputs
        .iter()
        .map(|arg| match arg {
            syn::FnArg::Typed(syn::PatType { ty, .. }) => &**ty,
            syn::FnArg::Receiver(_) => unreachable!(),
        })
        .collect();

    // TODO:
    //  - add compile time checks that devicecopy types are device copy
    //  - add compile time checks that rusttocuda types are rusttocuda
    //  - include the specialisation hash inside the kernel name
    let cuda_wrapper = quote! {
        #[cfg(target_os = "cuda")]
        #[rust_cuda::device::specialise_kernel_entry(#args)]
        #[no_mangle]
        #(#func_attrs)*
        pub unsafe extern "ptx-kernel" fn #func_ident(#ptx_func_inputs) {
            #(
                #[allow(non_camel_case_types)]
                struct #func_type_errors;
                const _: [#func_type_errors; 1 - { const ASSERT: bool = (::core::mem::size_of::<#cuda_func_types>() <= 8); ASSERT } as usize] = [];
            )*

            #ptx_func_input_unwrap
        }
    };

    (quote! {
        #args_trait

        #cpu_wrapper

        #cpu_linker_macro

        #cuda_wrapper

        #cuda_generic_function
    })
    .into()
}

struct SpecialiseTypeConfig {
    kernel: syn::Ident,
    typedef: syn::Ident,
}

impl syn::parse::Parse for SpecialiseTypeConfig {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let kernel: syn::Ident = input.parse()?;
        let _dc: syn::Token![::] = input.parse()?;
        let typedef: syn::Ident = input.parse()?;

        Ok(Self { kernel, typedef })
    }
}

#[proc_macro_error]
#[proc_macro]
pub fn specialise_kernel_type(tokens: TokenStream) -> TokenStream {
    let SpecialiseTypeConfig { kernel, typedef } = match syn::parse_macro_input::parse(tokens) {
        Ok(config) => config,
        Err(err) => {
            abort_call_site!(
                "specialise_kernel_type!(KERNEL::TYPEDEF) expects KERNEL and TYPEDEF identifiers: \
                 {:?}",
                err
            )
        },
    };

    let crate_name = match std::env::var("CARGO_CRATE_NAME") {
        Ok(crate_name) => crate_name.to_uppercase(),
        Err(err) => abort_call_site!("Failed to read crate name: {:?}", err),
    };

    let specialisation_var = format!(
        "RUST_CUDA_DERIVE_SPECIALISE_{}_{}",
        crate_name,
        kernel.to_string().to_uppercase()
    );

    match proc_macro::tracked_env::var(&specialisation_var) {
        Ok(specialisation) => {
            match format!("<() as {}{}>::{}", kernel, specialisation, typedef).parse() {
                Ok(parsed_specialisation) => parsed_specialisation,
                Err(err) => abort_call_site!("Failed to parse specialisation: {:?}", err),
            }
        },
        Err(err) => abort_call_site!(
            "Failed to read specialisation from {:?}: {:?}",
            &specialisation_var,
            err
        ),
    }
}

struct SpecialiseMangleConfig {
    kernel: syn::Ident,
    specialisation: Option<String>,
}

impl syn::parse::Parse for SpecialiseMangleConfig {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let kernel: syn::Ident = input.parse()?;

        let specialisation = if input.parse::<Option<syn::Token![<]>>()?.is_some() {
            let specialisation_types =
                syn::punctuated::Punctuated::<syn::Type, syn::Token![,]>::parse_separated_nonempty(
                    input,
                )?;
            let _gt_token: syn::Token![>] = input.parse()?;

            Some(
                (quote! { <#specialisation_types> })
                    .to_string()
                    .replace(&[' ', '\n', '\t'][..], ""),
            )
        } else {
            None
        };

        Ok(Self {
            kernel,
            specialisation,
        })
    }
}

#[proc_macro_error]
#[proc_macro]
pub fn specialise_kernel_call(tokens: TokenStream) -> TokenStream {
    let SpecialiseMangleConfig {
        kernel,
        specialisation,
    } = match syn::parse_macro_input::parse(tokens) {
        Ok(config) => config,
        Err(err) => {
            abort_call_site!(
                "specialise_kernel_call!(KERNEL SPECIALISATION) expects KERNEL identifier and \
                 SPECIALISATION tokens: {:?}",
                err
            )
        },
    };

    let mangled_kernel_ident = if let Some(specialisation) = specialisation {
        quote::format_ident!(
            "{}_kernel_{:016x}",
            kernel,
            seahash::hash(specialisation.as_bytes())
        )
    } else {
        quote::format_ident!("{}_kernel", kernel)
    };

    (quote! { #mangled_kernel_ident }).into()
}

#[proc_macro_error]
#[proc_macro_attribute]
pub fn specialise_kernel_entry(attr: TokenStream, func: TokenStream) -> TokenStream {
    let mut func: syn::ItemFn = syn::parse(func).unwrap_or_else(|err| {
        abort_call_site!(
            "#[specialise_kernel_entry(...)] must be wrapped around a function: {:?}",
            err
        )
    });

    let kernel: syn::Ident = match syn::parse_macro_input::parse(attr) {
        Ok(kernel) => kernel,
        Err(err) => abort_call_site!(
            "#[specialise_kernel_entry(KERNEL)] expects KERNEL identifier: {:?}",
            err
        ),
    };

    let crate_name = match std::env::var("CARGO_CRATE_NAME") {
        Ok(crate_name) => crate_name.to_uppercase(),
        Err(err) => abort_call_site!("Failed to read crate name: {:?}", err),
    };

    let specialisation_var = format!(
        "RUST_CUDA_DERIVE_SPECIALISE_{}_{}",
        crate_name,
        kernel.to_string().to_uppercase()
    );

    func.sig.ident = match proc_macro::tracked_env::var(&specialisation_var).as_deref() {
        Ok("") => quote::format_ident!("{}_kernel", func.sig.ident),
        Ok(specialisation) => {
            quote::format_ident!(
                "{}_kernel_{:016x}",
                func.sig.ident,
                seahash::hash(specialisation.as_bytes())
            )
        },
        Err(err) => abort_call_site!(
            "Failed to read specialisation from {:?}: {:?}",
            &specialisation_var,
            err
        ),
    };

    (quote! { #func }).into()
}

struct LinkConfig {
    kernel: syn::Ident,
    crate_name: String,
    crate_path: PathBuf,
    specialisation: Option<String>,
}

impl syn::parse::Parse for LinkConfig {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let kernel: syn::Ident = input.parse()?;
        let name: syn::LitStr = input.parse()?;
        let path: syn::LitStr = input.parse()?;

        let specialisation = if input.parse::<Option<syn::Token![<]>>()?.is_some() {
            let specialisation_types =
                syn::punctuated::Punctuated::<syn::Type, syn::Token![,]>::parse_separated_nonempty(
                    input,
                )?;
            let _gt_token: syn::Token![>] = input.parse()?;

            Some(
                (quote! { <#specialisation_types> })
                    .to_string()
                    .replace(&[' ', '\n', '\t'][..], ""),
            )
        } else {
            None
        };

        Ok(Self {
            kernel,
            crate_name: name.value(),
            crate_path: PathBuf::from(path.value()),
            specialisation,
        })
    }
}

use std::{
    env, fs,
    io::{Read, Write},
    path::Path,
};

use ptx_builder::{
    builder::{BuildStatus, Builder},
    error::{BuildErrorKind, Error, Result},
    reporter::ErrorLogPrinter,
};

fn build_kernel_with_specialisation(
    kernel_path: &Path,
    env_var: &str,
    specialisation: Option<&str>,
) -> Result<PathBuf> {
    env::set_var(env_var, specialisation.unwrap_or(""));

    match Builder::new(kernel_path)?.build()? {
        BuildStatus::Success(output) => {
            let ptx_path = output.get_assembly_path();

            let mut specialised_ptx_path = ptx_path.clone();
            if let Some(specialisation) = specialisation {
                specialised_ptx_path.set_extension(&format!(
                    "{:016x}.ptx",
                    seahash::hash(specialisation.as_bytes())
                ));
            }

            fs::copy(&ptx_path, &specialised_ptx_path).map_err(|err| {
                Error::from(BuildErrorKind::BuildFailed(vec![format!(
                    "Failed to copy kernel from {:?} to {:?}: {}",
                    ptx_path, specialised_ptx_path, err,
                )]))
            })?;

            fs::OpenOptions::new()
                .append(true)
                .open(&specialised_ptx_path)
                .and_then(|mut file| {
                    if let Some(specialisation) = specialisation {
                        writeln!(file, "\n// {}", specialisation)
                    } else {
                        Ok(())
                    }
                })
                .map_err(|err| {
                    Error::from(BuildErrorKind::BuildFailed(vec![format!(
                        "Failed to write specialisation to {:?}: {}",
                        specialised_ptx_path, err,
                    )]))
                })?;

            Ok(specialised_ptx_path)
        },
        BuildStatus::NotNeeded => Err(Error::from(BuildErrorKind::BuildFailed(vec![format!(
            "Kernel build for specialisation {:?} was not needed.",
            &specialisation
        )]))),
    }
}

#[proc_macro_error]
#[proc_macro]
pub fn link_kernel(tokens: TokenStream) -> TokenStream {
    let LinkConfig {
        kernel,
        crate_name,
        crate_path,
        specialisation,
    } = match syn::parse_macro_input::parse(tokens) {
        Ok(config) => config,
        Err(err) => {
            abort_call_site!(
                "link_kernel!(KERNEL NAME PATH SPECIALISATION) expects KERNEL identifier, NAME \
                 and PATH string literals, and SPECIALISATION tokens: {:?}",
                err
            )
        },
    };

    let specialisation_var = format!(
        "RUST_CUDA_DERIVE_SPECIALISE_{}_{}",
        crate_name,
        kernel.to_string().to_uppercase()
    );

    let kernel_ptx = match build_kernel_with_specialisation(
        &crate_path,
        &specialisation_var,
        specialisation.as_deref(),
    ) {
        Ok(kernel_path) => {
            let mut file = fs::File::open(&kernel_path)
                .unwrap_or_else(|_| panic!("Failed to open kernel file at {:?}.", &kernel_path));

            let mut kernel_ptx = String::new();

            file.read_to_string(&mut kernel_ptx)
                .unwrap_or_else(|_| panic!("Failed to read kernel file at {:?}.", &kernel_path));

            kernel_ptx.push('\0');

            kernel_ptx
        },
        Err(err) => {
            abort_call_site!(ErrorLogPrinter::print(err));
        },
    };

    (quote! { #kernel_ptx }).into()
}
