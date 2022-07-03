use syn::spanned::Spanned;

use crate::kernel::utils::r2c_move_lifetime;

use super::{InputCudaType, InputPtxJit};

mod attribute;
use attribute::{KernelInputAttribute, KernelInputAttributes};

pub(super) struct FunctionInputs {
    pub(super) func_inputs: syn::punctuated::Punctuated<syn::FnArg, syn::token::Comma>,
    pub(super) func_input_cuda_types: Vec<(InputCudaType, InputPtxJit)>,
}

pub(super) fn parse_function_inputs(
    func: &syn::ItemFn,
    generic_params: &mut syn::punctuated::Punctuated<syn::GenericParam, syn::token::Comma>,
) -> FunctionInputs {
    let mut implicit_lifetime_id: usize = 0;

    let (func_inputs, func_input_cuda_types): (
        syn::punctuated::Punctuated<syn::FnArg, syn::token::Comma>,
        Vec<(InputCudaType, InputPtxJit)>,
    ) = func
        .sig
        .inputs
        .iter()
        .enumerate()
        .map(|(i, arg)| match arg {
            receiver @ syn::FnArg::Receiver(_) => {
                abort!(receiver.span(), "Kernel function must not have a receiver.")
            },
            syn::FnArg::Typed(
                input @ syn::PatType {
                    attrs,
                    pat,
                    colon_token,
                    ty,
                },
            ) => {
                let mut cuda_type: Option<InputCudaType> = None;
                let mut ptx_jit: Option<InputPtxJit> = None;

                let attrs = attrs
                    .iter()
                    .filter(|attr| match attr.path.get_ident() {
                        Some(ident) if ident == "kernel" => {
                            let attrs: KernelInputAttributes =
                                match syn::parse_macro_input::parse(attr.tokens.clone().into()) {
                                    Ok(data) => data,
                                    Err(err) => abort!(attr.span(), err),
                                };

                            for attr in attrs {
                                match attr {
                                    KernelInputAttribute::PassType(_span, pass_type)
                                        if cuda_type.is_none() =>
                                    {
                                        cuda_type = Some(pass_type);
                                    },
                                    KernelInputAttribute::PassType(span, _pass_type) => {
                                        abort!(span, "Duplicate CUDA transfer mode declaration.");
                                    },
                                    KernelInputAttribute::PtxJit(span, jit)
                                        if ptx_jit.is_none() =>
                                    {
                                        if !matches!(&**ty, syn::Type::Reference(_)) && jit {
                                            abort!(
                                                span,
                                                "Only reference types can be PTX JIT loaded."
                                            );
                                        }

                                        ptx_jit = Some(InputPtxJit(jit));
                                    },
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

                let ty = ensure_reference_type_lifetime(
                    i,
                    ty,
                    &cuda_type,
                    &mut implicit_lifetime_id,
                    generic_params,
                );

                (
                    syn::FnArg::Typed(syn::PatType {
                        attrs,
                        pat: pat.clone(),
                        colon_token: *colon_token,
                        ty,
                    }),
                    (cuda_type, ptx_jit.unwrap_or(InputPtxJit(false))),
                )
            },
        })
        .unzip();

    FunctionInputs {
        func_inputs,
        func_input_cuda_types,
    }
}

fn ensure_reference_type_lifetime(
    i: usize,
    ty: &syn::Type,
    cuda_type: &InputCudaType,
    implicit_lifetime_id: &mut usize,
    generic_params: &mut syn::punctuated::Punctuated<syn::GenericParam, syn::token::Comma>,
) -> Box<syn::Type> {
    match ty {
        syn::Type::Reference(syn::TypeReference {
            and_token,
            lifetime,
            mutability,
            elem,
        }) => {
            let lifetime = lifetime.clone().unwrap_or_else(|| {
                let lifetime = syn::Lifetime::new(
                    &format!("'__r2c_lt_{}", implicit_lifetime_id),
                    lifetime.span(),
                );

                generic_params.insert(
                    *implicit_lifetime_id,
                    syn::GenericParam::Lifetime(syn::LifetimeDef {
                        attrs: Vec::new(),
                        lifetime: lifetime.clone(),
                        colon_token: None,
                        bounds: syn::punctuated::Punctuated::new(),
                    }),
                );

                *implicit_lifetime_id += 1;

                lifetime
            });

            let elem = if matches!(cuda_type, InputCudaType::LendRustToCuda) {
                (|| {
                    if let syn::Type::Path(syn::TypePath {
                        path: syn::Path { segments, .. },
                        qself: None,
                    }) = &**elem
                    {
                        if let Some(syn::PathSegment {
                            ident,
                            arguments:
                                syn::PathArguments::AngleBracketed(
                                    syn::AngleBracketedGenericArguments { args, .. },
                                ),
                        }) = segments.last()
                        {
                            if ident == "ShallowCopy" && segments.len() == 1 {
                                match args.last() {
                                    Some(syn::GenericArgument::Type(elem)) if args.len() == 1 => {
                                        return Box::new(elem.clone());
                                    },
                                    _ => {
                                        abort!(
                                            args.span(),
                                            "`ShallowCopy<T>` takes exactly one generic type \
                                             argument."
                                        );
                                    },
                                }
                            }
                        }
                    }

                    emit_warning!(
                        elem.span(),
                        "RustToCuda kernel parameters should be explicitly wrapped with the \
                         `ShallowCopy<T>` marker to communicate their aliasing behaviour."
                    );

                    elem.clone()
                })()
            } else {
                elem.clone()
            };

            Box::new(syn::Type::Reference(syn::TypeReference {
                and_token: *and_token,
                lifetime: Some(lifetime),
                mutability: *mutability,
                elem,
            }))
        },
        ty => {
            if matches!(cuda_type, InputCudaType::LendRustToCuda) {
                generic_params.insert(
                    *implicit_lifetime_id,
                    syn::GenericParam::Lifetime(syn::LifetimeDef {
                        attrs: Vec::new(),
                        lifetime: r2c_move_lifetime(i, ty),
                        colon_token: None,
                        bounds: syn::punctuated::Punctuated::new(),
                    }),
                );

                *implicit_lifetime_id += 1;
            }

            Box::new(ty.clone())
        },
    }
}
