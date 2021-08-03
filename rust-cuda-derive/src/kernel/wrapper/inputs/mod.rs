use syn::spanned::Spanned;

use super::{InputCudaType, InputPtxJit};

mod attribute;
use attribute::{KernelInputAttribute, KernelInputAttributes};

pub(super) struct FunctionInputs {
    pub(super) func_inputs: syn::punctuated::Punctuated<syn::FnArg, syn::token::Comma>,
    pub(super) func_input_cuda_types: Vec<(InputCudaType, InputPtxJit)>,
}

pub(super) fn parse_function_inputs(func: &syn::ItemFn) -> FunctionInputs {
    let (func_inputs, func_input_cuda_types): (
        syn::punctuated::Punctuated<syn::FnArg, syn::token::Comma>,
        Vec<(InputCudaType, InputPtxJit)>,
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
                                    }
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
