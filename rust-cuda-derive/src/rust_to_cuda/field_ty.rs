use syn::{parse_quote, spanned::Spanned};

#[allow(clippy::module_name_repetitions)]
pub enum CudaReprFieldTy {
    SafeDeviceCopy,
    RustToCuda {
        field_ty: Box<syn::Type>,
    },
    RustToCudaProxy {
        proxy_ty: Box<syn::Type>,
        field_ty: Box<syn::Type>,
    },
}

pub fn swap_field_type_and_filter_attrs(field: &mut syn::Field) -> CudaReprFieldTy {
    let mut cuda_repr_field_ty: Option<CudaReprFieldTy> = None;
    let mut field_ty = field.ty.clone();

    let mut r2c_ignore = false;

    // Remove all attributes from the fields in the Cuda representation
    field.attrs.retain(|attr| {
        if attr.path.is_ident("cuda") {
            if let Ok(syn::Meta::List(list)) = attr.parse_meta() {
                for meta in &list.nested {
                    match meta {
                        syn::NestedMeta::Meta(syn::Meta::Path(path)) if path.is_ident("ignore") => {
                            r2c_ignore = true;
                        },
                        syn::NestedMeta::Meta(syn::Meta::Path(path)) if path.is_ident("embed") => {
                            if cuda_repr_field_ty.is_none() {
                                cuda_repr_field_ty = Some(CudaReprFieldTy::RustToCuda {
                                    field_ty: Box::new(field_ty.clone()),
                                });
                                field_ty = parse_quote! {
                                    rust_cuda::common::DeviceAccessible<
                                        <#field_ty as rust_cuda::common::RustToCuda>::CudaRepresentation
                                    >
                                };
                            } else {
                                emit_error!(
                                    attr.span(),
                                    "[rust-cuda]: Duplicate #[cuda(embed)] field attribute."
                                );
                            }
                        },
                        syn::NestedMeta::Meta(syn::Meta::NameValue(syn::MetaNameValue {
                            path,
                            lit: syn::Lit::Str(s),
                            ..
                        })) if path.is_ident("embed") => {
                            if cuda_repr_field_ty.is_none() {
                                match syn::parse_str(&s.value()) {
                                    Ok(proxy_ty) => {
                                        let old_field_ty = Box::new(field_ty.clone());
                                        field_ty = parse_quote! {
                                            rust_cuda::common::DeviceAccessible<
                                                <#proxy_ty as rust_cuda::common::RustToCuda>::CudaRepresentation
                                            >
                                        };
                                        cuda_repr_field_ty = Some(CudaReprFieldTy::RustToCudaProxy {
                                            proxy_ty: Box::new(proxy_ty),
                                            field_ty: old_field_ty,
                                        });
                                    },
                                    Err(err) => emit_error!(
                                        s.span(),
                                        "[rust-cuda]: Invalid #[cuda(embed = \
                                        \"<type>\")] field attribute: {}.",
                                        err
                                    ),
                                }
                            } else {
                                emit_error!(
                                    attr.span(),
                                    "[rust-cuda]: Duplicate #[cuda(embed)] field attribute."
                                );
                            }
                        },
                        _ => {
                            emit_error!(
                                meta.span(),
                                "[rust-cuda]: Expected #[cuda(ignore)] / #[cdua(embed)] / \
                                #[cuda(embed = \"<type>\")] field attribute"
                            );
                        }
                    }
                }
            } else {
                emit_error!(
                    attr.span(),
                    "[rust-cuda]: Expected #[cuda(ignore)] / #[cdua(embed)] / \
                    #[cuda(embed = \"<type>\")] field attribute."
                );
            }

            false
        } else {
            !r2c_ignore
        }
    });

    let cuda_repr_field_ty = if let Some(cuda_repr_field_ty) = cuda_repr_field_ty {
        cuda_repr_field_ty
    } else {
        field_ty = parse_quote! {
            rust_cuda::common::DeviceAccessible<
                rust_cuda::utils::device_copy::SafeDeviceCopyWrapper<#field_ty>
            >
        };

        CudaReprFieldTy::SafeDeviceCopy
    };

    field.ty = field_ty;

    cuda_repr_field_ty
}
