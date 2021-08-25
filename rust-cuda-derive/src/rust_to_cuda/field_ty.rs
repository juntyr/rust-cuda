use syn::{parse_quote, spanned::Spanned};

#[allow(clippy::module_name_repetitions)]
pub enum CudaReprFieldTy {
    StackOnly,
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
        if attr.path.is_ident("r2cEmbed") {
            if cuda_repr_field_ty.is_none() {
                if attr.tokens.is_empty() {
                    cuda_repr_field_ty = Some(CudaReprFieldTy::RustToCuda {
                        field_ty: Box::new(field_ty.clone()),
                    });
                    field_ty = parse_quote! {
                        rust_cuda::common::DeviceAccessible<
                            <#field_ty as rust_cuda::common::RustToCuda>::CudaRepresentation
                        >
                    };
                } else {
                    let proxy_ty: syn::Type = match attr.parse_args() {
                        Ok(proxy_ty) => proxy_ty,
                        Err(_) => {
                            abort!(
                                attr.tokens.span(),
                                "#[r2cEmbed] either takes no arguments, or the type of a proxy \
                                 that implements `RustToCudaProxy`."
                            );
                        },
                    };

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
                }
            } else {
                emit_error!(attr.span(), "Duplicate #[r2cEmbed] attribute definition.");
            }

            false
        } else if attr.path.is_ident("r2cIgnore") {
            if !attr.tokens.is_empty() {
                emit_error!(
                    attr.tokens.span(),
                    "#[r2cIgnore] does not take any arguments."
                );
            }

            r2c_ignore = true;

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
                rust_cuda::utils::stack::StackOnlyWrapper<#field_ty>
            >
        };

        CudaReprFieldTy::StackOnly
    };

    field.ty = field_ty;

    cuda_repr_field_ty
}
