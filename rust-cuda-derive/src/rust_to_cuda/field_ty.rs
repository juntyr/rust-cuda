use syn::{parse_quote, spanned::Spanned};

#[expect(clippy::module_name_repetitions)]
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

pub fn swap_field_type_and_filter_attrs(
    crate_path: &syn::Path,
    field: &mut syn::Field,
) -> CudaReprFieldTy {
    let mut cuda_repr_field_ty: Option<CudaReprFieldTy> = None;
    let mut field_ty = field.ty.clone();

    let mut r2c_ignore = false;

    // Remove all attributes from the fields in the Cuda representation
    field.attrs.retain(|attr| {
        if attr.path().is_ident("cuda") {
            if let Err(err) = attr.parse_nested_meta(|meta| {
                if meta.path.is_ident("ignore") {
                    r2c_ignore = true;
                    return Ok(());
                }

                if meta.path.is_ident("embed") {
                    if cuda_repr_field_ty.is_some() {
                        emit_error!(
                            attr.span(),
                            "[rust-cuda]: Duplicate #[cuda(embed)] field attribute."
                        );
                        return Ok(());
                    }

                    if let Ok(meta) = meta.value() {
                        match meta.parse::<syn::LitStr>().and_then(|s| syn::parse_str(&s.value())) {
                            Ok(proxy_ty) => {
                                let old_field_ty = Box::new(field_ty.clone());
                                field_ty = parse_quote! {
                                    #crate_path::utils::ffi::DeviceAccessible<
                                        <#proxy_ty as #crate_path::lend::RustToCuda>::CudaRepresentation
                                    >
                                };
                                cuda_repr_field_ty = Some(CudaReprFieldTy::RustToCudaProxy {
                                    proxy_ty: Box::new(proxy_ty),
                                    field_ty: old_field_ty,
                                });
                            },
                            Err(err) => emit_error!(
                                meta.span(),
                                "[rust-cuda]: Invalid #[cuda(embed = \
                                \"<proxy-type>\")] field attribute: {}.",
                                err
                            ),
                        }
                    } else {
                        cuda_repr_field_ty = Some(CudaReprFieldTy::RustToCuda {
                            field_ty: Box::new(field_ty.clone()),
                        });
                        field_ty = parse_quote! {
                            #crate_path::utils::ffi::DeviceAccessible<
                                <#field_ty as #crate_path::lend::RustToCuda>::CudaRepresentation
                            >
                        };
                    }

                    return Ok(());
                }

                emit_error!(
                    meta.path.span(),
                    "[rust-cuda]: Expected #[cuda(ignore)] / #[cuda(embed)] / \
                    #[cuda(embed = \"<proxy-type>\")] field attribute"
                );

                Ok(())
            }) {
                emit_error!(
                    attr.span(),
                    "[rust-cuda]: Expected #[cuda(ignore)] / #[cuda(embed)] / \
                    #[cuda(embed = \"<proxy-type>\")] field attribute: {}",
                    err
                );
            }

            false
        } else {
            !r2c_ignore
        }
    });

    #[expect(clippy::option_if_let_else)]
    let cuda_repr_field_ty = if let Some(cuda_repr_field_ty) = cuda_repr_field_ty {
        cuda_repr_field_ty
    } else {
        field_ty = parse_quote! {
            #crate_path::utils::ffi::DeviceAccessible<
                #crate_path::utils::adapter::RustToCudaWithPortableBitCopySemantics<#field_ty>
            >
        };

        CudaReprFieldTy::SafeDeviceCopy
    };

    field.ty = field_ty;

    cuda_repr_field_ty
}
