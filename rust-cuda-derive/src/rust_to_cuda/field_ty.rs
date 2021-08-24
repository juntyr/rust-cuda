use syn::{parse_quote, spanned::Spanned};

pub fn swap_field_type_and_filter_attrs(field: &mut syn::Field) -> syn::Type {
    let mut cuda_repr_field_ty: Option<syn::Type> = None;
    let old_field_ty = field.ty.clone();

    let mut r2c_ignore = false;

    // Remove all attributes from the fields in the Cuda representation
    field.attrs.retain(|attr| {
        if attr.path.is_ident("r2cEmbed") {
            if cuda_repr_field_ty.is_none() {
                if !attr.tokens.is_empty() {
                    emit_error!(
                        attr.tokens.span(),
                        "#[r2cEmbed] does not take any arguments."
                    );
                }

                cuda_repr_field_ty = Some(parse_quote! {
                    rust_cuda::common::DeviceAccessible<
                        <#old_field_ty as rust_cuda::common::RustToCuda>::CudaRepresentation
                    >
                });
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

    field.ty = if let Some(cuda_repr_field_ty) = cuda_repr_field_ty {
        cuda_repr_field_ty
    } else {
        parse_quote! {
            rust_cuda::common::DeviceAccessible<
                rust_cuda::utils::stack::StackOnlyWrapper<#old_field_ty>
            >
        }
    };

    old_field_ty
}
