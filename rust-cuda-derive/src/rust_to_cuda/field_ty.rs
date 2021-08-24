use syn::{parse_quote, spanned::Spanned};

pub fn swap_field_type_and_filter_attrs(field: &mut syn::Field) -> syn::Type {
    let cuda_repr_field_ty = field.ty.clone();

    field.ty = parse_quote! {
        rust_cuda::common::DeviceAccessible<
            <#cuda_repr_field_ty as rust_cuda::common::RustToCuda>::CudaRepresentation
        >
    };

    // Remove all field attributes after #[r2cIgnore]
    if let Some(ignore_from) = field.attrs.iter().position(|attr| {
        if attr.path.is_ident("r2cIgnore") {
            if !attr.tokens.is_empty() {
                emit_error!(
                    attr.tokens.span(),
                    "#[r2cIgnore] does not take any arguments."
                );
            }

            true
        } else {
            false
        }
    }) {
        field.attrs.truncate(ignore_from);
    }

    cuda_repr_field_ty
}
