use syn::{parse_quote, spanned::Spanned};

#[allow(clippy::module_name_repetitions)]
pub enum CudaReprFieldTy {
    Embedded(Box<syn::Type>),
}

pub fn swap_field_type_and_get_cuda_repr_ty(field: &mut syn::Field) -> Option<CudaReprFieldTy> {
    let mut cuda_repr_field_ty: Option<CudaReprFieldTy> = None;
    let mut field_ty = field.ty.clone();

    // Helper attribute `r2c` must be filtered out inside cuda representation
    field.attrs.retain(|attr| {
        if attr.path.is_ident("r2cEmbed") {
            if cuda_repr_field_ty.is_none() {
                if !attr.tokens.is_empty() {
                    emit_error!(
                        attr.tokens.span(),
                        "#[r2cEmbed] does not take any arguments."
                    );
                }

                cuda_repr_field_ty = Some(CudaReprFieldTy::Embedded(Box::new(field_ty.clone())));

                field_ty = parse_quote! {
                    <#field_ty as rust_cuda::common::RustToCuda>::CudaRepresentation
                };
            } else {
                emit_error!(attr.span(), "Duplicate #[r2cEmbed] attribute definition.");
            }

            false
        } else {
            true
        }
    });

    field.ty = field_ty;

    cuda_repr_field_ty
}
