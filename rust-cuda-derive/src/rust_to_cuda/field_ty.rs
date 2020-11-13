use syn::parse_quote;

#[allow(clippy::module_name_repetitions)]
pub enum CudaReprFieldTy {
    BoxedSlice(proc_macro2::TokenStream),
    Embedded(Box<syn::Type>),
    Eval(proc_macro2::TokenStream),
}

pub fn swap_field_type_and_get_cuda_repr_ty(field: &mut syn::Field) -> Option<CudaReprFieldTy> {
    let mut cuda_repr_field_ty: Option<CudaReprFieldTy> = None;
    let mut field_ty = field.ty.clone();

    // Helper attribute `r2c` must be filtered out inside cuda representation
    field.attrs.retain(|attr| match attr.path.get_ident() {
        Some(ident) if cuda_repr_field_ty.is_none() && format!("{}", ident) == "r2cEmbed" => {
            // Allow the shorthand `#[r2c]` which uses the field type
            // as well as the explicit `#[r2c(ty)]` which overwrites the type
            let attribute_str = if attr.tokens.is_empty() {
                format!("({})", quote! { #field_ty })
            } else {
                format!("{}", attr.tokens)
            };

            if let Some(slice_type) = attribute_str
                .strip_prefix("(Box < [")
                .and_then(|rest| rest.strip_suffix("] >)"))
            {
                // Check for the special case of a boxed slice: `Box<ty>`
                let slice_type = slice_type.parse().unwrap();

                field_ty = parse_quote! {
                    (rustacuda_core::DevicePointer<#slice_type>, usize)
                };

                cuda_repr_field_ty = Some(CudaReprFieldTy::BoxedSlice(slice_type));
            } else if let Some(struct_type) = attribute_str
                .strip_prefix("(")
                .and_then(|rest| rest.strip_suffix(")"))
            {
                // Check for the case where a type implementing is `RustToCuda` embedded
                let field_type = syn::parse_str(struct_type).unwrap();

                field_ty = parse_quote! {
                    <#field_type as rust_cuda::common::RustToCuda>::CudaRepresentation
                };

                cuda_repr_field_ty = Some(CudaReprFieldTy::Embedded(Box::new(field_type)));
            }

            false
        },
        Some(ident) if cuda_repr_field_ty.is_none() && format!("{}", ident) == "r2cEval" => {
            cuda_repr_field_ty = Some(CudaReprFieldTy::Eval(attr.tokens.clone()));

            false
        },
        _ => false,
    });

    field.ty = field_ty;

    cuda_repr_field_ty
}
