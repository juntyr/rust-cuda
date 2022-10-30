use syn::spanned::Spanned;

pub fn expand_cuda_struct_generics_where_requested_in_attrs(
    ast: &syn::DeriveInput,
) -> (Vec<syn::Attribute>, syn::Generics, Vec<syn::Attribute>) {
    let mut struct_attrs_cuda = ast.attrs.clone();
    let mut struct_generics_cuda = ast.generics.clone();
    let mut struct_layout_attrs = Vec::new();

    let mut r2c_ignore = false;

    struct_attrs_cuda.retain(|attr| {
        if attr.path.is_ident("r2cBound") {
            let type_param: syn::TypeParam = match attr.parse_args() {
                Ok(type_param) => type_param,
                Err(err) => {
                    emit_error!(err);

                    return false;
                },
            };

            let mut type_param_has_been_inserted = false;

            // Append the additional trait bounds if the generic type is already bounded
            if let Some(matching_param) = struct_generics_cuda
                .type_params_mut()
                .find(|tp| tp.ident == type_param.ident)
            {
                for bound in &type_param.bounds {
                    matching_param.bounds.push(bound.clone());
                }

                type_param_has_been_inserted = true;
            }

            if !type_param_has_been_inserted {
                struct_generics_cuda
                    .params
                    .push(syn::GenericParam::Type(type_param));
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
        } else if attr.path.is_ident("r2cLayout") {
            struct_layout_attrs.push(syn::Attribute {
                pound_token: attr.pound_token,
                style: attr.style,
                bracket_token: attr.bracket_token,
                path: proc_macro2::Ident::new("layout", attr.path.span()).into(),
                tokens: attr.tokens.clone(),
            });

            false
        } else {
            !r2c_ignore
        }
    });

    (struct_attrs_cuda, struct_generics_cuda, struct_layout_attrs)
}
