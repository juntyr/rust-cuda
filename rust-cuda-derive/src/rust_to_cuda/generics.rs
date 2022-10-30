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
            let bound: syn::WherePredicate = match attr.parse_args() {
                Ok(bound) => bound,
                Err(err) => {
                    emit_error!(err);

                    return false;
                },
            };

            struct_generics_cuda
                .make_where_clause()
                .predicates
                .push(bound);

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
