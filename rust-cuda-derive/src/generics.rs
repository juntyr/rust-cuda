pub fn expand_cuda_struct_generics_where_requested_in_attrs(
    ast: &syn::DeriveInput,
) -> (Vec<syn::Attribute>, syn::Generics) {
    let mut struct_attrs_cuda = ast.attrs.clone();
    let mut struct_generics_cuda = ast.generics.clone();

    struct_attrs_cuda.retain(|attr| match attr.path.get_ident() {
        Some(ident) if format!("{}", ident) == "r2cBound" => {
            let attribute_str = format!("{}", attr.tokens);

            if let Some(type_trait_bound) = attribute_str
                .strip_prefix("(")
                .and_then(|rest| rest.strip_suffix(")"))
            {
                let type_param: syn::TypeParam = syn::parse_str(type_trait_bound).unwrap();

                let mut type_param_has_been_inserted = false;

                // Append the additional trait bounds if the generic type is already bounded
                if let Some(matching_param) = struct_generics_cuda
                    .type_params_mut()
                    .find(|tp| tp.ident == type_param.ident)
                {
                    for bound in &type_param.bounds {
                        matching_param.bounds.push(bound.clone())
                    }

                    type_param_has_been_inserted = true;
                }

                if !type_param_has_been_inserted {
                    struct_generics_cuda
                        .params
                        .push(syn::GenericParam::Type(type_param));
                }
            }

            false
        },
        _ => true,
    });

    (struct_attrs_cuda, struct_generics_cuda)
}
