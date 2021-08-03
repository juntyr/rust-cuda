use proc_macro::TokenStream;
use syn::spanned::Spanned;

pub(super) fn parse_kernel_fn(tokens: TokenStream) -> syn::ItemFn {
    let func: syn::ItemFn = syn::parse(tokens).unwrap_or_else(|err| {
        abort_call_site!(
            "#[kernel(...)] must be wrapped around a function: {:?}",
            err
        )
    });

    if !matches!(func.vis, syn::Visibility::Public(_)) {
        abort!(func.vis.span(), "Kernel function must be public.");
    }

    if func.sig.constness.is_some() {
        abort!(
            func.sig.constness.span(),
            "Kernel function must not be const."
        );
    }

    if func.sig.asyncness.is_some() {
        abort!(
            func.sig.asyncness.span(),
            "Kernel function must not (yet) be async."
        );
    }

    if func.sig.abi.is_some() {
        abort!(
            func.sig.abi.span(),
            "Kernel function must not have an explicit ABI."
        );
    }

    if func.sig.variadic.is_some() {
        abort!(
            func.sig.variadic.span(),
            "Kernel function must not be variadic."
        );
    }

    match &func.sig.output {
        syn::ReturnType::Default => (),
        syn::ReturnType::Type(_, box syn::Type::Tuple(tuple)) if tuple.elems.is_empty() => (),
        syn::ReturnType::Type(_, non_unit_type) => abort!(
            non_unit_type.span(),
            "Kernel function must return the unit type."
        ),
    };

    func
}
