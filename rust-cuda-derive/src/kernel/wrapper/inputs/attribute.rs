use syn::spanned::Spanned;

pub(super) enum KernelInputAttribute {
    PtxJit(proc_macro2::Span, bool),
}

impl syn::parse::Parse for KernelInputAttribute {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let ident: syn::Ident = input.parse()?;

        match &*ident.to_string() {
            "jit" => {
                let eq: Option<syn::token::Eq> = input.parse()?;

                let (ptx_jit, span) = if eq.is_some() {
                    let value: syn::LitBool = input.parse()?;

                    (
                        value.value(),
                        ident
                            .span()
                            .join(eq.span())
                            .unwrap()
                            .span()
                            .join(value.span())
                            .unwrap(),
                    )
                } else {
                    (true, ident.span())
                };

                Ok(KernelInputAttribute::PtxJit(span, ptx_jit))
            },
            _ => abort!(
                ident.span(),
                "Unexpected kernel attribute `{:?}`: Expected `jit`.",
                ident
            ),
        }
    }
}

pub(super) struct KernelInputAttributes(Vec<KernelInputAttribute>);

impl syn::parse::Parse for KernelInputAttributes {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let content;
        let _parens = syn::parenthesized!(content in input);

        syn::punctuated::Punctuated::<
            KernelInputAttribute, syn::token::Comma
        >::parse_separated_nonempty(&content).map(|punctuated| {
            Self(punctuated.into_iter().collect())
        })
    }
}

impl IntoIterator for KernelInputAttributes {
    type IntoIter = std::vec::IntoIter<Self::Item>;
    type Item = KernelInputAttribute;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}
