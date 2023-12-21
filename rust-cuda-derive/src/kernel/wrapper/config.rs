pub(super) struct KernelConfig {
    pub(super) visibility: Option<syn::token::Pub>,
    pub(super) linker: syn::Ident,
}

impl syn::parse::Parse for KernelConfig {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let visibility: Option<syn::token::Pub> = input.parse()?;
        let _use: syn::token::Use = input.parse()?;
        let linker: syn::Ident = input.parse()?;
        let _bang: syn::token::Bang = input.parse()?;
        let _for: syn::token::For = input.parse()?;
        let _impl: syn::token::Impl = input.parse()?;

        Ok(Self { visibility, linker })
    }
}
