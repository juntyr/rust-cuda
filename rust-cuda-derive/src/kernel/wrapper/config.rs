pub(super) struct KernelConfig {
    pub(super) visibility: Option<syn::token::Pub>,
    pub(super) linker: syn::Ident,
    pub(super) kernel: syn::Ident,
    pub(super) args: syn::Ident,
    pub(super) launcher: syn::Ident,
}

impl syn::parse::Parse for KernelConfig {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let visibility: Option<syn::token::Pub> = input.parse()?;
        let _use: syn::token::Use = input.parse()?;
        let linker: syn::Ident = input.parse()?;
        let _bang: syn::token::Bang = input.parse()?;
        let _as: syn::token::As = input.parse()?;
        let _impl: syn::token::Impl = input.parse()?;
        let kernel: syn::Ident = input.parse()?;
        let _lt_token: syn::token::Lt = input.parse()?;
        let args: syn::Ident = input.parse()?;
        let _comma: Option<syn::token::Comma> = input.parse()?;
        let _gt_token: syn::token::Gt = input.parse()?;
        let _for: syn::token::For = input.parse()?;
        let launcher: syn::Ident = input.parse()?;

        Ok(Self {
            visibility,
            linker,
            kernel,
            args,
            launcher,
        })
    }
}
