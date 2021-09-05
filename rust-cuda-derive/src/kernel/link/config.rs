use std::path::PathBuf;

#[allow(clippy::module_name_repetitions)]
pub(super) struct LinkKernelConfig {
    pub(super) kernel: syn::Ident,
    pub(super) crate_name: String,
    pub(super) crate_path: PathBuf,
    pub(super) specialisation: Option<String>,
}

impl syn::parse::Parse for LinkKernelConfig {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let kernel: syn::Ident = input.parse()?;
        let name: syn::LitStr = input.parse()?;
        let path: syn::LitStr = input.parse()?;

        let specialisation = if input.parse::<Option<syn::token::Lt>>()?.is_some() {
            if input.parse::<Option<syn::token::Gt>>()?.is_some() {
                None
            } else {
                let specialisation_types = syn::punctuated::Punctuated::<
                    syn::Type,
                    syn::token::Comma,
                >::parse_separated_nonempty(input)?;

                let _gt_token: syn::token::Gt = input.parse()?;

                Some(
                    (quote! { <#specialisation_types> })
                        .to_string()
                        .replace(&[' ', '\n', '\t'][..], ""),
                )
            }
        } else {
            None
        };

        Ok(Self {
            kernel,
            crate_name: name.value(),
            crate_path: PathBuf::from(path.value()),
            specialisation,
        })
    }
}
