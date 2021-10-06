use proc_macro::TokenStream;

#[allow(clippy::module_name_repetitions)]
pub fn specialise_kernel_call(tokens: TokenStream) -> TokenStream {
    let SpecialiseMangleConfig {
        kernel,
        specialisation,
    } = match syn::parse_macro_input::parse(tokens) {
        Ok(config) => config,
        Err(err) => {
            abort_call_site!(
                "specialise_kernel_call!(KERNEL SPECIALISATION) expects KERNEL identifier and \
                 SPECIALISATION tokens: {:?}",
                err
            )
        },
    };

    let mangled_kernel_ident = if let Some(specialisation) = specialisation {
        format!(
            "{}_kernel_{:016x}",
            kernel,
            seahash::hash(specialisation.as_bytes())
        )
    } else {
        format!("{}_kernel", kernel)
    };

    (quote! { #mangled_kernel_ident }).into()
}

struct SpecialiseMangleConfig {
    kernel: syn::Ident,
    specialisation: Option<String>,
}

impl syn::parse::Parse for SpecialiseMangleConfig {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let kernel: syn::Ident = input.parse()?;

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
            specialisation,
        })
    }
}
