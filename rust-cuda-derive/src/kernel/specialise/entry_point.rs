use std::ffi::CStr;

use proc_macro::TokenStream;

#[allow(clippy::module_name_repetitions)]
pub fn specialise_kernel_entry_point(tokens: TokenStream) -> TokenStream {
    let SpecialiseMangleConfig {
        kernel,
        specialisation,
    } = match syn::parse_macro_input::parse(tokens) {
        Ok(config) => config,
        Err(err) => {
            abort_call_site!(
                "specialise_kernel_entry_point!(KERNEL SPECIALISATION) expects KERNEL identifier \
                 and SPECIALISATION tokens: {:?}",
                err
            )
        },
    };

    let mangled_kernel_ident = if let Some(specialisation) = specialisation {
        format!(
            "{kernel}_kernel_{:016x}",
            seahash::hash(specialisation.as_bytes())
        )
    } else {
        format!("{kernel}_kernel")
    };

    let mut mangled_kernel_ident = mangled_kernel_ident.into_bytes();
    mangled_kernel_ident.push(b'\0');

    if let Err(err) = CStr::from_bytes_with_nul(&mangled_kernel_ident) {
        abort_call_site!(
            "Kernel compilation generated invalid kernel entry point: internal nul byte: {:?}",
            err
        );
    }

    // TODO: CStr constructor blocked on https://github.com/rust-lang/rust/issues/118560
    let mangled_kernel_ident =
        syn::LitByteStr::new(&mangled_kernel_ident, proc_macro2::Span::call_site());
    // Safety: the validity of mangled_kernel_ident as a CStr was just checked above
    let mangled_kernel_ident = quote! { unsafe { ::core::ffi::CStr::from_bytes_with_nul_unchecked(#mangled_kernel_ident) } };

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
