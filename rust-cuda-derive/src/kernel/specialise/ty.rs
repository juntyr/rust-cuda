use proc_macro::TokenStream;
use quote::ToTokens;

pub fn specialise_kernel_type(tokens: TokenStream) -> TokenStream {
    let SpecialiseTypeConfig {
        _private, // TODO: either use or remove the private path
        args,
        typedef,
    } = match syn::parse_macro_input::parse(tokens) {
        Ok(config) => config,
        Err(err) => {
            abort_call_site!(
                "specialise_kernel_type!(ARGS::TYPEDEF) expects ARGS path and TYPEDEF identifier: \
                 {:?}",
                err
            )
        },
    };

    let crate_name = match proc_macro::tracked_env::var("CARGO_CRATE_NAME") {
        Ok(crate_name) => crate_name.to_uppercase(),
        Err(err) => abort_call_site!("Failed to read crate name: {:?}", err),
    };

    let specialisation_var = format!(
        "RUST_CUDA_DERIVE_SPECIALISE_{}_{}",
        crate_name,
        args.to_string().to_uppercase()
    );

    match proc_macro::tracked_env::var(&specialisation_var) {
        Ok(specialisation) => {
            let specialisation = match syn::parse_str(&specialisation) {
                _ if specialisation.is_empty() => syn::PathArguments::None,
                Ok(specialisation) => syn::PathArguments::AngleBracketed(specialisation),
                Err(err) => abort_call_site!("Failed to parse specialisation: {:?}", err),
            };

            syn::Type::Path(syn::TypePath {
                qself: Some(syn::QSelf {
                    lt_token: syn::parse_quote!(<),
                    ty: syn::parse_quote!(()),
                    position: 1, // 2,
                    as_token: syn::parse_quote!(as),
                    gt_token: syn::parse_quote!(>),
                }),
                path: syn::Path {
                    leading_colon: None,
                    segments: [
                        // syn::PathSegment {
                        //     ident: private,
                        //     arguments: syn::PathArguments::None,
                        // },
                        syn::PathSegment {
                            ident: args,
                            arguments: specialisation,
                        },
                        syn::PathSegment {
                            ident: typedef,
                            arguments: syn::PathArguments::None,
                        },
                    ]
                    .into_iter()
                    .collect(),
                },
            })
            .into_token_stream()
            .into()
        },
        Err(err) => abort_call_site!(
            "Failed to read specialisation from {:?}: {:?}",
            &specialisation_var,
            err
        ),
    }
}

struct SpecialiseTypeConfig {
    _private: syn::Ident,
    args: syn::Ident,
    typedef: syn::Ident,
}

impl syn::parse::Parse for SpecialiseTypeConfig {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let private: syn::Ident = input.parse()?;
        let _dc: syn::token::Colon2 = input.parse()?;
        let args: syn::Ident = input.parse()?;
        let _dc: syn::token::Colon2 = input.parse()?;
        let typedef: syn::Ident = input.parse()?;

        Ok(Self {
            _private: private,
            args,
            typedef,
        })
    }
}
