use proc_macro::TokenStream;

pub fn specialise_kernel_type(tokens: TokenStream) -> TokenStream {
    let SpecialiseTypeConfig { kernel, typedef } = match syn::parse_macro_input::parse(tokens) {
        Ok(config) => config,
        Err(err) => {
            abort_call_site!(
                "specialise_kernel_type!(KERNEL::TYPEDEF) expects KERNEL and TYPEDEF identifiers: \
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
        kernel.to_string().to_uppercase()
    );

    match proc_macro::tracked_env::var(&specialisation_var) {
        Ok(specialisation) => {
            match format!("<() as {}{}>::{}", kernel, specialisation, typedef).parse() {
                Ok(parsed_specialisation) => parsed_specialisation,
                Err(err) => abort_call_site!("Failed to parse specialisation: {:?}", err),
            }
        },
        Err(err) => abort_call_site!(
            "Failed to read specialisation from {:?}: {:?}",
            &specialisation_var,
            err
        ),
    }
}

struct SpecialiseTypeConfig {
    kernel: syn::Ident,
    typedef: syn::Ident,
}

impl syn::parse::Parse for SpecialiseTypeConfig {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let kernel: syn::Ident = input.parse()?;
        let _dc: syn::token::Colon2 = input.parse()?;
        let typedef: syn::Ident = input.parse()?;

        Ok(Self { kernel, typedef })
    }
}
