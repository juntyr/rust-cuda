use std::env::VarError;

use proc_macro::TokenStream;
use quote::quote;

#[allow(clippy::module_name_repetitions)]
pub fn specialise_kernel_function(attr: TokenStream, func: TokenStream) -> TokenStream {
    let mut func: syn::ItemFn = syn::parse(func).unwrap_or_else(|err| {
        abort_call_site!(
            "#[specialise_kernel_function(...)] must be wrapped around a function: {:?}",
            err
        )
    });

    let kernel: syn::Ident = match syn::parse_macro_input::parse(attr) {
        Ok(kernel) => kernel,
        Err(err) => abort_call_site!(
            "#[specialise_kernel_function(KERNEL)] expects KERNEL identifier: {:?}",
            err
        ),
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

    func.sig.ident = match proc_macro::tracked_env::var(&specialisation_var).as_deref() {
        Ok("") => quote::format_ident!("{}_kernel", func.sig.ident),
        Ok("chECK") => {
            let func_ident = quote::format_ident!("{}_chECK", func.sig.ident);

            return (quote! {
                #[cfg(target_os = "cuda")]
                #[no_mangle]
                pub unsafe extern "ptx-kernel" fn #func_ident() {}
            })
            .into();
        },
        Ok(specialisation) => {
            quote::format_ident!(
                "{}_kernel_{:016x}",
                func.sig.ident,
                seahash::hash(specialisation.as_bytes())
            )
        },
        Err(err @ VarError::NotUnicode(_)) => abort_call_site!(
            "Failed to read specialisation from {:?}: {:?}",
            &specialisation_var,
            err
        ),
        Err(VarError::NotPresent) => return quote!().into(),
    };

    (quote! { #func }).into()
}
