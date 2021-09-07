use std::{
    env, fs,
    io::{Read, Write},
    path::{Path, PathBuf},
};

use proc_macro::TokenStream;
use ptx_builder::{
    builder::{BuildStatus, Builder},
    error::{BuildErrorKind, Error, Result},
    reporter::ErrorLogPrinter,
};

use super::utils::skip_kernel_compilation;

mod config;
use config::{CheckKernelConfig, LinkKernelConfig};

pub fn check_kernel(tokens: TokenStream) -> TokenStream {
    let CheckKernelConfig {
        kernel,
        crate_name,
        crate_path,
    } = match syn::parse_macro_input::parse(tokens) {
        Ok(config) => config,
        Err(err) => {
            abort_call_site!(
                "check_kernel!(KERNEL NAME PATH) expects KERNEL identifier, NAME and PATH string \
                 literals: {:?}",
                err
            )
        },
    };

    let _kernel_ptx = compile_kernel(&kernel, &crate_name, &crate_path, None);

    TokenStream::new()
}

#[allow(clippy::module_name_repetitions)]
pub fn link_kernel(tokens: TokenStream) -> TokenStream {
    proc_macro_error::set_dummy(quote! {
        "ERROR in this PTX compilation"
    });

    let LinkKernelConfig {
        kernel,
        crate_name,
        crate_path,
        specialisation,
    } = match syn::parse_macro_input::parse(tokens) {
        Ok(config) => config,
        Err(err) => {
            abort_call_site!(
                "link_kernel!(KERNEL NAME PATH SPECIALISATION) expects KERNEL identifier, NAME \
                 and PATH string literals, and SPECIALISATION tokens: {:?}",
                err
            )
        },
    };

    let kernel_ptx = compile_kernel(&kernel, &crate_name, &crate_path, Some(&specialisation));

    (quote! { #kernel_ptx }).into()
}

fn compile_kernel(
    kernel: &syn::Ident,
    crate_name: &str,
    crate_path: &Path,
    specialisation: Option<&str>,
) -> String {
    let colourful = match std::env::var_os("TERM") {
        None => false,
        Some(term) if term == "dumb" => false,
        Some(_) => std::env::var_os("NO_COLOR").is_none(),
    };

    if let Ok(rust_flags) = proc_macro::tracked_env::var("RUSTFLAGS") {
        env::set_var("RUSTFLAGS", rust_flags.replace("-Zinstrument-coverage", ""));
    }

    let specialisation_var = format!(
        "RUST_CUDA_DERIVE_SPECIALISE_{}_{}",
        crate_name,
        kernel.to_string().to_uppercase()
    );

    match build_kernel_with_specialisation(
        crate_path,
        &specialisation_var,
        if skip_kernel_compilation() {
            None
        } else {
            specialisation
        },
        colourful,
    ) {
        Ok(kernel_path) => {
            let mut file = fs::File::open(&kernel_path)
                .unwrap_or_else(|_| panic!("Failed to open kernel file at {:?}.", &kernel_path));

            let mut kernel_ptx = String::new();

            file.read_to_string(&mut kernel_ptx)
                .unwrap_or_else(|_| panic!("Failed to read kernel file at {:?}.", &kernel_path));

            kernel_ptx
        },
        Err(err) => {
            let mut error_printer = ErrorLogPrinter::print(err);

            if !colourful {
                error_printer.disable_colors();
            }

            abort_call_site!(error_printer);
        },
    }
}

fn build_kernel_with_specialisation(
    kernel_path: &Path,
    env_var: &str,
    specialisation: Option<&str>,
    colourful: bool,
) -> Result<PathBuf> {
    if let Some(specialisation) = specialisation {
        env::set_var(env_var, specialisation);
    }

    let mut builder = Builder::new(kernel_path)?;

    if !colourful {
        builder = builder.disable_colors();
        builder = builder.set_error_format(ptx_builder::builder::ErrorFormat::JSON);
    }

    match builder.build()? {
        BuildStatus::Success(output) => {
            let ptx_path = output.get_assembly_path();

            let mut specialised_ptx_path = ptx_path.clone();
            if let Some(specialisation) = specialisation {
                specialised_ptx_path.set_extension(&format!(
                    "{:016x}.ptx",
                    seahash::hash(specialisation.as_bytes())
                ));
            }

            fs::copy(&ptx_path, &specialised_ptx_path).map_err(|err| {
                Error::from(BuildErrorKind::BuildFailed(vec![format!(
                    "Failed to copy kernel from {:?} to {:?}: {}",
                    ptx_path, specialised_ptx_path, err,
                )]))
            })?;

            fs::OpenOptions::new()
                .append(true)
                .open(&specialised_ptx_path)
                .and_then(|mut file| {
                    if let Some(specialisation) = specialisation {
                        writeln!(file, "\n// {}", specialisation)
                    } else {
                        Ok(())
                    }
                })
                .map_err(|err| {
                    Error::from(BuildErrorKind::BuildFailed(vec![format!(
                        "Failed to write specialisation to {:?}: {}",
                        specialised_ptx_path, err,
                    )]))
                })?;

            Ok(specialised_ptx_path)
        },
        BuildStatus::NotNeeded => Err(Error::from(BuildErrorKind::BuildFailed(vec![format!(
            "Kernel build for specialisation {:?} was not needed.",
            &specialisation
        )]))),
    }
}
