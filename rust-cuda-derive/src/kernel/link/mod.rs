use std::{
    env, fs,
    io::{Read, Write},
    path::{Path, PathBuf},
};

use proc_macro::TokenStream;
use ptx_builder::{
    builder::{BuildStatus, Builder, ErrorFormat},
    error::{BuildErrorKind, Error, Result},
    reporter::ErrorLogPrinter,
};

use super::utils::skip_kernel_compilation;

mod config;
use config::{CheckKernelConfig, LinkKernelConfig};

pub fn check_kernel(tokens: TokenStream) -> TokenStream {
    proc_macro_error::set_dummy(quote! {
        "ERROR in this PTX compilation"
    });

    let CheckKernelConfig {
        args,
        crate_name,
        crate_path,
    } = match syn::parse_macro_input::parse(tokens) {
        Ok(config) => config,
        Err(err) => {
            abort_call_site!(
                "check_kernel!(ARGS NAME PATH) expects ARGS identifier, NAME and PATH string \
                 literals: {:?}",
                err
            )
        },
    };

    let kernel_ptx = compile_kernel(&args, &crate_name, &crate_path, None);

    quote!(#kernel_ptx).into()
}

lazy_static::lazy_static! {
    pub static ref CONST_LAYOUT_REGEX: regex::Regex = {
        regex::Regex::new(r"(?m)^\.global \.align 1 \.b8 (?P<param>[A-Z_0-9]+)\[(?P<len>\d+)\] = \{(?P<bytes>\d+(?:, \d+)*)\};$").unwrap()
    };
}

#[allow(clippy::module_name_repetitions)]
pub fn link_kernel(tokens: TokenStream) -> TokenStream {
    proc_macro_error::set_dummy(quote! {
        const PTX_STR: &'static str = "ERROR in this PTX compilation";
    });

    let LinkKernelConfig {
        kernel,
        args,
        crate_name,
        crate_path,
        specialisation,
    } = match syn::parse_macro_input::parse(tokens) {
        Ok(config) => config,
        Err(err) => {
            abort_call_site!(
                "link_kernel!(KERNEL ARGS NAME PATH SPECIALISATION) expects KERNEL and ARGS \
                 identifiers, NAME and PATH string literals, and SPECIALISATION tokens: {:?}",
                err
            )
        },
    };

    if skip_kernel_compilation() {
        return quote! {
            const PTX_STR: &'static str = "CLIPPY skips specialised PTX compilation";
        }
        .into();
    }

    let mut kernel_ptx = compile_kernel(&args, &crate_name, &crate_path, Some(&specialisation));

    let kernel_layout_name = if specialisation.is_empty() {
        format!("{}_type_layout_kernel", kernel)
    } else {
        format!(
            "{}_type_layout_kernel_{:016x}",
            kernel,
            seahash::hash(specialisation.as_bytes())
        )
    };

    let mut type_layouts = Vec::new();

    if let Some(start) = kernel_ptx.find(&format!("\n\t// .globl\t{}", kernel_layout_name)) {
        let middle =
            match kernel_ptx[start..].find(&format!(".visible .entry {}", kernel_layout_name)) {
                Some(middle) => middle,
                None => abort_call_site!(
                    "Kernel compilation generated invalid PTX: incomplete type layout information."
                ),
            };

        for capture in CONST_LAYOUT_REGEX.captures_iter(&kernel_ptx[start..(start + middle)]) {
            match (
                capture.name("param"),
                capture.name("len"),
                capture.name("bytes"),
            ) {
                (Some(param), Some(len), Some(bytes)) => {
                    let param = quote::format_ident!("{}", param.as_str());

                    let len = match len.as_str().parse::<usize>() {
                        Ok(len) => len,
                        Err(err) => {
                            abort_call_site!("Kernel compilation generated invalid PTX: {}", err)
                        },
                    };

                    let bytes: Vec<u8> = match bytes
                        .as_str()
                        .split(", ")
                        .map(std::str::FromStr::from_str)
                        .collect()
                    {
                        Ok(len) => len,
                        Err(err) => {
                            abort_call_site!("Kernel compilation generated invalid PTX: {}", err)
                        },
                    };
                    let byte_str = syn::LitByteStr::new(&bytes, proc_macro2::Span::call_site());

                    type_layouts.push(quote! {
                        const #param: &[u8; #len] = #byte_str;
                    });
                },
                _ => abort_call_site!(
                    "Kernel compilation generated invalid PTX: invalid type layout."
                ),
            };
        }

        let stop = match kernel_ptx[(start + middle)..].find('}') {
            Some(stop) => stop,
            None => abort_call_site!("Kernel compilation generated invalid PTX"),
        };

        kernel_ptx.replace_range(start..(start + middle + stop + '}'.len_utf8()), "");
    }

    (quote! { const PTX_STR: &'static str = #kernel_ptx; #(#type_layouts)* }).into()
}

fn compile_kernel(
    args: &syn::Ident,
    crate_name: &str,
    crate_path: &Path,
    specialisation: Option<&str>,
) -> String {
    // Force this proc macro to always be evaluated using a dummy env var
    env::set_var("RUST_CUDA_DERIVE_MARKER", "42");
    assert_eq!(
        proc_macro::tracked_env::var("RUST_CUDA_DERIVE_MARKER").as_deref(),
        Ok("42")
    );

    let colourful = if let Some(cargo_colour) = with_cargo_args(|args| {
        let mut colourful = None;

        for arg in args.windows(2) {
            colourful = Some(match (arg[0].as_str(), arg[1].as_str()) {
                ("--color=always", _) | ("--color", "always") | (_, "--color=always") => true,
                ("--color=never", _) | ("--color", "never") | (_, "--color=never") => false,
                _ => continue,
            });
        }

        colourful
    }) {
        cargo_colour
    } else if let Ok(cargo_colour @ ("always" | "never")) =
        proc_macro::tracked_env::var("CARGO_TERM_COLOR").as_deref()
    {
        cargo_colour == "always"
    } else if let Ok(term) = proc_macro::tracked_env::var("TERM") {
        term != "dumb"
            && matches!(
                proc_macro::tracked_env::var("NO_COLOR"),
                Err(env::VarError::NotPresent)
            )
    } else {
        false
    };

    if let Ok(rust_flags) = proc_macro::tracked_env::var("RUSTFLAGS") {
        env::set_var("RUSTFLAGS", rust_flags.replace("-Zinstrument-coverage", ""));
    }

    let specialisation_var = format!(
        "RUST_CUDA_DERIVE_SPECIALISE_{}_{}",
        crate_name,
        args.to_string().to_uppercase()
    );

    match build_kernel_with_specialisation(
        crate_path,
        &specialisation_var,
        specialisation,
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
    } else {
        env::remove_var(env_var);
    }

    let result = (|| {
        let mut builder = Builder::new(kernel_path)?;

        if !colourful {
            builder = builder.disable_colors();
        }

        builder = builder.set_error_format(
            with_cargo_args(|args| {
                let mut message_format = None;

                for arg in args.windows(2) {
                    message_format = Some(match (arg[0].as_str(), arg[1].as_str()) {
                        ("--message-format=human", _)
                        | ("--message-format", "human")
                        | (_, "--message-format=human") => ErrorFormat::Human,
                        ("--message-format=json", _)
                        | ("--message-format", "json")
                        | (_, "--message-format=json") => ErrorFormat::JSON,
                        ("--message-format=short", _)
                        | ("--message-format", "short")
                        | (_, "--message-format=short") => ErrorFormat::Short,
                        _ => continue,
                    });
                }

                message_format
            })
            .unwrap_or(ptx_builder::builder::ErrorFormat::Human),
        );

        match builder.build()? {
            BuildStatus::Success(output) => {
                let ptx_path = output.get_assembly_path();

                let specialisation = match specialisation {
                    None => return Ok(ptx_path),
                    Some(specialisation) => specialisation,
                };

                let mut specialised_ptx_path = ptx_path.clone();
                specialised_ptx_path.set_extension(&format!(
                    "{:016x}.ptx",
                    seahash::hash(specialisation.as_bytes())
                ));

                fs::copy(&ptx_path, &specialised_ptx_path).map_err(|err| {
                    Error::from(BuildErrorKind::BuildFailed(vec![format!(
                        "Failed to copy kernel from {:?} to {:?}: {}",
                        ptx_path, specialised_ptx_path, err,
                    )]))
                })?;

                fs::OpenOptions::new()
                    .append(true)
                    .open(&specialised_ptx_path)
                    .and_then(|mut file| writeln!(file, "\n// {}", specialisation))
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
    })();

    env::remove_var(env_var);

    result
}

fn with_cargo_args<Q, F: FnOnce(&[String]) -> Option<Q>>(inner: F) -> Option<Q> {
    use sysinfo::{ProcessExt, RefreshKind, System, SystemExt};

    let mut system = System::new_with_specifics(RefreshKind::new());

    let pid = sysinfo::get_current_pid().ok()?;
    system.refresh_process(pid);
    let process = system.process(pid)?;

    let p_pid = process.parent()?;
    system.refresh_process(p_pid);
    let parent = system.process(p_pid)?;

    inner(parent.cmd())
}
