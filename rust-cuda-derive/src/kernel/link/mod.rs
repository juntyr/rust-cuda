use std::{
    env, fs,
    io::{Read, Write},
    path::{Path, PathBuf},
    sync::atomic::{AtomicBool, Ordering},
};

use colored::Colorize;
use proc_macro::TokenStream;
use ptx_builder::{
    builder::{BuildStatus, Builder, MessageFormat, Profile},
    error::{BuildErrorKind, Error, Result},
};

use super::utils::skip_kernel_compilation;

mod config;
mod error;

use config::{CheckKernelConfig, LinkKernelConfig};
use error::emit_ptx_build_error;

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

    let kernel_ptx = compile_kernel(&args, &crate_name, &crate_path, Specialisation::Check);

    match kernel_ptx {
        Some(kernel_ptx) => quote!(#kernel_ptx).into(),
        None => quote!("ERROR in this PTX compilation").into(),
    }
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

    let Some(mut kernel_ptx) = compile_kernel(
        &args,
        &crate_name,
        &crate_path,
        Specialisation::Link(&specialisation),
    ) else {
        return (quote! {
            const PTX_STR: &'static str = "ERROR in this PTX compilation";
        })
        .into()
    };

    let kernel_layout_name = if specialisation.is_empty() {
        format!("{kernel}_type_layout_kernel")
    } else {
        format!(
            "{kernel}_type_layout_kernel_{:016x}",
            seahash::hash(specialisation.as_bytes())
        )
    };

    let mut type_layouts = Vec::new();

    if let Some(start) = kernel_ptx.find(&format!("\n\t// .globl\t{kernel_layout_name}")) {
        let Some(middle) = kernel_ptx[start..].find(&format!(".visible .entry {kernel_layout_name}")) else {
            abort_call_site!(
                "Kernel compilation generated invalid PTX: incomplete type layout information."
            )
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

        let Some(stop) = kernel_ptx[(start + middle)..].find('}') else {
            abort_call_site!("Kernel compilation generated invalid PTX")
        };

        kernel_ptx.replace_range(start..(start + middle + stop + '}'.len_utf8()), "");
    }

    (quote! { const PTX_STR: &'static str = #kernel_ptx; #(#type_layouts)* }).into()
}

fn compile_kernel(
    args: &syn::Ident,
    crate_name: &str,
    crate_path: &Path,
    specialisation: Specialisation,
) -> Option<String> {
    if let Ok(rust_flags) = proc_macro::tracked_env::var("RUSTFLAGS") {
        env::set_var(
            "RUSTFLAGS",
            rust_flags
                .replace("-Zinstrument-coverage", "")
                .replace("-Cinstrument-coverage", ""),
        );
    }

    let specialisation_var = format!(
        "RUST_CUDA_DERIVE_SPECIALISE_{}_{}",
        crate_name,
        args.to_string().to_uppercase()
    );

    if let Ok(kernel_path) =
        build_kernel_with_specialisation(crate_path, &specialisation_var, specialisation)
    {
        let mut file = fs::File::open(&kernel_path)
            .unwrap_or_else(|_| panic!("Failed to open kernel file at {:?}.", &kernel_path));

        let mut kernel_ptx = String::new();

        file.read_to_string(&mut kernel_ptx)
            .unwrap_or_else(|_| panic!("Failed to read kernel file at {:?}.", &kernel_path));

        colored::control::set_override(true);
        eprintln!(
            "{} {} compiling a PTX crate.",
            "[PTX]".bright_black().bold(),
            "Finished".green().bold()
        );
        colored::control::unset_override();

        Some(kernel_ptx)
    } else {
        emit_ptx_build_error();

        None
    }
}

#[allow(clippy::too_many_lines)]
fn build_kernel_with_specialisation(
    kernel_path: &Path,
    env_var: &str,
    specialisation: Specialisation,
) -> Result<PathBuf> {
    match specialisation {
        Specialisation::Check => env::set_var(env_var, "chECK"),
        Specialisation::Link(specialisation) => env::set_var(env_var, specialisation),
    };

    let result = (|| {
        let mut builder = Builder::new(kernel_path)?;

        builder = match specialisation {
            Specialisation::Check => builder.set_profile(Profile::Debug),
            Specialisation::Link(_) => builder.set_profile(Profile::Release),
        };

        builder = builder.set_message_format(MessageFormat::Json {
            render_diagnostics: false,
            short: false,
            ansi: true,
        });

        let specialisation_prefix = match specialisation {
            Specialisation::Check => String::from("chECK"),
            Specialisation::Link(specialisation) => {
                format!("{:016x}", seahash::hash(specialisation.as_bytes()))
            },
        };
        builder = builder.set_prefix(specialisation_prefix.clone());

        let any_output = AtomicBool::new(false);
        let crate_name = String::from(builder.get_crate_name());

        match builder.build_live(
            |stdout_line| {
                if let Ok(cargo_metadata::Message::CompilerMessage(mut message)) =
                    serde_json::from_str(stdout_line)
                {
                    if any_output
                        .compare_exchange(false, true, Ordering::Relaxed, Ordering::Relaxed)
                        .is_ok()
                    {
                        colored::control::set_override(true);
                        eprintln!(
                            "{} of {} ({})",
                            "[PTX]".bright_black().bold(),
                            crate_name.bold(),
                            specialisation_prefix.to_ascii_lowercase(),
                        );
                        colored::control::unset_override();
                    }

                    if let Some(rendered) = &mut message.message.rendered {
                        colored::control::set_override(true);
                        let prefix = "  | ".bright_black().bold().to_string();
                        colored::control::unset_override();

                        let glue = String::from('\n') + &prefix;

                        let mut lines = rendered
                            .split('\n')
                            .rev()
                            .skip_while(|l| l.trim().is_empty())
                            .collect::<Vec<_>>();
                        lines.reverse();

                        let mut prefixed = prefix + &lines.join(&glue);

                        std::mem::swap(rendered, &mut prefixed);
                    }

                    eprintln!("{}", serde_json::to_string(&message.message).unwrap());
                }
            },
            |stderr_line| {
                if stderr_line.trim().is_empty() {
                    return;
                }

                if any_output
                    .compare_exchange(false, true, Ordering::Relaxed, Ordering::Relaxed)
                    .is_ok()
                {
                    colored::control::set_override(true);
                    eprintln!(
                        "{} of {} ({})",
                        "[PTX]".bright_black().bold(),
                        crate_name.bold(),
                        specialisation_prefix.to_ascii_lowercase(),
                    );
                    colored::control::unset_override();
                }

                colored::control::set_override(true);
                eprintln!(
                    "  {} {}",
                    "|".bright_black().bold(),
                    stderr_line.replace("   ", "")
                );
                colored::control::unset_override();
            },
        )? {
            BuildStatus::Success(output) => {
                let ptx_path = output.get_assembly_path();

                let mut specialised_ptx_path = ptx_path.clone();

                specialised_ptx_path.set_extension(format!("{specialisation_prefix}.ptx"));

                fs::copy(&ptx_path, &specialised_ptx_path).map_err(|err| {
                    Error::from(BuildErrorKind::BuildFailed(vec![format!(
                        "Failed to copy kernel from {ptx_path:?} to {specialised_ptx_path:?}: \
                         {err}"
                    )]))
                })?;

                if let Specialisation::Link(specialisation) = specialisation {
                    fs::OpenOptions::new()
                        .append(true)
                        .open(&specialised_ptx_path)
                        .and_then(|mut file| writeln!(file, "\n// {specialisation}"))
                        .map_err(|err| {
                            Error::from(BuildErrorKind::BuildFailed(vec![format!(
                                "Failed to write specialisation to {specialised_ptx_path:?}: {err}"
                            )]))
                        })?;
                }

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

#[derive(Copy, Clone, Debug)]
enum Specialisation<'a> {
    Check,
    Link(&'a str),
}
