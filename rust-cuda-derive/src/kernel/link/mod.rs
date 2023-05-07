use std::{
    env,
    ffi::CString,
    fmt::Write as FmtWrite,
    fs,
    io::{Read, Write},
    os::raw::c_int,
    path::{Path, PathBuf},
    ptr::addr_of_mut,
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
mod ptx_compiler_sys;

use config::{CheckKernelConfig, LinkKernelConfig};
use error::emit_ptx_build_error;
use ptx_compiler_sys::NvptxError;

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

#[allow(clippy::module_name_repetitions, clippy::too_many_lines)]
pub fn link_kernel(tokens: TokenStream) -> TokenStream {
    proc_macro_error::set_dummy(quote! {
        const PTX_STR: &'static str = "ERROR in this PTX compilation";
    });

    let LinkKernelConfig {
        kernel,
        kernel_hash,
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
        .into();
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

    let type_layout_start_pattern = format!("\n\t// .globl\t{kernel_layout_name}");

    if let Some(type_layout_start) = kernel_ptx.find(&type_layout_start_pattern) {
        const BEFORE_PARAM_PATTERN: &str = ".global .align 1 .b8 ";
        const PARAM_LEN_PATTERN: &str = "[";
        const LEN_BYTES_PATTERN: &str = "] = {";
        const AFTER_BYTES_PATTERN: &str = "};";

        let after_type_layout_start = type_layout_start + type_layout_start_pattern.len();

        let Some(type_layout_middle) = kernel_ptx[after_type_layout_start..]
            .find(&format!(".visible .entry {kernel_layout_name}"))
            .map(|i| after_type_layout_start + i)
        else {
            abort_call_site!(
                "Kernel compilation generated invalid PTX: incomplete type layout information"
            )
        };

        let mut next_type_layout = after_type_layout_start;

        while let Some(param_start_offset) =
            kernel_ptx[next_type_layout..type_layout_middle].find(BEFORE_PARAM_PATTERN)
        {
            let param_start = next_type_layout + param_start_offset + BEFORE_PARAM_PATTERN.len();

            if let Some(len_start_offset) =
                kernel_ptx[param_start..type_layout_middle].find(PARAM_LEN_PATTERN)
            {
                let len_start = param_start + len_start_offset + PARAM_LEN_PATTERN.len();

                if let Some(bytes_start_offset) =
                    kernel_ptx[len_start..type_layout_middle].find(LEN_BYTES_PATTERN)
                {
                    let bytes_start = len_start + bytes_start_offset + LEN_BYTES_PATTERN.len();

                    if let Some(bytes_end_offset) =
                        kernel_ptx[bytes_start..type_layout_middle].find(AFTER_BYTES_PATTERN)
                    {
                        let param = &kernel_ptx[param_start..(param_start + len_start_offset)];
                        let len = &kernel_ptx[len_start..(len_start + bytes_start_offset)];
                        let bytes = &kernel_ptx[bytes_start..(bytes_start + bytes_end_offset)];

                        let param = quote::format_ident!("{}", param);

                        let Ok(len) = len.parse::<usize>() else {
                            abort_call_site!(
                                "Kernel compilation generated invalid PTX: invalid type layout \
                                 length"
                            )
                        };
                        let Ok(bytes) = bytes
                            .split(", ")
                            .map(std::str::FromStr::from_str)
                            .collect::<Result<Vec<u8>, _>>()
                        else {
                            abort_call_site!(
                                "Kernel compilation generated invalid PTX: invalid type layout \
                                 byte"
                            )
                        };

                        if bytes.len() != len {
                            abort_call_site!(
                                "Kernel compilation generated invalid PTX: type layout length \
                                 mismatch"
                            );
                        }

                        let byte_str = syn::LitByteStr::new(&bytes, proc_macro2::Span::call_site());

                        type_layouts.push(quote! {
                            const #param: &[u8; #len] = #byte_str;
                        });

                        next_type_layout =
                            bytes_start + bytes_end_offset + AFTER_BYTES_PATTERN.len();
                    } else {
                        next_type_layout = bytes_start;
                    }
                } else {
                    next_type_layout = len_start;
                }
            } else {
                next_type_layout = param_start;
            }
        }

        let Some(type_layout_end) = kernel_ptx[type_layout_middle..]
            .find('}')
            .map(|i| type_layout_middle + i + '}'.len_utf8())
        else {
            abort_call_site!("Kernel compilation generated invalid PTX")
        };

        kernel_ptx.replace_range(type_layout_start..type_layout_end, "");
    }

    let (result, error_log, info_log, version, drop) =
        check_kernel_ptx(&kernel_ptx, &specialisation, &kernel_hash);

    let ptx_compiler = match &version {
        Ok((major, minor)) => format!("PTX compiler v{major}.{minor}"),
        Err(_) => String::from("PTX compiler"),
    };

    // TODO: allow user to select
    // - warn on double
    // - warn on float
    // - warn on spills
    // - verbose warn
    // - warnings as errors
    // - show PTX source if warning or error

    let mut errors = String::new();
    if let Err(err) = drop {
        let _ = errors.write_fmt(format_args!("Error dropping the {ptx_compiler}: {err}\n"));
    }
    if let Err(err) = version {
        let _ = errors.write_fmt(format_args!(
            "Error fetching the version of the {ptx_compiler}: {err}\n"
        ));
    }
    if let (Ok(Some(_)), _) | (_, Ok(Some(_))) = (&info_log, &error_log) {
        let mut max_lines = kernel_ptx.chars().filter(|c| *c == '\n').count() + 1;
        let mut indent = 0;
        while max_lines > 0 {
            max_lines /= 10;
            indent += 1;
        }

        emit_call_site_warning!(
            "PTX source code:\n{}",
            kernel_ptx
                .lines()
                .enumerate()
                .map(|(i, l)| format!("{:indent$}| {l}", i + 1))
                .collect::<Vec<_>>()
                .join("\n")
        );
    }
    match info_log {
        Ok(None) => (),
        Ok(Some(info_log)) => emit_call_site_warning!("{ptx_compiler} info log:\n{}", info_log),
        Err(err) => {
            let _ = errors.write_fmt(format_args!(
                "Error fetching the info log of the {ptx_compiler}: {err}\n"
            ));
        },
    };
    match error_log {
        Ok(None) => (),
        Ok(Some(error_log)) => emit_call_site_error!("{ptx_compiler} error log:\n{}", error_log),
        Err(err) => {
            let _ = errors.write_fmt(format_args!(
                "Error fetching the error log of the {ptx_compiler}: {err}\n"
            ));
        },
    };
    if let Err(err) = result {
        let _ = errors.write_fmt(format_args!("Error compiling the PTX source code: {err}\n"));
    }
    if !errors.is_empty() {
        abort_call_site!("{}", errors);
    }

    (quote! { const PTX_STR: &'static str = #kernel_ptx; #(#type_layouts)* }).into()
}

#[allow(clippy::type_complexity)]
fn check_kernel_ptx(
    kernel_ptx: &str,
    specialisation: &str,
    kernel_hash: &proc_macro2::Ident,
) -> (
    Result<(), NvptxError>,
    Result<Option<String>, NvptxError>,
    Result<Option<String>, NvptxError>,
    Result<(u32, u32), NvptxError>,
    Result<(), NvptxError>,
) {
    let compiler = {
        let mut compiler = std::ptr::null_mut();
        if let Err(err) = NvptxError::try_err_from(unsafe {
            ptx_compiler_sys::nvPTXCompilerCreate(
                addr_of_mut!(compiler),
                kernel_ptx.len() as ptx_compiler_sys::size_t,
                kernel_ptx.as_ptr().cast(),
            )
        }) {
            abort_call_site!("PTX compiler creation failed: {}", err);
        }
        compiler
    };

    let result = {
        let kernel_name = if specialisation.is_empty() {
            format!("{kernel_hash}_kernel")
        } else {
            format!(
                "{kernel_hash}_kernel_{:016x}",
                seahash::hash(specialisation.as_bytes())
            )
        };

        let options = vec![
            CString::new("--entry").unwrap(),
            CString::new(kernel_name).unwrap(),
            CString::new("--verbose").unwrap(),
            CString::new("--warn-on-double-precision-use").unwrap(),
            CString::new("--warn-on-local-memory-usage").unwrap(),
            CString::new("--warn-on-spills").unwrap(),
        ];
        let options_ptrs = options.iter().map(|o| o.as_ptr()).collect::<Vec<_>>();

        NvptxError::try_err_from(unsafe {
            ptx_compiler_sys::nvPTXCompilerCompile(
                compiler,
                c_int::try_from(options_ptrs.len()).unwrap(),
                options_ptrs.as_ptr().cast(),
            )
        })
    };

    let error_log = (|| {
        let mut error_log_size = 0;

        NvptxError::try_err_from(unsafe {
            ptx_compiler_sys::nvPTXCompilerGetErrorLogSize(compiler, addr_of_mut!(error_log_size))
        })?;

        if error_log_size == 0 {
            return Ok(None);
        }

        #[allow(clippy::cast_possible_truncation)]
        let mut error_log: Vec<u8> = Vec::with_capacity(error_log_size as usize);

        NvptxError::try_err_from(unsafe {
            ptx_compiler_sys::nvPTXCompilerGetErrorLog(compiler, error_log.as_mut_ptr().cast())
        })?;

        #[allow(clippy::cast_possible_truncation)]
        unsafe {
            error_log.set_len(error_log_size as usize);
        }

        Ok(Some(String::from_utf8_lossy(&error_log).into_owned()))
    })();

    let info_log = (|| {
        let mut info_log_size = 0;

        NvptxError::try_err_from(unsafe {
            ptx_compiler_sys::nvPTXCompilerGetInfoLogSize(compiler, addr_of_mut!(info_log_size))
        })?;

        if info_log_size == 0 {
            return Ok(None);
        }

        #[allow(clippy::cast_possible_truncation)]
        let mut info_log: Vec<u8> = Vec::with_capacity(info_log_size as usize);

        NvptxError::try_err_from(unsafe {
            ptx_compiler_sys::nvPTXCompilerGetInfoLog(compiler, info_log.as_mut_ptr().cast())
        })?;

        #[allow(clippy::cast_possible_truncation)]
        unsafe {
            info_log.set_len(info_log_size as usize);
        }

        Ok(Some(String::from_utf8_lossy(&info_log).into_owned()))
    })();

    let version = (|| {
        let mut major = 0;
        let mut minor = 0;

        NvptxError::try_err_from(unsafe {
            ptx_compiler_sys::nvPTXCompilerGetVersion(addr_of_mut!(major), addr_of_mut!(minor))
        })?;

        Ok((major, minor))
    })();

    let drop = {
        let mut compiler = compiler;
        NvptxError::try_err_from(unsafe {
            ptx_compiler_sys::nvPTXCompilerDestroy(addr_of_mut!(compiler))
        })
    };

    (result, error_log, info_log, version, drop)
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

    match build_kernel_with_specialisation(crate_path, &specialisation_var, specialisation) {
        Ok(kernel_path) => {
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
        },
        Err(err) => {
            eprintln!("{err:?}");
            emit_ptx_build_error();
            None
        },
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
