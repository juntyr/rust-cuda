use std::{
    collections::HashMap,
    env,
    ffi::{CStr, CString},
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

use super::{
    lints::{LintLevel, PtxLint},
    utils::skip_kernel_compilation,
    KERNEL_TYPE_USE_END_CANARY, KERNEL_TYPE_USE_START_CANARY,
};

mod config;
mod error;
mod ptx_compiler_sys;

use config::{CheckKernelConfig, LinkKernelConfig};
use error::emit_ptx_build_error;
use ptx_compiler_sys::NvptxError;

pub fn check_kernel(tokens: TokenStream) -> TokenStream {
    proc_macro_error::set_dummy(quote! {::core::result::Result::Err(())});

    let CheckKernelConfig {
        kernel_hash,
        args,
        crate_name,
        crate_path,
    } = match syn::parse_macro_input::parse(tokens) {
        Ok(config) => config,
        Err(err) => {
            abort_call_site!(
                "check_kernel!(HASH ARGS NAME PATH) expects HASH and ARGS identifiers, annd NAME \
                 and PATH string literals: {:?}",
                err
            )
        },
    };

    let kernel_ptx = compile_kernel(&args, &crate_name, &crate_path, Specialisation::Check);

    let Some(kernel_ptx) = kernel_ptx else {
        return quote!(::core::result::Result::Err(())).into();
    };

    check_kernel_ptx_and_report(
        &kernel_ptx,
        Specialisation::Check,
        &kernel_hash,
        &HashMap::new(),
    );

    quote!(::core::result::Result::Ok(())).into()
}

#[allow(clippy::module_name_repetitions)]
pub fn link_kernel(tokens: TokenStream) -> TokenStream {
    proc_macro_error::set_dummy(quote! {
        const PTX_CSTR: &'static ::core::ffi::CStr = c"ERROR in this PTX compilation";
    });

    let LinkKernelConfig {
        kernel: _kernel,
        kernel_hash,
        args,
        crate_name,
        crate_path,
        specialisation,
        ptx_lint_levels,
    } = match syn::parse_macro_input::parse(tokens) {
        Ok(config) => config,
        Err(err) => {
            abort_call_site!(
                "link_kernel!(KERNEL HASH ARGS NAME PATH SPECIALISATION LINTS,*) expects KERNEL, \
                 HASH, and ARGS identifiers, NAME and PATH string literals, and SPECIALISATION \
                 and LINTS tokens: {:?}",
                err
            )
        },
    };

    if skip_kernel_compilation() {
        return quote! {
            const PTX_CSTR: &'static ::core::ffi::CStr = c"CLIPPY skips specialised PTX compilation";
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
            const PTX_CSTR: &'static ::core::ffi::CStr = c"ERROR in this PTX compilation";
        })
        .into();
    };

    let type_layouts = extract_ptx_kernel_layout(&mut kernel_ptx);
    remove_kernel_type_use_from_ptx(&mut kernel_ptx);

    check_kernel_ptx_and_report(
        &kernel_ptx,
        Specialisation::Link(&specialisation),
        &kernel_hash,
        &ptx_lint_levels,
    );

    let mut kernel_ptx = kernel_ptx.into_bytes();
    kernel_ptx.push(b'\0');

    if let Err(err) = CStr::from_bytes_with_nul(&kernel_ptx) {
        abort_call_site!(
            "Kernel compilation generated invalid PTX: internal nul byte: {:?}",
            err
        );
    }

    // TODO: CStr constructor blocked on https://github.com/rust-lang/rust/issues/118560
    let kernel_ptx = syn::LitByteStr::new(&kernel_ptx, proc_macro2::Span::call_site());
    // Safety: the validity of kernel_ptx as a CStr was just checked above
    let kernel_ptx =
        quote! { unsafe { ::core::ffi::CStr::from_bytes_with_nul_unchecked(#kernel_ptx) } };

    (quote! { const PTX_CSTR: &'static ::core::ffi::CStr = #kernel_ptx; #(#type_layouts)* }).into()
}

fn extract_ptx_kernel_layout(kernel_ptx: &mut String) -> Vec<proc_macro2::TokenStream> {
    const BEFORE_PARAM_PATTERN: &str = ".visible .global .align 1 .b8 ";
    const PARAM_LEN_PATTERN: &str = "[";
    const LEN_BYTES_PATTERN: &str = "] = {";
    const AFTER_BYTES_PATTERN: &str = "};";

    let mut type_layouts = Vec::new();

    while let Some(type_layout_start) = kernel_ptx.find(BEFORE_PARAM_PATTERN) {
        let param_start = type_layout_start + BEFORE_PARAM_PATTERN.len();

        let Some(len_start_offset) = kernel_ptx[param_start..].find(PARAM_LEN_PATTERN) else {
            abort_call_site!("Kernel compilation generated invalid PTX: missing type layout data")
        };
        let len_start = param_start + len_start_offset + PARAM_LEN_PATTERN.len();

        let Some(bytes_start_offset) = kernel_ptx[len_start..].find(LEN_BYTES_PATTERN) else {
            abort_call_site!("Kernel compilation generated invalid PTX: missing type layout length")
        };
        let bytes_start = len_start + bytes_start_offset + LEN_BYTES_PATTERN.len();

        let Some(bytes_end_offset) = kernel_ptx[bytes_start..].find(AFTER_BYTES_PATTERN) else {
            abort_call_site!("Kernel compilation generated invalid PTX: invalid type layout data")
        };
        let param = &kernel_ptx[param_start..(param_start + len_start_offset)];
        let len = &kernel_ptx[len_start..(len_start + bytes_start_offset)];
        let bytes = &kernel_ptx[bytes_start..(bytes_start + bytes_end_offset)];

        let param = quote::format_ident!("{}", param);

        let Ok(len) = len.parse::<usize>() else {
            abort_call_site!("Kernel compilation generated invalid PTX: invalid type layout length")
        };
        let Ok(bytes) = bytes
            .split(", ")
            .map(std::str::FromStr::from_str)
            .collect::<Result<Vec<u8>, _>>()
        else {
            abort_call_site!("Kernel compilation generated invalid PTX: invalid type layout byte")
        };

        if bytes.len() != len {
            abort_call_site!(
                "Kernel compilation generated invalid PTX: type layout length mismatch"
            );
        }

        let byte_str = syn::LitByteStr::new(&bytes, proc_macro2::Span::call_site());

        type_layouts.push(quote! {
            const #param: &[u8; #len] = #byte_str;
        });

        let type_layout_end = bytes_start + bytes_end_offset + AFTER_BYTES_PATTERN.len();

        kernel_ptx.replace_range(type_layout_start..type_layout_end, "");
    }

    type_layouts
}

fn remove_kernel_type_use_from_ptx(kernel_ptx: &mut String) {
    while let Some(kernel_type_layout_start) = kernel_ptx.find(KERNEL_TYPE_USE_START_CANARY) {
        let kernel_type_layout_start = kernel_ptx[..kernel_type_layout_start]
            .rfind('\n')
            .unwrap_or(kernel_type_layout_start);

        let Some(kernel_type_layout_end_offset) =
            kernel_ptx[kernel_type_layout_start..].find(KERNEL_TYPE_USE_END_CANARY)
        else {
            abort_call_site!(
                "Kernel compilation generated invalid PTX: incomplete type layout use section"
            );
        };

        let kernel_type_layout_end_offset = kernel_type_layout_end_offset
            + kernel_ptx[kernel_type_layout_start + kernel_type_layout_end_offset..]
                .find('\n')
                .unwrap_or(KERNEL_TYPE_USE_END_CANARY.len());

        let kernel_type_layout_end = kernel_type_layout_start + kernel_type_layout_end_offset;

        kernel_ptx.replace_range(kernel_type_layout_start..kernel_type_layout_end, "");
    }
}

#[allow(clippy::too_many_lines)]
fn check_kernel_ptx_and_report(
    kernel_ptx: &str,
    specialisation: Specialisation,
    kernel_hash: &proc_macro2::Ident,
    ptx_lint_levels: &HashMap<PtxLint, LintLevel>,
) {
    let (result, error_log, info_log, binary, version, drop) =
        check_kernel_ptx(kernel_ptx, specialisation, kernel_hash, ptx_lint_levels);

    let ptx_compiler = match &version {
        Ok((major, minor)) => format!("PTX compiler v{major}.{minor}"),
        Err(_) => String::from("PTX compiler"),
    };

    let mut errors = String::new();

    if let Err(err) = drop {
        let _ = errors.write_fmt(format_args!("Error dropping the {ptx_compiler}: {err}\n"));
    }

    if let Err(err) = version {
        let _ = errors.write_fmt(format_args!(
            "Error fetching the version of the {ptx_compiler}: {err}\n"
        ));
    }

    let ptx_source_code = {
        let mut max_lines = kernel_ptx.chars().filter(|c| *c == '\n').count() + 1;
        let mut indent = 0;
        while max_lines > 0 {
            max_lines /= 10;
            indent += 1;
        }

        format!(
            "PTX source code:\n{}",
            kernel_ptx
                .lines()
                .enumerate()
                .map(|(i, l)| format!("{:indent$}| {l}", i + 1))
                .collect::<Vec<_>>()
                .join("\n")
        )
    };

    match binary {
        Ok(None) => (),
        Ok(Some(binary)) => {
            if ptx_lint_levels
                .get(&PtxLint::DumpBinary)
                .map_or(false, |level| *level > LintLevel::Allow)
            {
                const HEX: [char; 16] = [
                    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f',
                ];

                let mut binary_hex = String::with_capacity(binary.len() * 2);
                for byte in binary {
                    binary_hex.push(HEX[usize::from(byte >> 4)]);
                    binary_hex.push(HEX[usize::from(byte & 0x0F)]);
                }

                if ptx_lint_levels
                    .get(&PtxLint::DumpBinary)
                    .map_or(false, |level| *level > LintLevel::Warn)
                {
                    emit_call_site_error!(
                        "{} compiled binary:\n{}\n\n{}",
                        ptx_compiler,
                        binary_hex,
                        ptx_source_code
                    );
                } else {
                    emit_call_site_warning!(
                        "{} compiled binary:\n{}\n\n{}",
                        ptx_compiler,
                        binary_hex,
                        ptx_source_code
                    );
                }
            }
        },
        Err(err) => {
            let _ = errors.write_fmt(format_args!(
                "Error fetching the compiled binary from {ptx_compiler}: {err}\n"
            ));
        },
    }

    match info_log {
        Ok(None) => (),
        Ok(Some(info_log)) => emit_call_site_warning!(
            "{} info log:\n{}\n{}",
            ptx_compiler,
            info_log,
            ptx_source_code
        ),
        Err(err) => {
            let _ = errors.write_fmt(format_args!(
                "Error fetching the info log of the {ptx_compiler}: {err}\n"
            ));
        },
    };

    let error_log = match error_log {
        Ok(None) => String::new(),
        Ok(Some(error_log)) => {
            format!("{ptx_compiler} error log:\n{error_log}\n{ptx_source_code}")
        },
        Err(err) => {
            let _ = errors.write_fmt(format_args!(
                "Error fetching the error log of the {ptx_compiler}: {err}\n"
            ));
            String::new()
        },
    };

    if let Err(err) = result {
        let _ = errors.write_fmt(format_args!("Error compiling the PTX source code: {err}\n"));
    }

    if !error_log.is_empty() || !errors.is_empty() {
        abort_call_site!(
            "{error_log}{}{errors}",
            if !error_log.is_empty() && !errors.is_empty() {
                "\n\n"
            } else {
                ""
            }
        );
    }
}

#[allow(clippy::type_complexity)]
#[allow(clippy::too_many_lines)]
fn check_kernel_ptx(
    kernel_ptx: &str,
    specialisation: Specialisation,
    kernel_hash: &proc_macro2::Ident,
    ptx_lint_levels: &HashMap<PtxLint, LintLevel>,
) -> (
    Result<(), NvptxError>,
    Result<Option<String>, NvptxError>,
    Result<Option<String>, NvptxError>,
    Result<Option<Vec<u8>>, NvptxError>,
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

    let result = (|| {
        let kernel_name = match specialisation {
            Specialisation::Check => format!("{kernel_hash}_chECK"),
            Specialisation::Link("") => format!("{kernel_hash}_kernel"),
            Specialisation::Link(specialisation) => format!(
                "{kernel_hash}_kernel_{:016x}",
                seahash::hash(specialisation.as_bytes())
            ),
        };
        let kernel_name = CString::new(kernel_name).unwrap();

        let mut options = vec![c"--entry", kernel_name.as_c_str()];

        if ptx_lint_levels
            .values()
            .any(|level| *level > LintLevel::Warn)
        {
            let mut options = options.clone();

            if ptx_lint_levels
                .get(&PtxLint::Verbose)
                .map_or(false, |level| *level > LintLevel::Warn)
            {
                options.push(c"--verbose");
            }
            if ptx_lint_levels
                .get(&PtxLint::DoublePrecisionUse)
                .map_or(false, |level| *level > LintLevel::Warn)
            {
                options.push(c"--warn-on-double-precision-use");
            }
            if ptx_lint_levels
                .get(&PtxLint::LocalMemoryUsage)
                .map_or(false, |level| *level > LintLevel::Warn)
            {
                options.push(c"--warn-on-local-memory-usage");
            }
            if ptx_lint_levels
                .get(&PtxLint::RegisterSpills)
                .map_or(false, |level| *level > LintLevel::Warn)
            {
                options.push(c"--warn-on-spills");
            }
            if ptx_lint_levels
                .get(&PtxLint::DynamicStackSize)
                .map_or(true, |level| *level <= LintLevel::Warn)
            {
                options.push(c"--suppress-stack-size-warning");
            }
            options.push(c"--warning-as-error");

            let options_ptrs = options.iter().map(|o| o.as_ptr()).collect::<Vec<_>>();

            NvptxError::try_err_from(unsafe {
                ptx_compiler_sys::nvPTXCompilerCompile(
                    compiler,
                    c_int::try_from(options_ptrs.len()).unwrap(),
                    options_ptrs.as_ptr().cast(),
                )
            })?;
        };

        if ptx_lint_levels
            .get(&PtxLint::Verbose)
            .map_or(false, |level| *level > LintLevel::Allow)
        {
            options.push(c"--verbose");
        }
        if ptx_lint_levels
            .get(&PtxLint::DoublePrecisionUse)
            .map_or(false, |level| *level > LintLevel::Allow)
        {
            options.push(c"--warn-on-double-precision-use");
        }
        if ptx_lint_levels
            .get(&PtxLint::LocalMemoryUsage)
            .map_or(false, |level| *level > LintLevel::Allow)
        {
            options.push(c"--warn-on-local-memory-usage");
        }
        if ptx_lint_levels
            .get(&PtxLint::RegisterSpills)
            .map_or(false, |level| *level > LintLevel::Allow)
        {
            options.push(c"--warn-on-spills");
        }
        if ptx_lint_levels
            .get(&PtxLint::DynamicStackSize)
            .map_or(true, |level| *level < LintLevel::Warn)
        {
            options.push(c"--suppress-stack-size-warning");
        }

        let options_ptrs = options.iter().map(|o| o.as_ptr()).collect::<Vec<_>>();

        NvptxError::try_err_from(unsafe {
            ptx_compiler_sys::nvPTXCompilerCompile(
                compiler,
                c_int::try_from(options_ptrs.len()).unwrap(),
                options_ptrs.as_ptr().cast(),
            )
        })
    })();

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

    let binary = (|| {
        if result.is_err() {
            return Ok(None);
        }

        let mut binary_size = 0;

        NvptxError::try_err_from(unsafe {
            ptx_compiler_sys::nvPTXCompilerGetCompiledProgramSize(
                compiler,
                addr_of_mut!(binary_size),
            )
        })?;

        if binary_size == 0 {
            return Ok(None);
        }

        #[allow(clippy::cast_possible_truncation)]
        let mut binary: Vec<u8> = Vec::with_capacity(binary_size as usize);

        NvptxError::try_err_from(unsafe {
            ptx_compiler_sys::nvPTXCompilerGetCompiledProgram(compiler, binary.as_mut_ptr().cast())
        })?;

        #[allow(clippy::cast_possible_truncation)]
        unsafe {
            binary.set_len(binary_size as usize);
        }

        Ok(Some(binary))
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

    (result, error_log, info_log, binary, version, drop)
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
            eprintln!("{err}");
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

        let build = builder.build_live(
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
        )?;

        match build {
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
