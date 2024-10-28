use std::{
    cell::Cell,
    collections::HashMap,
    ffi::CString,
    fmt::Write as FmtWrite,
    fs,
    io::{Read, Write},
    os::raw::c_int,
    path::{Path, PathBuf},
    ptr::addr_of_mut,
    sync::LazyLock,
};

use cargo_util::ProcessBuilder;
use colored::Colorize;
use proc_macro::TokenStream;
use proc_macro2::Span;
use quote::quote;
use regex::Regex;

use crate::kernel::{
    lints::{LintLevel, PtxLint},
    utils::skip_kernel_compilation,
    KERNEL_TYPE_LAYOUT_HASH_SEED_IDENT, KERNEL_TYPE_LAYOUT_IDENT, KERNEL_TYPE_USE_END_CANARY,
    KERNEL_TYPE_USE_START_CANARY, PTX_CSTR_IDENT,
};

mod config;
mod ptx_compiler_sys;

use config::{CheckKernelConfig, LinkKernelConfig};
use ptx_compiler_sys::NvptxError;

const TARGET_NAME: &str = "nvptx64-nvidia-cuda";

pub fn check_kernel(tokens: TokenStream) -> TokenStream {
    proc_macro_error2::set_dummy(
        quote! {::core::compile_error!("rust-cuda PTX kernel check failed");},
    );

    let CheckKernelConfig {
        kernel,
        kernel_hash,
        crate_name,
        crate_path,
    } = match syn::parse(tokens) {
        Ok(config) => config,
        Err(err) => {
            abort_call_site!(
                "check_kernel!(KERNEL HASH NAME PATH) expects KERNEL and HASH identifiers, annd \
                 NAME and PATH string literals: {:?}",
                err
            )
        },
    };

    let kernel_ptx = compile_kernel_ptx(&kernel, &crate_name, &crate_path, Specialisation::Check);

    check_kernel_ptx_and_report(
        &kernel_ptx,
        Specialisation::Check,
        &kernel_hash,
        &HashMap::new(),
    );

    quote!().into()
}

pub fn compile_kernel(tokens: TokenStream) -> TokenStream {
    let ptx_cstr_ident = syn::Ident::new(PTX_CSTR_IDENT, Span::call_site());
    let ffi_signature_ident = syn::Ident::new(KERNEL_TYPE_LAYOUT_IDENT, Span::call_site());
    let ffi_signature_hash_seed_ident =
        syn::Ident::new(KERNEL_TYPE_LAYOUT_HASH_SEED_IDENT, Span::call_site());

    proc_macro_error2::set_dummy(quote! {
        const #ptx_cstr_ident: &'static ::core::ffi::CStr = c"ERROR in this PTX compilation";

        const fn #ffi_signature_ident(_hashes: &[u64]) -> HostAndDeviceKernelSignatureTypeLayout {
            HostAndDeviceKernelSignatureTypeLayout::Match
        }

        const fn #ffi_signature_hash_seed_ident() -> u64 {
            42
        }

        ::core::compile_error!("rust-cuda PTX kernel compilation failed");
    });

    let LinkKernelConfig {
        kernel,
        kernel_hash,
        crate_name,
        crate_path,
        specialisation,
        ptx_lint_levels,
    } = match syn::parse(tokens) {
        Ok(config) => config,
        Err(err) => {
            abort_call_site!(
                "compile_kernel!(KERNEL HASH NAME PATH SPECIALISATION LINTS,*) expects KERNEL and \
                 HASH identifiers, NAME and PATH string literals, and SPECIALISATION and LINTS \
                 tokens: {:?}",
                err
            )
        },
    };

    if skip_kernel_compilation() {
        return quote! {
            const #ptx_cstr_ident: &'static ::core::ffi::CStr = c"CLIPPY skips specialised PTX compilation";
        }
        .into();
    }

    let mut kernel_ptx = compile_kernel_ptx(
        &kernel,
        &crate_name,
        &crate_path,
        Specialisation::Link(&specialisation),
    );

    let type_layout = extract_ptx_kernel_layout(&mut kernel_ptx);
    remove_kernel_type_use_from_ptx(&mut kernel_ptx);

    check_kernel_ptx_and_report(
        &kernel_ptx,
        Specialisation::Link(&specialisation),
        &kernel_hash,
        &ptx_lint_levels,
    );

    let kernel_ptx = match CString::new(kernel_ptx) {
        Ok(kernel_ptx) => kernel_ptx,
        Err(err) => abort_call_site!(
            "Kernel compilation generated invalid PTX: internal nul byte: {:?}",
            err
        ),
    };

    let kernel_ptx = proc_macro::Literal::c_string(&kernel_ptx);
    let kernel_ptx = proc_macro2::TokenStream::from(proc_macro::TokenStream::from(
        proc_macro::TokenTree::Literal(kernel_ptx),
    ));

    (quote! { const #ptx_cstr_ident: &'static ::core::ffi::CStr = #kernel_ptx; #type_layout })
        .into()
}

fn extract_ptx_kernel_layout(kernel_ptx: &mut String) -> proc_macro2::TokenStream {
    const BEFORE_PARAM_PATTERN: &str = ".visible .global .align 1 .b8 ";
    const PARAM_LEN_PATTERN: &str = "[";
    const LEN_BYTES_PATTERN: &str = "] = {";
    const AFTER_BYTES_PATTERN: &str = "};";

    let mut type_layout_metas = HashMap::new();

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

        if type_layout_metas
            .insert(String::from(param), bytes)
            .is_some()
        {
            abort_call_site!(
                "Kernel compilation generated invalid PTX: duplicate type information for {}",
                param
            );
        }

        let type_layout_end = bytes_start + bytes_end_offset + AFTER_BYTES_PATTERN.len();

        kernel_ptx.replace_range(type_layout_start..type_layout_end, "");
    }

    let Some(type_layout_hash_seed) = type_layout_metas.remove(KERNEL_TYPE_LAYOUT_HASH_SEED_IDENT)
    else {
        abort_call_site!(
            "Kernel compilation generated invalid PTX: missing type information hash seed"
        );
    };
    let Ok(type_layout_hash_seed) = type_layout_hash_seed.as_slice().try_into() else {
        abort_call_site!(
            "Kernel compilation generated invalid PTX: invalid type information hash seed"
        );
    };
    let type_layout_hash_seed = u64::from_le_bytes(type_layout_hash_seed);

    let Some(type_layout_hash) = type_layout_metas.remove(KERNEL_TYPE_LAYOUT_IDENT) else {
        abort_call_site!("Kernel compilation generated invalid PTX: missing type information");
    };
    let mut type_layout_hash_remainder = type_layout_hash.as_slice();
    let mut type_layout_hash = Vec::new();
    while let Some((hash, remainder)) = type_layout_hash_remainder.split_first_chunk() {
        type_layout_hash.push(u64::from_le_bytes(*hash));
        type_layout_hash_remainder = remainder;
    }
    if !type_layout_hash_remainder.is_empty() {
        abort_call_site!("Kernel compilation generated invalid PTX: invalid type information");
    }

    let ffi_signature_ident = syn::Ident::new(KERNEL_TYPE_LAYOUT_IDENT, Span::call_site());
    let ffi_signature_hash_seed_ident =
        syn::Ident::new(KERNEL_TYPE_LAYOUT_HASH_SEED_IDENT, Span::call_site());

    quote! {
        const fn #ffi_signature_ident(hashes: &[u64]) -> HostAndDeviceKernelSignatureTypeLayout {
            check_ptx_kernel_signature(hashes, &[#(#type_layout_hash),*])
        }

        const fn #ffi_signature_hash_seed_ident() -> u64 {
            #type_layout_hash_seed
        }
    }
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

#[expect(clippy::too_many_lines)]
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
                .get(&PtxLint::DumpAssembly)
                .map_or(false, |level| *level > LintLevel::Allow)
            {
                const HEX: [char; 16] = [
                    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f',
                ];

                let mut binary_hex = String::with_capacity(binary.len() * 2);
                #[expect(clippy::indexing_slicing)] // index always in 0..16
                for byte in binary {
                    binary_hex.push(HEX[usize::from(byte >> 4)]);
                    binary_hex.push(HEX[usize::from(byte & 0x0F)]);
                }

                if ptx_lint_levels
                    .get(&PtxLint::DumpAssembly)
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

#[expect(clippy::type_complexity)]
#[expect(clippy::too_many_lines)]
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
        #[expect(unsafe_code)]
        // Safety: FFI
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
                .get(&PtxLint::LocalMemoryUse)
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

            #[expect(unsafe_code)]
            // Safety: FFI
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
            .get(&PtxLint::LocalMemoryUse)
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

        #[expect(unsafe_code)]
        // Safety: FFI
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

        #[expect(unsafe_code)]
        // Safety: FFI
        NvptxError::try_err_from(unsafe {
            ptx_compiler_sys::nvPTXCompilerGetErrorLogSize(compiler, addr_of_mut!(error_log_size))
        })?;

        if error_log_size == 0 {
            return Ok(None);
        }

        #[expect(clippy::cast_possible_truncation)]
        let mut error_log: Vec<u8> = vec![0; error_log_size as usize];

        #[expect(unsafe_code)]
        // Safety: FFI
        NvptxError::try_err_from(unsafe {
            ptx_compiler_sys::nvPTXCompilerGetErrorLog(compiler, error_log.as_mut_ptr().cast())
        })?;

        Ok(Some(String::from_utf8_lossy(&error_log).into_owned()))
    })();

    let info_log = (|| {
        let mut info_log_size = 0;

        #[expect(unsafe_code)]
        // Safety: FFI
        NvptxError::try_err_from(unsafe {
            ptx_compiler_sys::nvPTXCompilerGetInfoLogSize(compiler, addr_of_mut!(info_log_size))
        })?;

        if info_log_size == 0 {
            return Ok(None);
        }

        #[expect(clippy::cast_possible_truncation)]
        let mut info_log: Vec<u8> = vec![0; info_log_size as usize];

        #[expect(unsafe_code)]
        // Safety: FFI
        NvptxError::try_err_from(unsafe {
            ptx_compiler_sys::nvPTXCompilerGetInfoLog(compiler, info_log.as_mut_ptr().cast())
        })?;

        Ok(Some(String::from_utf8_lossy(&info_log).into_owned()))
    })();

    let binary = (|| {
        if result.is_err() {
            return Ok(None);
        }

        let mut binary_size = 0;

        #[expect(unsafe_code)]
        // Safety: FFI
        NvptxError::try_err_from(unsafe {
            ptx_compiler_sys::nvPTXCompilerGetCompiledProgramSize(
                compiler,
                addr_of_mut!(binary_size),
            )
        })?;

        if binary_size == 0 {
            return Ok(None);
        }

        #[expect(clippy::cast_possible_truncation)]
        let mut binary: Vec<u8> = vec![0; binary_size as usize];

        #[expect(unsafe_code)]
        // Safety: FFI
        NvptxError::try_err_from(unsafe {
            ptx_compiler_sys::nvPTXCompilerGetCompiledProgram(compiler, binary.as_mut_ptr().cast())
        })?;

        Ok(Some(binary))
    })();

    let version = (|| {
        let mut major = 0;
        let mut minor = 0;

        #[expect(unsafe_code)]
        // Safety: FFI
        NvptxError::try_err_from(unsafe {
            ptx_compiler_sys::nvPTXCompilerGetVersion(addr_of_mut!(major), addr_of_mut!(minor))
        })?;

        Ok((major, minor))
    })();

    let drop = {
        let mut compiler = compiler;
        #[expect(unsafe_code)]
        // Safety: FFI
        NvptxError::try_err_from(unsafe {
            ptx_compiler_sys::nvPTXCompilerDestroy(addr_of_mut!(compiler))
        })
    };

    (result, error_log, info_log, binary, version, drop)
}

fn compile_kernel_ptx(
    kernel: &syn::Ident,
    crate_name: &str,
    crate_path: &Path,
    specialisation: Specialisation,
) -> String {
    let specialisation_var = format!(
        "RUST_CUDA_DERIVE_SPECIALISE_{}_{}",
        crate_name.to_uppercase(),
        kernel.to_string().to_uppercase()
    );

    let kernel_path = build_kernel_with_specialisation(
        crate_name,
        crate_path,
        &specialisation_var,
        specialisation,
    );

    let mut file = fs::File::open(&kernel_path)
        .unwrap_or_else(|_| abort_call_site!("Failed to open kernel file at {:?}.", &kernel_path));

    let mut kernel_ptx = String::new();

    file.read_to_string(&mut kernel_ptx)
        .unwrap_or_else(|_| abort_call_site!("Failed to read kernel file at {:?}.", &kernel_path));

    colored::control::set_override(true);
    eprintln!(
        "{} {} compiling a PTX crate.",
        "[PTX]".bright_black().bold(),
        "Finished".green().bold()
    );
    colored::control::unset_override();

    kernel_ptx
}

fn build_kernel_with_specialisation(
    crate_name: &str,
    crate_path: &Path,
    env_var: &str,
    specialisation: Specialisation,
) -> PathBuf {
    let specialisation_prefix = match specialisation {
        Specialisation::Check => String::from("chECK"),
        Specialisation::Link(specialisation) => {
            format!("{:016x}", seahash::hash(specialisation.as_bytes()))
        },
    };

    let any_output = Cell::new(false);

    let ptx_path = cargo_build_with_specialisation(
        crate_path,
        crate_name,
        &specialisation_prefix,
        (env_var, specialisation),
        |stdout_line| {
            if let Ok(cargo_metadata::Message::CompilerMessage(mut message)) =
                serde_json::from_str(stdout_line)
            {
                if !any_output.replace(true) {
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

            if !any_output.replace(true) {
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
    );

    let mut specialised_ptx_path = ptx_path.clone();

    specialised_ptx_path.set_extension(format!("{specialisation_prefix}.ptx"));

    if let Err(err) = fs::copy(&ptx_path, &specialised_ptx_path) {
        abort_call_site!(
            "Failed to copy kernel from {:?} to {:?}: {}",
            ptx_path,
            specialised_ptx_path,
            err
        );
    }

    if let Specialisation::Link(specialisation) = specialisation {
        if let Err(err) = fs::OpenOptions::new()
            .append(true)
            .open(&specialised_ptx_path)
            .and_then(|mut file| writeln!(file, "\n// {specialisation}"))
        {
            abort_call_site!(
                "Failed to write specialisation to {:?}: {}",
                specialised_ptx_path,
                err
            );
        }
    }

    specialised_ptx_path
}

#[expect(clippy::too_many_lines)]
fn cargo_build_with_specialisation<O: FnMut(&str), E: FnMut(&str)>(
    crate_path: &Path,
    crate_name: &str,
    crate_suffix: &str,
    specialisation: (&str, Specialisation),
    mut on_stdout_line: O,
    mut on_stderr_line: E,
) -> PathBuf {
    fn output_is_not_verbose(line: &str) -> bool {
        !line.starts_with("+ ")
            && !line.contains("Running")
            && !line.contains("Fresh")
            && !line.starts_with("Caused by:")
            && !line.starts_with("  process didn\'t exit successfully: ")
    }

    const LAST_BUILD_CMD: &str = ".last-build-command";

    static SUFFIX_REGEX: LazyLock<Regex> = LazyLock::new(|| {
        Regex::new(r"-C extra-filename=([\S]+)").expect("Unable to parse regex...")
    });

    if !matches!(fs::metadata(crate_path.join("Cargo.toml")), Ok(metadata) if metadata.is_file()) {
        abort_call_site!(
            "{:?} is not a valid crate manifest path",
            crate_path.join("Cargo.toml")
        );
    }

    let is_library = crate_path.join("src").join("lib.rs").exists();
    let is_binary = crate_path.join("src").join("main.rs").exists();

    if !is_library {
        if is_binary {
            abort_call_site!(
                "{} is a binary-only crate, which is not supported",
                crate_name
            );
        } else {
            abort_call_site!(
                "unable to find either `lib.rs` nor `main.rs` for {}",
                crate_name
            );
        }
    }

    let output_file_prefix = crate_name.replace('-', "_");

    // TODO: use the cargo env variable
    let mut cargo = ProcessBuilder::new("cargo");
    let mut args = vec!["rustc"];

    if matches!(specialisation.1, Specialisation::Link(_)) {
        args.push("--release");
    }

    args.push("--color=always");
    args.push("--message-format=json,json-diagnostic-rendered-ansi");

    args.push("--target");
    args.push(TARGET_NAME);

    args.push("--lib");

    args.push("-v");

    args.push("--");

    let crate_type = "cdylib,rlib";
    args.push("--crate-type");
    args.push(crate_type);

    let mut output_path = PathBuf::from(env!("OUT_DIR"));
    output_path.push(&output_file_prefix);
    if let Err(err) = fs::create_dir_all(&output_path) {
        abort_call_site!(
            "failed to create the output path {:?}: {}",
            output_path,
            err
        );
    }

    cargo
        .args(&args)
        .cwd(crate_path)
        .env("CARGO_TARGET_DIR", output_path.as_path())
        .env(
            specialisation.0,
            match specialisation.1 {
                Specialisation::Check => "chECK",
                Specialisation::Link(specialisation) => specialisation,
            },
        );

    let raw_output = cargo
        .exec_with_streaming(
            &mut |s| {
                on_stdout_line(s);
                Ok(())
            },
            &mut |s| {
                if output_is_not_verbose(s) {
                    on_stderr_line(s);
                }
                Ok(())
            },
            true,
        )
        .unwrap_or_else(|err| abort_call_site!("failed to execute command '{:?}': {}", cargo, err));

    let stderr = String::from_utf8(raw_output.stderr)
        .unwrap_or_else(|err| abort_call_site!("failed to parse stderr as UTF8: {}", err));
    let mut stderr_lines = stderr.trim_matches('\n').split('\n');

    if !raw_output.status.success() {
        #[allow(clippy::manual_filter_map)]
        let lines = stderr_lines
            .filter(|s| output_is_not_verbose(s))
            .map(String::from)
            .collect::<Vec<_>>()
            .join("\n");

        abort_call_site!(
            "Failed to build PTX for {} ({})\n\n{}",
            crate_name,
            match specialisation.1 {
                Specialisation::Check => "chECK",
                Specialisation::Link(specialisation) => specialisation,
            },
            lines
        );
    }

    let last_build_cmd = output_path.join(format!("{LAST_BUILD_CMD}.{crate_suffix}"));

    // We need the build command to get real output filename.
    #[allow(clippy::manual_find_map)]
    let build_command = stderr_lines
        .find(|line| {
            line.contains(&format!("--crate-name {output_file_prefix}"))
                && line.contains(&format!("--crate-type {crate_type}"))
        })
        .map(String::from)
        .or_else(|| fs::read_to_string(&last_build_cmd).ok())
        .unwrap_or_else(|| {
            abort_call_site!("unable to find build command of the device crate");
        });

    if let Err(err) = fs::write(&last_build_cmd, build_command.as_bytes()) {
        abort_call_site!(
            "failed to cache the last build command to {:?}",
            last_build_cmd
        );
    };

    let (file_suffix, found_suffix) = match SUFFIX_REGEX.captures(&build_command) {
        Some(caps) => (caps[1].to_string(), true),
        None => (String::new(), false),
    };

    let assembly_path = output_path
        .join(TARGET_NAME)
        .join(match specialisation.1 {
            Specialisation::Check => "debug",
            Specialisation::Link(_) => "release",
        })
        .join("deps")
        .join(format!("{output_file_prefix}{file_suffix}.ptx"));

    if assembly_path.exists() {
        return assembly_path;
    }

    if found_suffix {
        abort_call_site!("unable to find PTX assembly as specified by `extra-filename` rustc flag");
    } else {
        abort_call_site!("Unable to find `extra-filename` rustc flag")
    }
}

#[derive(Copy, Clone, Debug)]
enum Specialisation<'a> {
    Check,
    Link(&'a str),
}
