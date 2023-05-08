use std::{collections::HashMap, fmt};

use syn::spanned::Spanned;

pub fn parse_ptx_lint_level(
    path: &syn::Path,
    nested: &syn::punctuated::Punctuated<syn::NestedMeta, syn::token::Comma>,
    ptx_lint_levels: &mut HashMap<PtxLint, LintLevel>,
) {
    let level = match path.get_ident() {
        Some(ident) if ident == "allow" => LintLevel::Allow,
        Some(ident) if ident == "warn" => LintLevel::Warn,
        Some(ident) if ident == "deny" => LintLevel::Deny,
        Some(ident) if ident == "forbid" => LintLevel::Forbid,
        _ => {
            emit_error!(
                path.span(),
                "[rust-cuda]: Invalid lint #[kernel(<level>(<lint>))] attribute: unknown lint \
                 level, must be one of `allow`, `warn`, `deny`, `forbid`.",
            );

            return;
        },
    };

    for meta in nested {
        let syn::NestedMeta::Meta(syn::Meta::Path(path)) = meta else {
            emit_error!(
                meta.span(),
                "[rust-cuda]: Invalid #[kernel({}(<lint>))] attribute.",
                level,
            );
            continue;
        };

        if path.leading_colon.is_some()
            || path.segments.empty_or_trailing()
            || path.segments.len() != 2
        {
            emit_error!(
                meta.span(),
                "[rust-cuda]: Invalid #[kernel({}(<lint>))] attribute: <lint> must be of the form \
                 `ptx::lint`.",
                level,
            );
            continue;
        }

        let Some(syn::PathSegment { ident: namespace, arguments: syn::PathArguments::None }) = path.segments.first() else {
            emit_error!(
                meta.span(),
                "[rust-cuda]: Invalid #[kernel({}(<lint>))] attribute: <lint> must be of the form `ptx::lint`.",
                level,
            );
            continue;
        };

        if namespace != "ptx" {
            emit_error!(
                meta.span(),
                "[rust-cuda]: Invalid #[kernel({}(<lint>))] attribute: <lint> must be of the form \
                 `ptx::lint`.",
                level,
            );
            continue;
        }

        let Some(syn::PathSegment { ident: lint, arguments: syn::PathArguments::None }) = path.segments.last() else {
            emit_error!(
                meta.span(),
                "[rust-cuda]: Invalid #[kernel({}(<lint>))] attribute: <lint> must be of the form `ptx::lint`.",
                level,
            );
            continue;
        };

        let lint = match lint {
            l if l == "verbose" => PtxLint::Verbose,
            l if l == "double_precision_use" => PtxLint::DoublePrecisionUse,
            l if l == "local_memory_usage" => PtxLint::LocalMemoryUsage,
            l if l == "register_spills" => PtxLint::RegisterSpills,
            _ => {
                emit_error!(
                    meta.span(),
                    "[rust-cuda]: Unknown PTX kernel lint `ptx::{}`.",
                    lint,
                );
                continue;
            },
        };

        match ptx_lint_levels.get(&lint) {
            None => (),
            Some(LintLevel::Forbid) if level < LintLevel::Forbid => {
                emit_error!(
                    meta.span(),
                    "[rust-cuda]: {}(ptx::{}) incompatible with previous forbid.",
                    level,
                    lint,
                );
                continue;
            },
            Some(previous) => {
                emit_warning!(
                    meta.span(),
                    "[rust-cuda]: {}(ptx::{}) overwrites previous {}.",
                    level,
                    lint,
                    previous,
                );
            },
        }

        ptx_lint_levels.insert(lint, level);
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum LintLevel {
    Allow,
    Warn,
    Deny,
    Forbid,
}

impl fmt::Display for LintLevel {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Allow => fmt.write_str("allow"),
            Self::Warn => fmt.write_str("warn"),
            Self::Deny => fmt.write_str("deny"),
            Self::Forbid => fmt.write_str("forbid"),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum PtxLint {
    Verbose,
    DoublePrecisionUse,
    LocalMemoryUsage,
    RegisterSpills,
}

impl fmt::Display for PtxLint {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Verbose => fmt.write_str("verbose"),
            Self::DoublePrecisionUse => fmt.write_str("double_precision_use"),
            Self::LocalMemoryUsage => fmt.write_str("local_memory_usage"),
            Self::RegisterSpills => fmt.write_str("register_spills"),
        }
    }
}
