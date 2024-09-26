use std::{collections::HashMap, fmt};

use syn::spanned::Spanned;

#[expect(clippy::too_many_lines)]
pub fn parse_ptx_lint_level(
    meta: &impl NestedMetaParser,
    ptx_lint_levels: &mut HashMap<PtxLint, LintLevel>,
) {
    let level = match meta.path().get_ident() {
        Some(ident) if ident == "allow" => LintLevel::Allow,
        Some(ident) if ident == "warn" => LintLevel::Warn,
        Some(ident) if ident == "deny" => LintLevel::Deny,
        Some(ident) if ident == "forbid" => LintLevel::Forbid,
        _ => {
            emit_error!(
                meta.path().span(),
                "[rust-cuda]: Invalid lint #[kernel(<level>(<lint>))] attribute: unknown lint \
                 level, must be one of `allow`, `warn`, `deny`, `forbid`.",
            );

            return;
        },
    };

    if meta
        .parse_nested_meta(|meta| {
            if meta.path.leading_colon.is_some()
                || meta.path.segments.empty_or_trailing()
                || meta.path.segments.len() != 2
            {
                emit_error!(
                    meta.path.span(),
                    "[rust-cuda]: Invalid #[kernel({}(<lint>))] attribute: <lint> must be of the \
                     form `ptx::lint`.",
                    level,
                );
                return Ok(());
            }

            let Some(syn::PathSegment {
                ident: namespace,
                arguments: syn::PathArguments::None,
            }) = meta.path.segments.first()
            else {
                emit_error!(
                    meta.path.span(),
                    "[rust-cuda]: Invalid #[kernel({}(<lint>))] attribute: <lint> must be of the \
                     form `ptx::lint`.",
                    level,
                );
                return Ok(());
            };

            if namespace != "ptx" {
                emit_error!(
                    meta.path.span(),
                    "[rust-cuda]: Invalid #[kernel({}(<lint>))] attribute: <lint> must be of the \
                     form `ptx::lint`.",
                    level,
                );
                return Ok(());
            }

            let Some(syn::PathSegment {
                ident: lint,
                arguments: syn::PathArguments::None,
            }) = meta.path.segments.last()
            else {
                emit_error!(
                    meta.path.span(),
                    "[rust-cuda]: Invalid #[kernel({}(<lint>))] attribute: <lint> must be of the \
                     form `ptx::lint`.",
                    level,
                );
                return Ok(());
            };

            let lint = match lint {
                l if l == "verbose" => PtxLint::Verbose,
                l if l == "double_precision_use" => PtxLint::DoublePrecisionUse,
                l if l == "local_memory_use" => PtxLint::LocalMemoryUse,
                l if l == "register_spills" => PtxLint::RegisterSpills,
                l if l == "dump_assembly" => PtxLint::DumpAssembly,
                l if l == "dynamic_stack_size" => PtxLint::DynamicStackSize,
                _ => {
                    emit_error!(
                        meta.path.span(),
                        "[rust-cuda]: Unknown PTX kernel lint `ptx::{}`.",
                        lint,
                    );
                    return Ok(());
                },
            };

            match ptx_lint_levels.get(&lint) {
                None => (),
                Some(LintLevel::Forbid) if level < LintLevel::Forbid => {
                    emit_error!(
                        meta.path.span(),
                        "[rust-cuda]: {}(ptx::{}) incompatible with previous forbid.",
                        level,
                        lint,
                    );
                    return Ok(());
                },
                Some(previous) => {
                    emit_warning!(
                        meta.path.span(),
                        "[rust-cuda]: {}(ptx::{}) overwrites previous {}.",
                        level,
                        lint,
                        previous,
                    );
                },
            }

            ptx_lint_levels.insert(lint, level);

            Ok(())
        })
        .is_err()
    {
        emit_error!(
            meta.path().span(),
            "[rust-cuda]: Invalid #[kernel({}(<lint>))] attribute.",
            level,
        );
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
    LocalMemoryUse,
    RegisterSpills,
    DumpAssembly,
    DynamicStackSize,
}

impl fmt::Display for PtxLint {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Verbose => fmt.write_str("verbose"),
            Self::DoublePrecisionUse => fmt.write_str("double_precision_use"),
            Self::LocalMemoryUse => fmt.write_str("local_memory_use"),
            Self::RegisterSpills => fmt.write_str("register_spills"),
            Self::DumpAssembly => fmt.write_str("dump_assembly"),
            Self::DynamicStackSize => fmt.write_str("dynamic_stack_size"),
        }
    }
}

pub trait NestedMetaParser {
    fn path(&self) -> &syn::Path;

    fn parse_nested_meta(
        &self,
        logic: impl FnMut(syn::meta::ParseNestedMeta) -> syn::Result<()>,
    ) -> syn::Result<()>;
}

impl<'a> NestedMetaParser for syn::meta::ParseNestedMeta<'a> {
    fn path(&self) -> &syn::Path {
        &self.path
    }

    fn parse_nested_meta(
        &self,
        logic: impl FnMut(syn::meta::ParseNestedMeta) -> syn::Result<()>,
    ) -> syn::Result<()> {
        self.parse_nested_meta(logic)
    }
}

impl NestedMetaParser for syn::MetaList {
    fn path(&self) -> &syn::Path {
        &self.path
    }

    fn parse_nested_meta(
        &self,
        logic: impl FnMut(syn::meta::ParseNestedMeta) -> syn::Result<()>,
    ) -> syn::Result<()> {
        self.parse_nested_meta(logic)
    }
}
