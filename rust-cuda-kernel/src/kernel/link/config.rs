use std::{collections::HashMap, path::PathBuf};

use crate::kernel::lints::{parse_ptx_lint_level, LintLevel, PtxLint};

#[allow(clippy::module_name_repetitions)]
pub(super) struct LinkKernelConfig {
    pub(super) kernel: syn::Ident,
    pub(super) kernel_hash: syn::Ident,
    pub(super) crate_name: String,
    pub(super) crate_path: PathBuf,
    pub(super) specialisation: String,
    pub(super) ptx_lint_levels: HashMap<PtxLint, LintLevel>,
}

impl syn::parse::Parse for LinkKernelConfig {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let kernel: syn::Ident = input.parse()?;
        let kernel_hash: syn::Ident = input.parse()?;
        let name: syn::LitStr = input.parse()?;
        let path: syn::LitStr = input.parse()?;

        let specialisation = if input.parse::<Option<syn::token::Lt>>()?.is_some() {
            if input.parse::<Option<syn::token::Gt>>()?.is_some() {
                String::new()
            } else {
                let specialisation_types = syn::punctuated::Punctuated::<
                    syn::Type,
                    syn::token::Comma,
                >::parse_separated_nonempty(input)?;

                let _gt_token: syn::token::Gt = input.parse()?;

                (quote! { <#specialisation_types> })
                    .to_string()
                    .replace(&[' ', '\n', '\t'][..], "")
            }
        } else {
            String::new()
        };

        let attrs = syn::punctuated::Punctuated::<
            syn::MetaList,
            syn::token::Comma,
        >::parse_separated_nonempty(input)?;

        let mut ptx_lint_levels = HashMap::new();

        for syn::MetaList { path, nested, .. } in attrs {
            parse_ptx_lint_level(&path, &nested, &mut ptx_lint_levels);
        }

        proc_macro_error::abort_if_dirty();

        Ok(Self {
            kernel,
            kernel_hash,
            crate_name: name.value(),
            crate_path: PathBuf::from(path.value()),
            specialisation,
            ptx_lint_levels,
        })
    }
}

#[allow(clippy::module_name_repetitions)]
pub(super) struct CheckKernelConfig {
    pub(super) kernel: syn::Ident,
    pub(super) kernel_hash: syn::Ident,
    pub(super) crate_name: String,
    pub(super) crate_path: PathBuf,
}

impl syn::parse::Parse for CheckKernelConfig {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let kernel: syn::Ident = input.parse()?;
        let kernel_hash: syn::Ident = input.parse()?;
        let name: syn::LitStr = input.parse()?;
        let path: syn::LitStr = input.parse()?;

        Ok(Self {
            kernel,
            kernel_hash,
            crate_name: name.value(),
            crate_path: PathBuf::from(path.value()),
        })
    }
}
