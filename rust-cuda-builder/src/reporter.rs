use std::{fmt, process::exit};

use colored::{control, Colorize};

use crate::{
    builder::{BuildStatus, Builder},
    error::{Error, Result},
};

/// Cargo integration adapter.
///
/// Provides PTX assembly path to Rust through specified environment variable
/// name and informs Cargo about device crate dependencies, so it can rebuild on
/// changes.
///
/// # Usage in `build.rs`
/// ```no_run
/// use ptx_builder::error::Result;
/// use ptx_builder::prelude::*;
///
/// fn main() -> Result<()> {
///     CargoAdapter::with_env_var("PTX_PATH").build(Builder::new(".")?);
/// }
/// ```
pub struct CargoAdapter {
    env_name: String,
}

impl CargoAdapter {
    /// Creates an instance of the adapter that will provide PTX assembly path
    /// to Rust via `env_name` environment variable.
    ///
    /// The PTX assembly can later be used **in host crate**:
    /// ```ignore
    /// use std::ffi::CString;
    ///
    /// let ptx = CString::new(include_str!(env!("PTX_PATH")))?;
    /// ```
    pub fn with_env_var<S: AsRef<str>>(env_name: S) -> Self {
        CargoAdapter {
            env_name: env_name.as_ref().to_string(),
        }
    }

    /// Runs build process and reports artifacts to Cargo.
    ///
    /// Depends on whether the build was successful or not, will either
    /// call `exit(0)` or `exit(1)` and print error log to `stderr`.
    #[allow(clippy::needless_pass_by_value)]
    pub fn build(&self, builder: Builder) -> ! {
        if let Err(error) = self.build_inner(&builder) {
            eprintln!("{}", ErrorLogPrinter::print(error));
            exit(1);
        } else {
            exit(0);
        }
    }

    fn build_inner(&self, builder: &Builder) -> Result<()> {
        match builder.build()? {
            BuildStatus::Success(output) => {
                let dependencies = output.dependencies()?;

                println!(
                    "cargo:rustc-env={}={}",
                    self.env_name,
                    output.get_assembly_path().display()
                );

                for path in dependencies {
                    println!("cargo:rerun-if-changed={}", path.display());
                }
            },

            BuildStatus::NotNeeded => {
                println!("cargo:rustc-env={}=/dev/null", self.env_name);
            },
        };

        Ok(())
    }
}

/// Nice error log printer.
///
/// ```no_run
/// use std::process::exit;
/// use ptx_builder::prelude::*;
/// # use ptx_builder::error::Result;
///
/// fn main() {
///     if let Err(error) = build() {
///         eprintln!("{}", ErrorLogPrinter::print(error));
///         exit(1);
///    }
/// }
/// # fn build() -> Result<()> {
/// #    use ptx_builder::error::*;
/// #    Err(BuildErrorKind::InternalError("any...".into()).into())
/// # }
pub struct ErrorLogPrinter {
    error: Error,
    colors: bool,
}

impl ErrorLogPrinter {
    /// Creates instance of the printer.
    #[must_use]
    pub fn print(error: Error) -> Self {
        Self {
            error,
            colors: true,
        }
    }

    /// Controls whether colors should be used in the error log.
    pub fn disable_colors(&mut self) -> &mut Self {
        self.colors = false;
        self
    }
}

trait StringExt {
    fn prefix_each_line<T>(self, prefix: T) -> Self
    where
        T: ToString;
}

impl StringExt for String {
    fn prefix_each_line<T: ToString>(self, prefix: T) -> Self {
        let owned_prefix = prefix.to_string();
        let glue = String::from("\n") + &owned_prefix;

        owned_prefix + &self.split('\n').collect::<Vec<_>>().join(&glue)
    }
}

impl fmt::Display for ErrorLogPrinter {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        control::set_override(self.colors);

        fmt.write_str(
            &format!("{:?}", self.error).prefix_each_line("[PTX] ".bright_black().bold()),
        )?;

        control::unset_override();
        Ok(())
    }
}
