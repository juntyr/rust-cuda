use std::{fmt, path::PathBuf};

use colored::Colorize;
use semver::{Version, VersionReq};

#[macro_export]
macro_rules! bail {
    ($err:expr) => {
        return Err($err.into())
    };
}

#[derive(thiserror::Error)]
pub struct Error {
    #[source]
    error: anyhow::Error,
    context: BuildErrorKind,
}

impl fmt::Debug for Error {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        self.error.fmt(fmt)
    }
}

impl fmt::Display for Error {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        self.error.fmt(fmt)
    }
}

impl From<BuildErrorKind> for Error {
    fn from(context: BuildErrorKind) -> Self {
        Self {
            error: anyhow::Error::new(context.clone()),
            context,
        }
    }
}

impl Error {
    #[must_use]
    pub fn kind(&self) -> &BuildErrorKind {
        &self.context
    }

    #[must_use]
    pub fn context(self, context: BuildErrorKind) -> Self {
        Self {
            error: self.error.context(context.clone()),
            context,
        }
    }
}

pub(crate) trait ResultExt<T, C> {
    fn context(self, context: C) -> Result<T, Error>;

    fn with_context<F: FnOnce() -> C>(self, f: F) -> Result<T, Error>;
}

impl<T, E: std::error::Error + Send + Sync + 'static> ResultExt<T, BuildErrorKind>
    for Result<T, E>
{
    fn context(self, context: BuildErrorKind) -> Result<T, Error> {
        anyhow::Context::with_context(self, || context.clone())
            .map_err(|error| Error { error, context })
    }

    fn with_context<F: FnOnce() -> BuildErrorKind>(self, f: F) -> Result<T, Error> {
        if let Ok(val) = self {
            return Ok(val);
        }

        self.context(f())
    }
}

impl<'a, T, E: std::error::Error + Send + Sync + 'static> ResultExt<T, &'a str> for Result<T, E> {
    fn context(self, context: &'a str) -> Result<T, Error> {
        self.with_context(|| BuildErrorKind::InternalError(String::from(context)))
    }

    fn with_context<F: FnOnce() -> &'a str>(self, f: F) -> Result<T, Error> {
        self.with_context(|| BuildErrorKind::InternalError(String::from(f())))
    }
}

pub type Result<T, E = Error> = std::result::Result<T, E>;

#[derive(Debug, PartialEq, Eq, thiserror::Error, Clone)]
pub enum BuildErrorKind {
    CommandNotFound {
        command: String,
        hint: String,
    },

    CommandFailed {
        command: String,
        code: i32,
        stderr: String,
    },
    CommandVersionNotFulfilled {
        command: String,
        current: Version,
        required: VersionReq,
        hint: String,
    },

    InvalidCratePath(PathBuf),
    BuildFailed(Vec<String>),
    InvalidCrateType(String),
    MissingCrateType,
    InternalError(String),
    OtherError,
}

impl fmt::Display for BuildErrorKind {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        use BuildErrorKind::{
            BuildFailed, CommandFailed, CommandNotFound, CommandVersionNotFulfilled, InternalError,
            InvalidCratePath, InvalidCrateType, MissingCrateType, OtherError,
        };

        match self {
            CommandNotFound { command, hint } => write!(
                fmt,
                "Command not found in PATH: '{}'. {}.",
                command.bold(),
                hint.underline()
            ),

            CommandFailed {
                command,
                code,
                stderr,
            } => write!(
                fmt,
                "Command failed: '{}' with code '{}' and output:\n{}",
                command.bold(),
                code,
                stderr.trim(),
            ),

            CommandVersionNotFulfilled {
                command,
                current,
                required,
                hint,
            } => write!(
                fmt,
                "Command version is not fulfilled: '{}' is currently '{}' but '{}' is required. \
                 {}.",
                command.bold(),
                current.to_string().underline(),
                required.to_string().underline(),
                hint.underline(),
            ),

            InvalidCratePath(path) => write!(
                fmt,
                "{}: {}",
                "Invalid device crate path".bold(),
                path.display()
            ),

            BuildFailed(lines) => write!(
                fmt,
                "{}\n{}",
                "Unable to build a PTX crate!".bold(),
                lines.join("\n")
            ),

            InvalidCrateType(crate_type) => write!(
                fmt,
                "{}: the crate cannot be build as '{}'",
                "Impossible CrateType".bold(),
                crate_type
            ),

            MissingCrateType => write!(
                fmt,
                "{}: it's mandatory for mixed-type crates",
                "Missing CrateType".bold()
            ),

            InternalError(message) => write!(fmt, "{}: {}", "Internal error".bold(), message),
            OtherError => write!(fmt, "Other error"),
        }
    }
}
