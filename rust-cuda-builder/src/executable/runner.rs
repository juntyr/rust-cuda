use std::{ffi::OsStr, path::Path, process::Command};

use regex::Regex;
use semver::Version;

use crate::error::{BuildErrorKind, Error, Result, ResultExt};

use super::{process::streaming_output, Executable};

#[allow(clippy::module_name_repetitions)]
pub struct ExecutableRunner<Ex: Executable> {
    command: Command,
    executable: Ex,
}

#[derive(Debug)]
pub struct Output {
    pub stdout: String,
    pub stderr: String,
}

impl<Ex: Executable> ExecutableRunner<Ex> {
    pub fn new(executable: Ex) -> Self {
        ExecutableRunner {
            command: Command::new(executable.get_name()),
            executable,
        }
    }

    pub fn with_args<I, S>(&mut self, args: I) -> &mut Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<OsStr>,
    {
        self.command.args(args);
        self
    }

    pub fn with_env<K, V>(&mut self, key: K, val: V) -> &mut Self
    where
        K: AsRef<OsStr>,
        V: AsRef<OsStr>,
    {
        self.command.env(key, val);
        self
    }

    pub fn with_cwd<P>(&mut self, path: P) -> &mut Self
    where
        P: AsRef<Path>,
    {
        self.command.current_dir(path);
        self
    }

    pub fn run(&mut self) -> Result<Output> {
        self.check_version()?;

        let raw_output = {
            self.command.output().with_context(|| {
                BuildErrorKind::InternalError(format!(
                    "Unable to execute command '{}'",
                    self.executable.get_name()
                ))
            })?
        };

        let output = Output {
            stdout: String::from_utf8(raw_output.stdout).context(BuildErrorKind::OtherError)?,
            stderr: String::from_utf8(raw_output.stderr).context(BuildErrorKind::OtherError)?,
        };

        if raw_output.status.success() {
            Ok(output)
        } else {
            Err(Error::from(BuildErrorKind::CommandFailed {
                command: self.executable.get_name(),
                code: raw_output.status.code().unwrap_or(-1),
                stderr: output.stderr,
            }))
        }
    }

    pub fn run_live<O: FnMut(&str), E: FnMut(&str)>(
        &mut self,
        on_stdout_line: O,
        on_stderr_line: E,
    ) -> Result<Output> {
        self.check_version()?;

        let raw_output = streaming_output(&mut self.command, on_stdout_line, on_stderr_line)
            .with_context(|| {
                BuildErrorKind::InternalError(format!(
                    "Unable to execute command '{}'",
                    self.executable.get_name()
                ))
            })?;

        let output = Output {
            stdout: String::from_utf8(raw_output.stdout).context(BuildErrorKind::OtherError)?,
            stderr: String::from_utf8(raw_output.stderr).context(BuildErrorKind::OtherError)?,
        };

        if raw_output.status.success() {
            Ok(output)
        } else {
            Err(Error::from(BuildErrorKind::CommandFailed {
                command: self.executable.get_name(),
                code: raw_output.status.code().unwrap_or(-1),
                stderr: output.stderr,
            }))
        }
    }

    fn check_version(&self) -> Result<()> {
        let current = self.executable.get_current_version()?;
        let required = self.executable.get_required_version();

        match required {
            Some(ref required) if !required.matches(&current) => {
                Err(Error::from(BuildErrorKind::CommandVersionNotFulfilled {
                    command: self.executable.get_name(),
                    current,
                    required: required.clone(),
                    hint: self.executable.get_version_hint(),
                }))
            },

            _ => Ok(()),
        }
    }
}

pub(crate) fn parse_executable_version<E: Executable>(executable: &E) -> Result<Version> {
    let mut command = Command::new(executable.get_name());

    command.args(["-V"]);

    let raw_output = {
        command
            .output()
            .with_context(|| BuildErrorKind::CommandNotFound {
                command: executable.get_name(),
                hint: executable.get_verification_hint(),
            })?
    };

    let output = Output {
        stdout: String::from_utf8(raw_output.stdout).context(BuildErrorKind::OtherError)?,
        stderr: String::from_utf8(raw_output.stderr).context(BuildErrorKind::OtherError)?,
    };

    if !raw_output.status.success() {
        bail!(BuildErrorKind::CommandFailed {
            command: executable.get_name(),
            code: raw_output.status.code().unwrap_or(-1),
            stderr: output.stderr,
        });
    }

    let version_regex = Regex::new(&format!(r"{}\s(\S+)", executable.get_name()))
        .context(BuildErrorKind::OtherError)?;

    match version_regex.captures(&(output.stdout + &output.stderr)) {
        Some(captures) => Ok(Version::parse(&captures[1]).context(BuildErrorKind::OtherError)?),

        None => Err(Error::from(BuildErrorKind::InternalError(
            "Unable to find executable version".into(),
        ))),
    }
}
