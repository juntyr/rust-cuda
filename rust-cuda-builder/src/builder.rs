use std::{
    collections::HashMap,
    env,
    ffi::OsString,
    fmt,
    fs::{read_to_string, write, File},
    io::{BufReader, Read},
    path::{Path, PathBuf},
    sync::LazyLock,
};

use regex::Regex;

use crate::{
    error::{BuildErrorKind, Error, Result, ResultExt},
    executable::{Cargo, ExecutableRunner},
    source::Crate,
};

const LAST_BUILD_CMD: &str = ".last-build-command";
const TARGET_NAME: &str = "nvptx64-nvidia-cuda";

/// Core of the crate - PTX assembly build controller.
#[derive(Debug)]
pub struct Builder {
    source_crate: Crate,

    profile: Profile,
    colors: bool,
    crate_type: Option<CrateType>,
    message_format: MessageFormat,
    prefix: String,

    env: HashMap<OsString, OsString>,
}

/// Successful build output.
#[derive(Debug)]
pub struct BuildOutput<'a> {
    builder: &'a Builder,
    output_path: PathBuf,
    file_suffix: String,
}

/// Non-failed build status.
#[derive(Debug)]
pub enum BuildStatus<'a> {
    /// The CUDA crate building was performed without errors.
    Success(BuildOutput<'a>),

    /// The CUDA crate building is not needed. Can happend in several cases:
    /// - `build.rs` script was called by **RLS**,
    /// - `build.rs` was called **recursively** (e.g. `build.rs` call for device
    ///   crate in single-source setup)
    NotNeeded,
}

/// Debug / Release profile.
///
/// # Usage
/// ``` no_run
/// use ptx_builder::prelude::*;
/// # use ptx_builder::error::Result;
///
/// # fn main() -> Result<()> {
/// Builder::new(".")?
///     .set_profile(Profile::Debug)
///     .build()?;
/// # Ok(())
/// # }
/// ```
#[derive(PartialEq, Eq, Clone, Debug)]
pub enum Profile {
    /// Equivalent for `cargo-build` **without** `--release` flag.
    Debug,

    /// Equivalent for `cargo-build` **with** `--release` flag.
    Release,
}

/// Message format.
///
/// # Usage
/// ``` no_run
/// use ptx_builder::prelude::*;
/// # use ptx_builder::error::Result;
///
/// # fn main() -> Result<()> {
/// Builder::new(".")?
///     .set_message_format(MessageFormat::Short)
///     .build()?;
/// # Ok(())
/// # }
/// ```
#[derive(PartialEq, Eq, Clone, Debug)]
pub enum MessageFormat {
    /// Equivalent for `cargo-build` with `--message-format=human` flag
    /// (default).
    Human,

    /// Equivalent for `cargo-build` with `--message-format=json` flag
    Json {
        /// Whether rustc diagnostics are rendered by cargo or included into the
        /// output stream.
        render_diagnostics: bool,
        /// Whether the `rendered` field of rustc diagnostics are using the
        /// "short" rendering.
        short: bool,
        /// Whether the `rendered` field of rustc diagnostics embed ansi color
        /// codes.
        ansi: bool,
    },

    /// Equivalent for `cargo-build` with `--message-format=short` flag
    Short,
}

/// Build specified crate type.
///
/// Mandatory for mixed crates - that have both `lib.rs` and `main.rs`,
/// otherwise Cargo won't know which to build:
/// ```text
/// error: extra arguments to `rustc` can only be passed to one target, consider filtering
/// the package by passing e.g. `--lib` or `--bin NAME` to specify a single target
/// ```
///
/// # Usage
/// ``` no_run
/// use ptx_builder::prelude::*;
/// # use ptx_builder::error::Result;
///
/// # fn main() -> Result<()> {
/// Builder::new(".")?
///     .set_crate_type(CrateType::Library)
///     .build()?;
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Copy, Debug)]
pub enum CrateType {
    Library,
    Binary,
}

impl Builder {
    /// Construct a builder for device crate at `path`.
    ///
    /// Can also be the same crate, for single-source mode:
    /// ``` no_run
    /// use ptx_builder::prelude::*;
    /// # use ptx_builder::error::Result;
    ///
    /// # fn main() -> Result<()> {
    /// match Builder::new(".")?.build()? {
    ///     BuildStatus::Success(output) => {
    ///         // do something with the output...
    ///     }
    ///
    ///     BuildStatus::NotNeeded => {
    ///         // ...
    ///     }
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self> {
        Ok(Builder {
            source_crate: Crate::analyse(path).context("Unable to analyse source crate")?,
            // TODO: choose automatically, e.g.:
            // `env::var("PROFILE").unwrap_or("release".to_string())`
            profile: Profile::Release,
            colors: true,
            crate_type: None,
            message_format: MessageFormat::Human,
            prefix: String::new(),
            env: HashMap::new(),
        })
    }

    /// Returns bool indicating whether the actual build is needed.
    ///
    /// Behavior is consistent with
    /// [`BuildStatus::NotNeeded`](enum.BuildStatus.html#variant.NotNeeded).
    #[must_use]
    pub fn is_build_needed() -> bool {
        let recursive_env = env::var("PTX_CRATE_BUILDING");

        let is_recursive_build = recursive_env.map_or(false, |recursive_env| recursive_env == "1");

        !is_recursive_build
    }

    /// Returns the name of the source crate at the construction `path`.
    #[must_use]
    pub fn get_crate_name(&self) -> &str {
        self.source_crate.get_name()
    }

    /// Disable colors for internal calls to `cargo`.
    #[must_use]
    pub fn disable_colors(mut self) -> Self {
        self.colors = false;
        self
    }

    /// Set build profile.
    #[must_use]
    pub fn set_profile(mut self, profile: Profile) -> Self {
        self.profile = profile;
        self
    }

    /// Set crate type that needs to be built.
    ///
    /// Mandatory for mixed crates - that have both `lib.rs` and `main.rs`,
    /// otherwise Cargo won't know which to build:
    /// ```text
    /// error: extra arguments to `rustc` can only be passed to one target, consider filtering
    /// the package by passing e.g. `--lib` or `--bin NAME` to specify a single target
    /// ```
    #[must_use]
    pub fn set_crate_type(mut self, crate_type: CrateType) -> Self {
        self.crate_type = Some(crate_type);
        self
    }

    /// Set the message format.
    #[must_use]
    pub fn set_message_format(mut self, message_format: MessageFormat) -> Self {
        self.message_format = message_format;
        self
    }

    /// Set the build command prefix.
    #[must_use]
    pub fn set_prefix(mut self, prefix: String) -> Self {
        self.prefix = prefix;
        self
    }

    /// Inserts or updates an environment variable for the build process.
    #[must_use]
    pub fn with_env<K: Into<OsString>, V: Into<OsString>>(mut self, key: K, val: V) -> Self {
        self.env.insert(key.into(), val.into());
        self
    }

    /// Performs an actual build: runs `cargo` with proper flags and
    /// environment.
    pub fn build(&self) -> Result<BuildStatus> {
        self.build_live(|_line| (), |_line| ())
    }

    /// Performs an actual build: runs `cargo` with proper flags and
    /// environment.
    pub fn build_live<O: FnMut(&str), E: FnMut(&str)>(
        &self,
        on_stdout_line: O,
        mut on_stderr_line: E,
    ) -> Result<BuildStatus> {
        if !Self::is_build_needed() {
            return Ok(BuildStatus::NotNeeded);
        }

        let mut cargo = ExecutableRunner::new(Cargo);
        let mut args = vec!["rustc"];

        if self.profile == Profile::Release {
            args.push("--release");
        }

        args.push("--color");
        args.push(if self.colors { "always" } else { "never" });

        let mut json_format = String::from("--message-format=json");
        args.push(match self.message_format {
            MessageFormat::Human => "--message-format=human",
            MessageFormat::Json {
                render_diagnostics,
                short,
                ansi,
            } => {
                if render_diagnostics {
                    json_format.push_str(",json-render-diagnostics");
                }

                if short {
                    json_format.push_str(",json-diagnostic-short");
                }

                if ansi {
                    json_format.push_str(",json-diagnostic-rendered-ansi");
                }

                &json_format
            },
            MessageFormat::Short => "--message-format=short",
        });

        args.push("--target");
        args.push(TARGET_NAME);

        match self.crate_type {
            Some(CrateType::Binary) => {
                args.push("--bin");
                args.push(self.source_crate.get_name());
            },

            Some(CrateType::Library) => {
                args.push("--lib");
            },

            _ => {},
        }

        args.push("-v");

        let crate_type = self.source_crate.get_crate_type(self.crate_type)?;

        args.push("--");

        args.push("--crate-type");
        args.push(crate_type);

        let output_path = {
            self.source_crate
                .get_output_path()
                .context("Unable to create output path")?
        };

        cargo
            .with_args(&args)
            .with_cwd(self.source_crate.get_path())
            .with_env("PTX_CRATE_BUILDING", "1")
            .with_env("CARGO_TARGET_DIR", output_path.clone());

        for (key, val) in &self.env {
            cargo.with_env(key, val);
        }

        let cargo_output = cargo
            .run_live(on_stdout_line, |line| {
                if Self::output_is_not_verbose(line) {
                    on_stderr_line(line);
                }
            })
            .map_err(|error| match error.kind() {
                BuildErrorKind::CommandFailed { stderr, .. } => {
                    #[allow(clippy::manual_filter_map)]
                    let lines = stderr
                        .trim_matches('\n')
                        .split('\n')
                        .filter(|s| Self::output_is_not_verbose(s))
                        .map(String::from)
                        .collect();

                    Error::from(BuildErrorKind::BuildFailed(lines))
                },
                _ => error,
            })?;

        Ok(BuildStatus::Success(self.prepare_output(
            output_path,
            &cargo_output.stderr,
            crate_type,
        )?))
    }

    fn prepare_output(
        &self,
        output_path: PathBuf,
        cargo_stderr: &str,
        crate_type: &str,
    ) -> Result<BuildOutput> {
        static SUFFIX_REGEX: LazyLock<Regex> = LazyLock::new(|| {
            Regex::new(r"-C extra-filename=([\S]+)").expect("Unable to parse regex...")
        });

        let crate_name = self.source_crate.get_output_file_prefix();

        // We need the build command to get real output filename.
        let build_command = {
            #[allow(clippy::manual_find_map)]
            cargo_stderr
                .trim_matches('\n')
                .split('\n')
                .find(|line| {
                    line.contains(&format!("--crate-name {crate_name}"))
                        && line.contains(&format!("--crate-type {crate_type}"))
                })
                .map(|line| BuildCommand::Realtime(line.to_string()))
                .or_else(|| Self::load_cached_build_command(&output_path, &self.prefix))
                .ok_or_else(|| {
                    Error::from(BuildErrorKind::InternalError(String::from(
                        "Unable to find build command of the device crate",
                    )))
                })?
        };

        if let BuildCommand::Realtime(ref command) = build_command {
            Self::store_cached_build_command(&output_path, &self.prefix, command)?;
        }

        let (file_suffix, found_suffix) = match SUFFIX_REGEX.captures(&build_command) {
            Some(caps) => (caps[1].to_string(), true),
            None => (String::new(), false),
        };

        let output = BuildOutput::new(self, output_path, file_suffix);

        if output.get_assembly_path().exists() {
            Ok(output)
        } else if found_suffix {
            Err(BuildErrorKind::InternalError(String::from(
                "Unable to find PTX assembly as specified by `extra-filename` rustc flag",
            ))
            .into())
        } else {
            Err(BuildErrorKind::InternalError(String::from(
                "Unable to find `extra-filename` rustc flag",
            ))
            .into())
        }
    }

    fn output_is_not_verbose(line: &str) -> bool {
        !line.starts_with("+ ")
            && !line.contains("Running")
            && !line.contains("Fresh")
            && !line.starts_with("Caused by:")
            && !line.starts_with("  process didn\'t exit successfully: ")
    }

    fn load_cached_build_command(output_path: &Path, prefix: &str) -> Option<BuildCommand> {
        match read_to_string(output_path.join(format!("{LAST_BUILD_CMD}.{prefix}"))) {
            Ok(contents) => Some(BuildCommand::Cached(contents)),
            Err(_) => None,
        }
    }

    fn store_cached_build_command(output_path: &Path, prefix: &str, command: &str) -> Result<()> {
        write(
            output_path.join(format!("{LAST_BUILD_CMD}.{prefix}")),
            command.as_bytes(),
        )
        .context(BuildErrorKind::OtherError)?;

        Ok(())
    }
}

impl<'a> BuildOutput<'a> {
    fn new(builder: &'a Builder, output_path: PathBuf, file_suffix: String) -> Self {
        BuildOutput {
            builder,
            output_path,
            file_suffix,
        }
    }

    /// Returns path to PTX assembly file.
    ///
    /// # Usage
    /// Can be used from `build.rs` script to provide Rust with the path
    /// via environment variable:
    /// ```no_run
    /// use ptx_builder::prelude::*;
    /// # use ptx_builder::error::Result;
    ///
    /// # fn main() -> Result<()> {
    /// if let BuildStatus::Success(output) = Builder::new(".")?.build()? {
    ///     println!(
    ///         "cargo:rustc-env=KERNEL_PTX_PATH={}",
    ///         output.get_assembly_path().display()
    ///     );
    /// }
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    pub fn get_assembly_path(&self) -> PathBuf {
        self.output_path
            .join(TARGET_NAME)
            .join(self.builder.profile.to_string())
            .join("deps")
            .join(format!(
                "{}{}.ptx",
                self.builder.source_crate.get_output_file_prefix(),
                self.file_suffix,
            ))
    }

    /// Returns a list of crate dependencies.
    ///
    /// # Usage
    /// Can be used from `build.rs` script to notify Cargo the dependencies,
    /// so it can automatically rebuild on changes:
    /// ```no_run
    /// use ptx_builder::prelude::*;
    /// # use ptx_builder::error::Result;
    ///
    /// # fn main() -> Result<()> {
    /// if let BuildStatus::Success(output) = Builder::new(".")?.build()? {
    ///     for path in output.dependencies()? {
    ///         println!("cargo:rerun-if-changed={}", path.display());
    ///     }
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn dependencies(&self) -> Result<Vec<PathBuf>> {
        let mut deps_contents = {
            self.get_deps_file_contents()
                .context("Unable to get crate deps")?
        };

        if deps_contents.is_empty() {
            bail!(BuildErrorKind::InternalError(String::from(
                "Empty deps file",
            )));
        }

        deps_contents = deps_contents
            .chars()
            .skip(3) // workaround for Windows paths starts wuth "[A-Z]:\"
            .skip_while(|c| *c != ':')
            .skip(1)
            .collect::<String>();

        let mut cargo_lock_dir = self.builder.source_crate.get_path();

        // Traverse the workspace directory structure towards the root
        while !cargo_lock_dir.join("Cargo.lock").is_file() {
            cargo_lock_dir = match cargo_lock_dir.parent() {
                Some(parent) => parent,
                None => bail!(BuildErrorKind::InternalError(String::from(
                    "Unable to find Cargo.lock file",
                ))),
            }
        }

        let cargo_deps = vec![
            self.builder.source_crate.get_path().join("Cargo.toml"),
            cargo_lock_dir.join("Cargo.lock"),
        ];

        Ok(deps_contents
            .trim()
            .split(' ')
            .map(|item| PathBuf::from(item.trim()))
            .chain(cargo_deps)
            .collect())
    }

    fn get_deps_file_contents(&self) -> Result<String> {
        let crate_deps_path = self
            .output_path
            .join(TARGET_NAME)
            .join(self.builder.profile.to_string())
            .join(format!(
                "{}.d",
                self.builder
                    .source_crate
                    .get_deps_file_prefix(self.builder.crate_type)?
            ));

        let mut crate_deps_reader =
            BufReader::new(File::open(crate_deps_path).context(BuildErrorKind::OtherError)?);

        let mut crate_deps_contents = String::new();

        crate_deps_reader
            .read_to_string(&mut crate_deps_contents)
            .context(BuildErrorKind::OtherError)?;

        Ok(crate_deps_contents)
    }
}

impl fmt::Display for Profile {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Profile::Debug => write!(fmt, "debug"),
            Profile::Release => write!(fmt, "release"),
        }
    }
}

enum BuildCommand {
    Realtime(String),
    Cached(String),
}

impl std::ops::Deref for BuildCommand {
    type Target = str;

    fn deref(&self) -> &str {
        match self {
            BuildCommand::Realtime(line) | BuildCommand::Cached(line) => line,
        }
    }
}
