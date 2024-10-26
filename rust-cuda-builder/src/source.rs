use std::{
    collections::hash_map::DefaultHasher,
    env, fs,
    hash::{Hash, Hasher},
    io::{BufReader, Read},
    path::{Path, PathBuf},
};

use crate::{
    builder::CrateType,
    error::{BuildErrorKind, Result, ResultExt},
};

#[derive(Hash, Clone, Debug)]
pub enum FilePrefix {
    Library(String),
    Binary(String),
    Mixed { lib: String, bin: String },
}

#[derive(Hash, Clone, Debug)]
/// Information about CUDA crate.
pub struct Crate {
    name: String,
    path: PathBuf,
    output_file_prefix: String,
    deps_file_prefix: FilePrefix,
}

impl Crate {
    /// Try to locate a crate at the `path` and collect needed information.
    pub fn analyse<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = {
            env::current_dir()
                .context(BuildErrorKind::OtherError)?
                .join(&path)
        };

        match fs::metadata(path.join("Cargo.toml")) {
            Ok(metadata) => {
                if metadata.is_dir() {
                    bail!(BuildErrorKind::InvalidCratePath(path.clone()));
                }
            },

            Err(_) => {
                bail!(BuildErrorKind::InvalidCratePath(path.clone()));
            },
        }

        let cargo_toml: toml::Value = {
            let mut reader = BufReader::new(
                fs::File::open(path.join("Cargo.toml")).context(BuildErrorKind::OtherError)?,
            );

            let mut contents = String::new();

            reader
                .read_to_string(&mut contents)
                .context(BuildErrorKind::OtherError)?;

            toml::from_str(&contents).context(BuildErrorKind::OtherError)?
        };

        let Some(cargo_toml_name) = cargo_toml["package"]["name"].as_str() else {
            bail!(BuildErrorKind::InternalError(String::from(
                "Cannot get crate name"
            )));
        };

        let is_library = path.join("src").join("lib.rs").exists();
        let is_binary = path.join("src").join("main.rs").exists();

        let output_file_prefix = cargo_toml_name.replace('-', "_");

        let deps_file_prefix = match (is_binary, is_library) {
            (false, true) => FilePrefix::Library(format!("lib{output_file_prefix}")),
            (true, false) => FilePrefix::Binary(cargo_toml_name.to_string()),

            (true, true) => FilePrefix::Mixed {
                lib: format!("lib{output_file_prefix}"),
                bin: cargo_toml_name.to_string(),
            },

            (false, false) => {
                bail!(BuildErrorKind::InternalError(
                    "Unable to find neither `lib.rs` nor `main.rs`".into()
                ));
            },
        };

        Ok(Crate {
            name: cargo_toml_name.to_string(),
            path,
            output_file_prefix,
            deps_file_prefix,
        })
    }

    /// Returns PTX assmbly filename prefix.
    pub fn get_output_file_prefix(&self) -> &str {
        &self.output_file_prefix
    }

    /// Returns deps file filename prefix.
    pub fn get_deps_file_prefix(&self, crate_type: Option<CrateType>) -> Result<String> {
        match (&self.deps_file_prefix, crate_type) {
            (FilePrefix::Library(prefix), Some(CrateType::Library) | None)
            | (FilePrefix::Binary(prefix), Some(CrateType::Binary) | None) => Ok(prefix.clone()),
            (FilePrefix::Mixed { bin, .. }, Some(CrateType::Binary)) => Ok(bin.clone()),
            (FilePrefix::Mixed { lib, .. }, Some(CrateType::Library)) => Ok(lib.clone()),
            (FilePrefix::Mixed { .. }, None) => {
                bail!(BuildErrorKind::MissingCrateType);
            },

            (FilePrefix::Library(_), Some(CrateType::Binary)) => {
                bail!(BuildErrorKind::InvalidCrateType("Binary".into()));
            },

            (FilePrefix::Binary(_), Some(CrateType::Library)) => {
                bail!(BuildErrorKind::InvalidCrateType("Library".into()));
            },
        }
    }

    /// Returns the crate type to build the PTX with
    pub fn get_crate_type(&self, crate_type: Option<CrateType>) -> Result<&str> {
        match (&self.deps_file_prefix, crate_type) {
            (FilePrefix::Library(_), Some(CrateType::Library) | None)
            | (FilePrefix::Mixed { .. }, Some(CrateType::Library)) => Ok("cdylib,rlib"),

            (FilePrefix::Binary(_), Some(CrateType::Binary) | None)
            | (FilePrefix::Mixed { .. }, Some(CrateType::Binary)) => Ok("bin"),

            (FilePrefix::Mixed { .. }, None) => {
                bail!(BuildErrorKind::MissingCrateType);
            },

            (FilePrefix::Library(_), Some(CrateType::Binary)) => {
                bail!(BuildErrorKind::InvalidCrateType("Binary".into()));
            },

            (FilePrefix::Binary(_), Some(CrateType::Library)) => {
                bail!(BuildErrorKind::InvalidCrateType("Library".into()));
            },
        }
    }

    /// Returns crate name.
    pub fn get_name(&self) -> &str {
        &self.name
    }

    /// Returns crate root path.
    pub fn get_path(&self) -> &Path {
        self.path.as_path()
    }

    /// Returns temporary crate build location that can be `cargo clean`ed.
    pub fn get_output_path(&self) -> Result<PathBuf> {
        let mut path = PathBuf::from(env!("OUT_DIR"));

        path.push(&self.output_file_prefix);
        path.push(format!("{:x}", self.get_hash()));

        fs::create_dir_all(&path).context(BuildErrorKind::OtherError)?;
        Ok(path)
    }

    fn get_hash(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.hash(&mut hasher);

        hasher.finish()
    }
}

#[test]
fn should_find_crate_names() {
    let source = Crate::analyse("tests/fixtures/sample-crate").unwrap();

    assert_eq!(source.get_output_file_prefix(), "sample_ptx_crate");

    assert_eq!(
        source.get_deps_file_prefix(None).unwrap(),
        "libsample_ptx_crate"
    );

    assert_eq!(
        source
            .get_deps_file_prefix(Some(CrateType::Library))
            .unwrap(),
        "libsample_ptx_crate"
    );

    match source
        .get_deps_file_prefix(Some(CrateType::Binary))
        .unwrap_err()
        .kind()
    {
        BuildErrorKind::InvalidCrateType(kind) => {
            assert_eq!(kind, "Binary");
        },

        _ => unreachable!("it should fail with proper error"),
    }
}

#[test]
fn should_find_mixed_crate_names() {
    let source = Crate::analyse("tests/fixtures/mixed-crate").unwrap();

    assert_eq!(source.get_output_file_prefix(), "mixed_crate");

    assert_eq!(
        source
            .get_deps_file_prefix(Some(CrateType::Binary))
            .unwrap(),
        "mixed-crate"
    );

    assert_eq!(
        source
            .get_deps_file_prefix(Some(CrateType::Library))
            .unwrap(),
        "libmixed_crate"
    );

    match source.get_deps_file_prefix(None).unwrap_err().kind() {
        BuildErrorKind::MissingCrateType => {},
        _ => unreachable!("it should fail with proper error"),
    }
}

#[test]
fn should_check_existence_of_crate_path() {
    let result = Crate::analyse("tests/fixtures/non-existing-crate");

    match result.unwrap_err().kind() {
        BuildErrorKind::InvalidCratePath(path) => {
            assert!(path.ends_with("tests/fixtures/non-existing-crate"));
        },

        _ => unreachable!("it should fail with proper error"),
    }
}

#[test]
fn should_check_validity_of_crate_path() {
    let result = Crate::analyse("tests/builder.rs");

    match result.unwrap_err().kind() {
        BuildErrorKind::InvalidCratePath(path) => {
            assert!(path.ends_with("tests/builder.rs"));
        },

        _ => unreachable!("it should fail with proper error"),
    }
}

#[test]
fn should_provide_output_path() {
    let source_crate = Crate::analyse("tests/fixtures/sample-crate").unwrap();

    assert!(source_crate
        .get_output_path()
        .unwrap()
        .starts_with(Path::new(env!("OUT_DIR")).join("sample_ptx_crate")));
}
