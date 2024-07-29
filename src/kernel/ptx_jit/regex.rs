use std::sync::OnceLock;

use regex::bytes::Regex;

#[expect(clippy::module_name_repetitions)]
pub fn const_marker_regex() -> &'static Regex {
    static CONST_MARKER_REGEX: OnceLock<Regex> = OnceLock::new();
    CONST_MARKER_REGEX.get_or_init(|| {
        Regex::new(r"(?-u)// <rust-cuda-ptx-jit-const-load-(?P<tmpreg>%r\d+)-(?P<param>\d+)> //")
            .unwrap()
    })
}

#[expect(clippy::module_name_repetitions)]
pub fn const_base_register_regex() -> &'static Regex {
    static CONST_BASE_REGISTER_REGEX: OnceLock<Regex> = OnceLock::new();
    CONST_BASE_REGISTER_REGEX.get_or_init(|| {
        Regex::new(r"(?-u)ld\.global\.u32\s*(?P<tmpreg>%r\d+)\s*,\s*\[(?P<basereg>%r[ds]?\d+)]\s*;")
            .unwrap()
    })
}

#[expect(clippy::module_name_repetitions)]
pub fn const_load_instruction_regex() -> &'static Regex {
    static CONST_LOAD_INSTRUCTION_REGEX: OnceLock<Regex> = OnceLock::new();
    CONST_LOAD_INSTRUCTION_REGEX.get_or_init(|| {
        Regex::new(
            r"(?x-u)(?P<instruction>
                ld\.global
                (?:\.(?P<vector>v[24]))?
                \.
                (?P<loadtype>[suf])
                (?P<loadwidth>8|16|32|64)
                \s*
                (?P<constreg>
                    (?:%[rf][sd]?\d+) |
                    (?:\{(?:\s*%[rf][sd]?\d+,)*\s*%[rf][sd]?\d+\s*\})
                )
                ,\s*
                \[
                (?P<basereg>%r[ds]?\d+)
                (?:
                    \+
                    (?P<loadoffset>\d+)
                )?
                \]
                \s*;
            )",
        )
        .unwrap()
    })
}

#[expect(clippy::module_name_repetitions)]
pub fn register_regex() -> &'static Regex {
    static REGISTER_REGEX: OnceLock<Regex> = OnceLock::new();
    REGISTER_REGEX.get_or_init(|| Regex::new(r"(?-u)(?P<register>%[rf][sd]?\d+)").unwrap())
}
