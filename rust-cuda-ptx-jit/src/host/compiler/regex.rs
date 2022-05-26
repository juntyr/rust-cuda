#[allow(unused_imports)]
use regex::bytes::Regex;

lazy_static::lazy_static! {
    pub static ref CONST_MARKER_REGEX: Regex = {
        Regex::new(
            r"(?-u)// <rust-cuda-ptx-jit-const-load-(?P<tmpreg>%r\d+)-(?P<param>\d+)> //"
        ).unwrap()
    };

    pub static ref CONST_BASE_REGISTER_REGEX: Regex = {
        Regex::new(
            r"(?-u)ld\.global\.u32\s*(?P<tmpreg>%r\d+)\s*,\s*\[(?P<basereg>%r[ds]?\d+)]\s*;",
        ).unwrap()
    };

    pub static ref CONST_LOAD_INSTRUCTION_REGEX: Regex = {
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
        ).unwrap()
    };

    pub static ref REGISTER_REGEX: Regex = {
        Regex::new(r"(?-u)(?P<register>%[rf][sd]?\d+)").unwrap()
    };
}
