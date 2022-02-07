use cargo_metadata::diagnostic::{DiagnosticBuilder, DiagnosticLevel, DiagnosticSpanBuilder};

use ptx_builder::{error::Error, reporter::ErrorLogPrinter};

lazy_static::lazy_static! {
    pub static ref PROC_MACRO_SPAN_REGEX: regex::Regex = {
        regex::Regex::new(r"\((?P<start>[0-9]+)[^0-9]+(?P<end>[0-9]+)\)").unwrap()
    };
}

#[allow(clippy::module_name_repetitions)]
pub fn emit_ptx_build_error(err: Error) {
    let err = ErrorLogPrinter::print(err);

    let rendered = err.to_string();
    let message = String::from_utf8(strip_ansi_escapes::strip(&rendered).unwrap()).unwrap();

    let call_site = proc_macro::Span::call_site();

    let (byte_start, byte_end) =
        if let Some(captures) = PROC_MACRO_SPAN_REGEX.captures(&format!("{:?}", call_site)) {
            (
                captures["start"].parse().unwrap_or(0_u32),
                captures["end"].parse().unwrap_or(0_u32),
            )
        } else {
            (0_u32, 0_u32)
        };

    let span = DiagnosticSpanBuilder::default()
        .file_name(
            call_site
                .source_file()
                .path()
                .to_string_lossy()
                .into_owned(),
        )
        .byte_start(byte_start)
        .byte_end(byte_end)
        .line_start(call_site.start().line)
        .line_end(call_site.end().line)
        .column_start(call_site.start().column)
        .column_end(call_site.end().column)
        .is_primary(true)
        .text(vec![])
        .label(None)
        .suggested_replacement(None)
        .suggestion_applicability(None)
        .expansion(None)
        .build()
        .unwrap();

    let diagnostic = DiagnosticBuilder::default()
        .message(message)
        .code(None)
        .level(DiagnosticLevel::Error)
        .spans(vec![span])
        .children(vec![])
        .rendered(rendered)
        .build()
        .unwrap();

    eprintln!("{}", serde_json::to_string(&diagnostic).unwrap());
}
