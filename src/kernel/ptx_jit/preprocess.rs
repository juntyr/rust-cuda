use std::{
    collections::HashMap,
    ffi::{CStr, CString},
};

use super::{
    regex::{
        const_base_register_regex, const_load_instruction_regex, const_marker_regex, register_regex,
    },
    PtxElement, PtxJITCompiler, PtxLoadWidth,
};

impl PtxJITCompiler {
    #[must_use]
    pub fn new(ptx: &CStr) -> Self {
        let ptx = ptx.to_bytes();

        let mut const_markers: HashMap<&[u8], usize> = HashMap::new();

        // Find injected rust-cuda-const-markers which identify dummy register rxx
        for const_marker in const_marker_regex().captures_iter(ptx) {
            if let Some(tmpreg) = const_marker.name("tmpreg").map(|s| s.as_bytes()) {
                if let Some(param) = const_marker
                    .name("param")
                    .map(|s| s.as_bytes())
                    .and_then(|b| std::str::from_utf8(b).ok())
                    .and_then(|s| s.parse().ok())
                {
                    const_markers.insert(tmpreg, param);
                }
            }
        }
        // const_markers now contains a mapping rxx => param index

        let mut const_base_registers: HashMap<&[u8], usize> = HashMap::new();

        // Find base register ryy which was used in `ld.global.u32 rxx, [ryy];`
        for const_base_register in const_base_register_regex().captures_iter(ptx) {
            if let Some(tmpreg) = const_base_register.name("tmpreg").map(|s| s.as_bytes()) {
                if let Some(param) = const_markers.get(tmpreg) {
                    if let Some(basereg) = const_base_register.name("basereg").map(|s| s.as_bytes())
                    {
                        const_base_registers.insert(basereg, *param);
                    }
                }
            }
        }
        // const_base_registers now contains a mapping ryy => param index

        let mut from_index = 0_usize;
        let mut last_slice = Vec::new();

        let mut ptx_slices: Vec<PtxElement> = Vec::new();

        // Iterate over all load from base register with offset instructions
        for const_load_instruction in const_load_instruction_regex().captures_iter(ptx) {
            // Only consider instructions where the base register is ryy
            if let Some(basereg) = const_load_instruction.name("basereg").map(|s| s.as_bytes()) {
                if let Some(param) = const_base_registers.get(basereg) {
                    if let Some(loadwidth) = match const_load_instruction
                        .name("loadwidth")
                        .map(|s| s.as_bytes())
                    {
                        Some(b"8") => Some(PtxLoadWidth::B1),
                        Some(b"16") => Some(PtxLoadWidth::B2),
                        Some(b"32") => Some(PtxLoadWidth::B4),
                        Some(b"64") => Some(PtxLoadWidth::B8),
                        _ => None,
                    } {
                        if let Some(constreg) = const_load_instruction
                            .name("constreg")
                            .map(|s| s.as_bytes())
                        {
                            if let Some(loadoffset) = std::str::from_utf8(
                                const_load_instruction
                                    .name("loadoffset")
                                    .map_or(b"0", |s| s.as_bytes()),
                            )
                            .ok()
                            .and_then(|s| s.parse().ok())
                            {
                                if let Some((range, instruction)) = const_load_instruction
                                    .name("instruction")
                                    .map(|s| (s.range(), s.as_bytes()))
                                {
                                    // Store the PTX source code before the load instruction
                                    last_slice.extend_from_slice(&ptx[from_index..range.start]);

                                    ptx_slices.push(PtxElement::CopiedSource {
                                        ptx: std::mem::take(&mut last_slice).into_boxed_slice(),
                                    });

                                    from_index = range.end;

                                    // Store the load instruction with extracted parameters to
                                    //  generate a constant load if requested
                                    ptx_slices.push(PtxElement::ConstLoad {
                                        ptx: instruction.to_owned().into_boxed_slice(),
                                        parameter_index: *param,
                                        byte_offset: loadoffset,
                                        load_width: loadwidth,
                                        registers: register_regex()
                                            .captures_iter(constreg)
                                            .filter_map(|m| {
                                                m.name("register").map(|s| {
                                                    s.as_bytes().to_owned().into_boxed_slice()
                                                })
                                            })
                                            .collect::<Vec<_>>()
                                            .into_boxed_slice(),
                                    });
                                }
                            }
                        }
                    }
                }
            }
        }

        // Store the remainder of the PTX source code
        last_slice.extend_from_slice(&ptx[from_index..ptx.len()]);

        if !last_slice.is_empty() {
            ptx_slices.push(PtxElement::CopiedSource {
                ptx: last_slice.into_boxed_slice(),
            });
        }

        // Create the `PtxJITCompiler` which also caches the last PTX version
        Self {
            ptx_slices: ptx_slices.into_boxed_slice(),
            last_arguments: None,
            last_ptx: unsafe { CString::from_vec_unchecked(ptx.to_owned()) },
        }
    }
}
