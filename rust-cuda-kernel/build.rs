fn main() {
    find_cuda_helper::include_cuda();

    println!("cargo:rustc-link-lib=nvptxcompiler_static");
}
