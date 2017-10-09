extern crate bindgen;

use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rustc-link-lib=cudart");

    let bindings = bindgen::Builder::default()
        .clang_arg("-I/opt/cuda/include")
        .header("src/cuda_headers.h")
        .generate()
        .expect("Unable to generate bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings.write_to_file(out_path.join("cuda_runtime.rs"))
        .expect("Couldn't write bindings");
}
