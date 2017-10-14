extern crate bindgen;

include!("src/cuda_buildhelper.rs");

use std::fs;
use std::io::Write;

const HEADERS: &[&'static str] = &["cuda_runtime.h", "cuda_profiler_api.h"];

fn do_bindgen_if_required() {
    let cuda_path = get_cuda_path_from_env().unwrap();
    let cuda_include = get_include_dir(&cuda_path);
    let out_path = path::PathBuf::from(env::var("OUT_DIR").unwrap());
    let target = out_path.join("cuda_runtime.rs");

    let is_bindgen_required = || {
        let get_modified = |path| fs::metadata(path).and_then(|md| md.modified()).ok();
        get_modified(target.to_str().unwrap())
            .map(|target_modified| {
                get_modified(file!()).map(|modified| target_modified < modified).unwrap_or(true) &&
                HEADERS.iter().all(|hdr| {
                    get_modified(hdr)
                        .map(|modified| target_modified < modified)
                        .unwrap_or(true)
                })
            })
            .unwrap_or(true)
    };

    if is_bindgen_required() {
        let headers_path = out_path.join("cuda_headers.h");
        {
            let mut headers = fs::File::create(&headers_path).unwrap();
            for header in HEADERS {
                writeln!(headers, "#include <{}>", header).unwrap();
            }
        }

        let bindings = bindgen::Builder::default()
            .clang_arg(format!("-I{}", cuda_include.to_str().unwrap()))
            .header(headers_path.to_str().unwrap())
            .generate()
            .expect("Unable to generate bindings");

        bindings.write_to_file(&target)
            .expect("Couldn't write bindings");
    }
}

fn main() {
    println!("cargo:rustc-link-lib=cudart");
    do_bindgen_if_required();
}
