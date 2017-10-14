use std::env;
use std::path;

#[cfg(windows)]
const NVCC: &'static str = "nvcc.exe";
#[cfg(not(windows))]
const NVCC: &'static str = "nvcc";

pub fn get_cuda_path_from_env() -> Result<String, &'static str> {
    let delim;
    if cfg!(unix) {
        delim = ':';
    } else if cfg!(windows) {
        delim = ';';
    } else {
        return Err("Unimplemented");
    }

    let path = env::var("PATH").map_err(|e| match e {
            env::VarError::NotPresent => "PATH not found",
            env::VarError::NotUnicode(_) => "Invalid PATH",
        })?;
    for p in path.split(delim) {
        let p = path::Path::new(p);
        if p.join(NVCC).exists() {
            if let Some(p_) = p.parent().and_then(|e| e.to_str()) {
                return Ok(p_.to_owned());
            }
        }
    }
    Err("Not found")
}

pub fn get_include_dir(cuda_path: &str) -> path::PathBuf {
    path::Path::new(cuda_path).join("include")
}

pub fn get_nvcc(cuda_path: &str) -> path::PathBuf {
    path::Path::new(cuda_path).join(NVCC)
}
