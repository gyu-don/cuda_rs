#![allow(dead_code)]

use std::error;
use std::ffi::CStr;
use std::fmt::{self, Display, Formatter};
use std::os::raw;
use std::ptr::null_mut;
use std::result;

use cuda_runtime;
pub use cuda_runtime::{cudaError_t, cudaMemcpyKind, dim3};

pub fn driver_version() -> raw::c_int {
    let mut v = 0;
    unsafe {
        cuda_runtime::cudaDriverGetVersion(&mut v);
    }
    v
}

pub fn runtime_version() -> raw::c_int {
    let mut v = 0;
    unsafe {
        cuda_runtime::cudaRuntimeGetVersion(&mut v);
    }
    v
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Error {
    raw: cudaError_t,
}
impl Display for Error {
    fn fmt(&self, f: &mut Formatter) -> result::Result<(), fmt::Error> {
        write!(f,
               "{}",
               unsafe { CStr::from_ptr(cuda_runtime::cudaGetErrorString(self.raw)) }
                   .to_string_lossy())
    }
}
impl Error {
    fn cuda_error(&self) -> cudaError_t {
        self.raw
    }
    fn error_name(&self) -> String {
        unsafe { CStr::from_ptr(cuda_runtime::cudaGetErrorName(self.raw)) }
            .to_string_lossy()
            .into_owned()
    }
    fn error_string(&self) -> String {
        unsafe { CStr::from_ptr(cuda_runtime::cudaGetErrorString(self.raw)) }
            .to_string_lossy()
            .into_owned()
    }
}
impl error::Error for Error {
    fn description(&self) -> &str {
        "CUDA Error: see error_name() or error_string()"
    }
}

pub type Result<T> = result::Result<T, Error>;

pub unsafe fn malloc<T>(size: usize) -> Result<*mut T> {
    let mut ptr: *mut T = null_mut();
    let cuda_error = cuda_runtime::cudaMalloc(&mut ptr as *mut *mut T as *mut *mut raw::c_void,
                                              size);
    if cuda_error == cuda_runtime::cudaError::cudaSuccess {
        assert_ne!(ptr,
                   null_mut(),
                   "cudaMalloc is succeeded, but returned null pointer!");
        Ok(ptr)
    } else {
        Err(Error { raw: cuda_error })
    }
}

pub unsafe fn memcpy<T>(dst: *mut T,
                        src: *const T,
                        size: usize,
                        kind: cudaMemcpyKind)
                        -> Result<()> {
    let cuda_error =
        cuda_runtime::cudaMemcpy(dst as *mut raw::c_void, src as *mut raw::c_void, size, kind);
    if cuda_error == cuda_runtime::cudaError::cudaSuccess {
        Ok(())
    } else {
        Err(Error { raw: cuda_error })
    }
}

pub unsafe fn free<T>(devptr: *mut T) -> Result<()> {
    let cuda_error = cuda_runtime::cudaFree(devptr as *mut raw::c_void);
    if cuda_error == cuda_runtime::cudaError::cudaSuccess {
        Ok(())
    } else {
        Err(Error { raw: cuda_error })
    }
}

pub unsafe fn launch_kernel(func: *const raw::c_void,
                            grid_dim: dim3,
                            block_dim: dim3,
                            args: &mut [*mut raw::c_void],
                            shared_mem: usize,
                            stream: cuda_runtime::cudaStream_t)
                            -> Result<()> {
    let cuda_error = cuda_runtime::cudaLaunchKernel(func,
                                                    grid_dim,
                                                    block_dim,
                                                    args.as_mut_ptr(),
                                                    shared_mem,
                                                    stream);
    if cuda_error == cuda_runtime::cudaError::cudaSuccess {
        Ok(())
    } else {
        Err(Error { raw: cuda_error })
    }
}

pub fn last_error() -> Result<()> {
    match unsafe { cuda_runtime::cudaGetLastError() } {
        cuda_runtime::cudaError::cudaSuccess => Ok(()),
        err => Err(Error { raw: err }),
    }
}

pub fn event_create() -> Result<cuda_runtime::cudaEvent_t> {
    let mut ev = null_mut();
    let cuda_error = unsafe { cuda_runtime::cudaEventCreate(&mut ev as *mut _) };
    match cuda_error {
        cuda_runtime::cudaError::cudaSuccess => Ok(ev),
        _ => Err(Error { raw: cuda_error }),
    }
}

pub unsafe fn event_destroy(ev: cuda_runtime::cudaEvent_t) -> Result<()> {
    match cuda_runtime::cudaEventDestroy(ev) {
        cuda_runtime::cudaError::cudaSuccess => Ok(()),
        err => Err(Error { raw: err }),
    }
}

pub unsafe fn event_elapsed_time(start: cuda_runtime::cudaEvent_t,
                                 end: cuda_runtime::cudaEvent_t)
                                 -> Result<f32> {
    let mut ms: f32 = 0.0;
    let cuda_error = cuda_runtime::cudaEventElapsedTime(&mut ms as *mut _, start, end);
    match cuda_error {
        cuda_runtime::cudaError::cudaSuccess => Ok(ms),
        _ => Err(Error { raw: cuda_error }),
    }
}

pub unsafe fn event_query(event: cuda_runtime::cudaEvent_t) -> Result<bool> {
    match cuda_runtime::cudaEventQuery(event) {
        cuda_runtime::cudaError::cudaSuccess => Ok(true),
        cuda_runtime::cudaError::cudaErrorNotReady => Ok(false),
        err => Err(Error { raw: err }),
    }
}

pub unsafe fn event_record_with_stream(event: cuda_runtime::cudaEvent_t,
                                       stream: cuda_runtime::cudaStream_t)
                                       -> Result<()> {
    match cuda_runtime::cudaEventRecord(event, stream) {
        cuda_runtime::cudaError::cudaSuccess => Ok(()),
        err => Err(Error { raw: err }),
    }
}

#[inline]
pub unsafe fn event_record(event: cuda_runtime::cudaEvent_t) -> Result<()> {
    event_record_with_stream(event, null_mut())
}

pub unsafe fn event_synchronize(event: cuda_runtime::cudaEvent_t) -> Result<()> {
    match cuda_runtime::cudaEventSynchronize(event) {
        cuda_runtime::cudaError::cudaSuccess => Ok(()),
        err => Err(Error { raw: err }),
    }
}

pub fn profiler_initialize(config_file: &CStr,
                           output_file: &CStr,
                           output_mode: cuda_runtime::cudaOutputMode_t)
                           -> Result<()> {
    match unsafe {
        cuda_runtime::cudaProfilerInitialize(config_file.as_ptr(),
                                             output_file.as_ptr(),
                                             output_mode)
    } {
        cuda_runtime::cudaError::cudaSuccess => Ok(()),
        err => Err(Error { raw: err }),
    }
}

pub fn profiler_start() -> Result<()> {
    match unsafe { cuda_runtime::cudaProfilerStart() } {
        cuda_runtime::cudaError::cudaSuccess => Ok(()),
        err => Err(Error { raw: err }),
    }
}

pub fn profiler_stop() -> Result<()> {
    match unsafe { cuda_runtime::cudaProfilerStop() } {
        cuda_runtime::cudaError::cudaSuccess => Ok(()),
        err => Err(Error { raw: err }),
    }
}
