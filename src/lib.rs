pub mod cuda_runtime;
pub mod cuda_ffi;

pub use self::cuda_ffi::{driver_version, runtime_version, profiler_initialize, profiler_start,
                         profiler_stop, Result};
pub use self::cuda_runtime::dim3;

use std::mem;

use cuda_runtime::cudaMemcpyKind;

pub struct DeviceMem<T> {
    ptr: *mut T,
    len: usize,
}

impl<T> DeviceMem<T> {
    pub fn from_host_slice(src: &[T]) -> Result<DeviceMem<T>>
        where T: Copy
    {
        let allocation_size = src.len() * mem::size_of::<T>();
        let ptr: *mut T = unsafe { cuda_ffi::malloc(allocation_size)? };
        unsafe {
            cuda_ffi::memcpy(ptr,
                             src.as_ptr(),
                             allocation_size,
                             cudaMemcpyKind::cudaMemcpyHostToDevice)?;
        }
        Ok(DeviceMem {
            ptr,
            len: src.len(),
        })
    }
    pub fn to_host_vec(self) -> Result<Vec<T>>
        where T: Copy + Default
    {
        let mut v = vec![T::default(); self.len()];
        unsafe {
            cuda_ffi::memcpy(v.as_mut_ptr(),
                             self.as_ptr(),
                             self.bytes_len(),
                             cudaMemcpyKind::cudaMemcpyDeviceToHost)?;
        }
        Ok(v)
    }

    pub unsafe fn allocate(len: usize) -> Result<DeviceMem<T>> {
        let allocation_size = len * mem::size_of::<T>();
        let ptr: *mut T = cuda_ffi::malloc(allocation_size)?;
        Ok(DeviceMem { ptr, len: len })
    }

    pub fn len(&self) -> usize {
        self.len
    }
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr
    }
    pub fn as_ptr(&self) -> *const T {
        self.ptr as *const T
    }
    pub fn into_raw(self) -> *mut T {
        self.ptr
    }
    pub fn from_raw(ptr: *mut T, len: usize) -> DeviceMem<T> {
        DeviceMem { ptr, len }
    }
    pub fn bytes_len(&self) -> usize {
        self.len() * mem::size_of::<T>()
    }
}

impl<T> Drop for DeviceMem<T> {
    fn drop(&mut self) {
        unsafe {
            let _ = cuda_ffi::free(self.ptr);
        }
    }
}

pub trait ToDim3 {
    fn to_dim3(self) -> dim3;
}
impl ToDim3 for dim3 {
    #[inline]
    fn to_dim3(self) -> dim3 {
        self
    }
}
impl<'a> ToDim3 for &'a dim3 {
    #[inline]
    fn to_dim3(self) -> dim3 {
        *self
    }
}

#[macro_export]
macro_rules! dim3 {
    ($x: expr, $y: expr, $z: expr) => {
        dim3 { x: $x as std::os::raw::c_uint, y: $y as std::os::raw::c_uint, z: $z as std::os::raw::c_uint }
    };
    ($x: expr, $y: expr) => {
        dim3 { x: $x as std::os::raw::c_uint, y: $y as std::os::raw::c_uint, z: 1 }
    };
    ($x: expr) => {
        dim3 { x: $x as std::os::raw::c_uint, y: 1, z: 1 }
    };
}

macro_rules! impl_todim3 {
    ($ty: ty) => {
        impl ToDim3 for $ty {
            #[inline]
            fn to_dim3(self) -> dim3 {
                dim3!(self)
            }
        }
        impl<'a> ToDim3 for &'a $ty {
            #[inline]
            fn to_dim3(self) -> dim3 {
                dim3!(*self)
            }
        }
    }
}
impl_todim3!(usize);
impl_todim3!(u64);
impl_todim3!(u32);

#[inline]
pub fn to_dim3<T: ToDim3>(value: T) -> dim3 {
    value.to_dim3()
}

#[macro_export]
macro_rules! cuda_call {
    ($func: ident <<< $blk: expr, $grd: expr, $shr: expr, >>> ($($arg: expr),*)) => ( {
        unsafe {
            if false { $func($($arg),*); } // type check of arguments
            $crate::cuda_ffi::launch_kernel($func as *const std::os::raw::c_void,
                                     $crate::to_dim3($blk),
                                     $crate::to_dim3($grd),
                                     &mut [$(&$arg as *const _ as *mut std::os::raw::c_void),*],
                                     $shr,
                                     std::ptr::null_mut())
        }
    } );
    ($func: ident <<< $blk: expr, $grd: expr, >>> ($($arg: expr),*)) => (
        cuda_call!($func<<<$blk, $grd, 0,>>> ($($arg),*))
    );
    ($func: ident <<< $blk: expr, >>> ($($arg: expr),*)) => (
        cuda_call!($func<<<$blk, $crate::dim3 { x: 1, y: 1, z: 1 }, 0,>>> ($($arg),*))
    );
}

pub struct Event(cuda_runtime::cudaEvent_t);
impl Event {
    pub fn create() -> Result<Event> {
        let ev = cuda_ffi::event_create()?;
        Ok(Event(ev))
    }
    pub fn elapsed_time(&self, end: &Event) -> Result<f32> {
        unsafe { cuda_ffi::event_elapsed_time(self.0, end.0) }
    }
    pub fn query(&self) -> Result<bool> {
        unsafe { cuda_ffi::event_query(self.0) }
    }
    pub fn record(&self) -> Result<()> {
        unsafe { cuda_ffi::event_record(self.0) }
    }
    pub fn synchronize(&self) -> Result<()> {
        unsafe { cuda_ffi::event_synchronize(self.0) }
    }
}
impl Drop for Event {
    fn drop(&mut self) {
        unsafe {
            let _ = cuda_ffi::event_destroy(self.0);
        }
    }
}


#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        unimplemented!();
    }
}
