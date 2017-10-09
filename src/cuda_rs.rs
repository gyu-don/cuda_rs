use std;
use std::mem;

use cuda_ffi::{self, Result};
use cuda_runtime::{dim3, cudaMemcpyKind};

pub struct CudaMem<T> {
    ptr: *mut T,
    len: usize,
}

impl<T> CudaMem<T> {
    pub fn from_host_slice(src: &[T]) -> Result<CudaMem<T>>
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
        Ok(CudaMem {
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
                             self.size(),
                             cudaMemcpyKind::cudaMemcpyDeviceToHost)?;
        }
        Ok(v)
    }

    pub unsafe fn allocate(len: usize) -> Result<CudaMem<T>> {
        let allocation_size = len * mem::size_of::<T>();
        let ptr: *mut T = cuda_ffi::malloc(allocation_size)?;
        Ok(CudaMem { ptr, len: len })
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
    pub fn from_raw(ptr: *mut T, len: usize) -> CudaMem<T> {
        CudaMem { ptr, len }
    }
    fn size(&self) -> usize {
        self.len() * mem::size_of::<T>()
    }
}

impl<T> Drop for CudaMem<T> {
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
