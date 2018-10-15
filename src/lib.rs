pub mod cuda_runtime;
pub mod cuda_ffi;
pub mod cuda_buildhelper;

pub use self::cuda_ffi::{driver_version, runtime_version, profiler_initialize, profiler_start,
                         profiler_stop, Error, Result};
pub use self::cuda_runtime::dim3;

use std::mem;

use cuda_runtime::cudaMemcpyKind;

/// Wrapped CUDA's `__global__` memory.
pub struct DeviceMem<T> {
    ptr: *mut T,
    len: usize,
}

impl<T> DeviceMem<T> {
    /// Creates a wrapped CUDA's `__global__` memory
    /// which is allocated and copied by specified slice.
    pub fn from_host_slice(src: &[T]) -> Result<DeviceMem<T>> {
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

    /// Creates an vector which is copied from `self`.
    pub fn to_host_vec(&self) -> Result<Vec<T>> {
        let mut v = Vec::with_capacity(self.len());
        unsafe {
            cuda_ffi::memcpy(v.as_mut_ptr(),
                             self.as_ptr(),
                             self.bytes_len(),
                             cudaMemcpyKind::cudaMemcpyDeviceToHost)?;
            v.set_len(self.len());
        }
        Ok(v)
    }

    /// Copies all elements from `self` to `dest`.
    /// The length of `dest` must be the same as `self`.
    /// # Panics
    /// This function will panic if `self` and `dest` have different lengths.
    pub fn to_host_slice(&self, dest: &mut [T]) -> Result<()> {
        assert_eq!(self.len(), dest.len());
        unsafe {
            cuda_ffi::memcpy(dest.as_mut_ptr(),
                             self.as_ptr(),
                             self.bytes_len(),
                             cudaMemcpyKind::cudaMemcpyDeviceToHost)?;
        }
        Ok(())
    }

    /// Copies all elements from `src` to `self`.
    /// The length of `src` must be the same as `self`.
    /// # Panics
    /// This function will panic if `self` and `src` have different lengths.
    pub fn replace_with_host_slice(&mut self, src: &[T]) -> Result<()> {
        assert_eq!(self.len(), src.len());
        unsafe {
            cuda_ffi::memcpy(self.as_mut_ptr(),
                             src.as_ptr(),
                             self.bytes_len(),
                             cudaMemcpyKind::cudaMemcpyHostToDevice)?;
        }
        Ok(())
    }

    /// Returns a value which is a copied element from specified position or `None` if out of bound.
    pub fn get(&self, idx: usize) -> Result<Option<T>> {
        if idx >= self.len() {
            return Ok(None);
        }
        unsafe {
            let mut v = mem::uninitialized();
            cuda_ffi::memcpy(&mut v as *mut T,
                             self.as_ptr().offset(idx as isize),
                             mem::size_of::<T>(),
                             cudaMemcpyKind::cudaMemcpyDeviceToHost)?;
            Ok(Some(v))
        }
    }

    /// Set a value to specified element.
    /// Returns `Ok(Some(()))` if succeeded, `Err(CudaError)` when CudaError occured,
    /// `Ok(None)` if out of bound.
    pub fn set(&mut self, idx: usize, val: &T) -> Result<Option<()>> {
        if idx >= self.len() {
            return Ok(None);
        }
        unsafe {
            cuda_ffi::memcpy(self.as_mut_ptr().offset(idx as isize),
                             val,
                             mem::size_of::<T>(),
                             cudaMemcpyKind::cudaMemcpyHostToDevice)?;
            Ok(Some(()))
        }
    }

    /// Returns a value which is copied an element at that position, without doing bounds checking.
    /// This is generally not recommended, use with caution! For a safe alternative see [`get`].
    pub unsafe fn get_unchecked(&self, idx: usize) -> Result<T> {
        let mut v = mem::uninitialized();
        cuda_ffi::memcpy(&mut v as *mut T,
                         self.as_ptr().offset(idx as isize),
                         mem::size_of::<T>(),
                         cudaMemcpyKind::cudaMemcpyDeviceToHost)?;
        Ok(v)
    }

    /// Constructs wrapped CUDA's `__global__` memory which is allocated but uninitialized.
    pub unsafe fn allocate(len: usize) -> Result<DeviceMem<T>> {
        let allocation_size = len * mem::size_of::<T>();
        let ptr: *mut T = cuda_ffi::malloc(allocation_size)?;
        Ok(DeviceMem { ptr, len: len })
    }

    /// Returns the number of elements of the memory.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns an unsafe mutable pointer to device memory.
    /// The caller must ensure that the memory outlives the pointer this function retunrs, or else
    /// it will end up pointing to garbage.
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr
    }

    /// Returns a raw pointer to device memory.
    /// The caller must ensure that the memory outlives the pointer this function retunrs, or else
    /// it will end up pointing to garbage.
    pub fn as_ptr(&self) -> *const T {
        self.ptr as *const T
    }

    /// Consumes `DeviceMem`, returning the raw pointer.
    ///
    /// After calling this function, the caller is respoinsible for the device memory previously
    /// managed by the `DeviceMem`. In particular, the caller should properly free the device
    /// memory.
    pub fn into_raw(self) -> *mut T {
        self.ptr
    }

    /// Constructs a `DeviceMem` from a raw pointer and size.
    ///
    /// After calling this function, the raw pointer is owned by the resulting `DeviceMem`.
    pub unsafe fn from_raw(ptr: *mut T, len: usize) -> DeviceMem<T> {
        DeviceMem { ptr, len }
    }

    /// Returns a byte length of allocated memory.
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

/*
/// Trait that can be convert to dim3.
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
*/

/// Make new `dim3`.
#[macro_export]
macro_rules! dim3 {
    ($x: expr, $y: expr, $z: expr) => {
        dim3 {
            x: $x as std::os::raw::c_uint,
            y: $y as std::os::raw::c_uint,
            z: $z as std::os::raw::c_uint
        }
    };
    ($x: expr, $y: expr) => {
        dim3 { x: $x as std::os::raw::c_uint, y: $y as std::os::raw::c_uint, z: 1 }
    };
    ($x: expr) => {
        dim3 { x: $x as std::os::raw::c_uint, y: 1, z: 1 }
    };
}

macro_rules! impl_from_for_dim3 {
    ($ty: ty) => {
        impl From<$ty> for dim3 {
            #[inline]
            fn from(x: $ty) -> dim3 {
                dim3!(x)
            }
        }
        impl<'a> From<&'a $ty> for dim3 {
            #[inline]
            fn from(x: &'a $ty) -> dim3 {
                dim3!(*x)
            }
        }
        impl From<($ty, $ty)> for dim3 {
            #[inline]
            fn from(v: ($ty, $ty)) -> dim3 {
                dim3!(v.0, v.1)
            }
        }
        impl<'a> From<(&'a $ty, &'a $ty)> for dim3 {
            #[inline]
            fn from(v: (&'a $ty, &'a $ty)) -> dim3 {
                dim3!(*v.0, *v.1)
            }
        }
        impl<'a> From<&'a ($ty, $ty)> for dim3 {
            #[inline]
            fn from(v: &'a ($ty, $ty)) -> dim3 {
                dim3!(v.0, v.1)
            }
        }
        impl<'a> From<&'a (&'a $ty, &'a $ty)> for dim3 {
            #[inline]
            fn from(v: &'a (&'a $ty, &'a $ty)) -> dim3 {
                dim3!(*v.0, *v.1)
            }
        }
        impl From<($ty, $ty, $ty)> for dim3 {
            #[inline]
            fn from(v: ($ty, $ty, $ty)) -> dim3 {
                dim3!(v.0, v.1, v.2)
            }
        }
        impl<'a> From<(&'a $ty, &'a $ty, &'a $ty)> for dim3 {
            #[inline]
            fn from(v: (&'a $ty, &'a $ty, &'a $ty)) -> dim3 {
                dim3!(*v.0, *v.1, *v.2)
            }
        }
        impl<'a> From<&'a ($ty, $ty, $ty)> for dim3 {
            #[inline]
            fn from(v: &'a ($ty, $ty, $ty)) -> dim3 {
                dim3!(v.0, v.1, v.2)
            }
        }
        impl<'a> From<&'a (&'a $ty, &'a $ty, &'a $ty)> for dim3 {
            #[inline]
            fn from(v: &'a (&'a $ty, &'a $ty, &'a $ty)) -> dim3 {
                dim3!(*v.0, *v.1, *v.2)
            }
        }
    }
}

impl_from_for_dim3!(usize);
impl_from_for_dim3!(u64);
impl_from_for_dim3!(u32);

/// Call CUDA's kernel. Returns Result<()>.
/// ```
/// call_cuda!(name_of_cuda_kernel<<<block_dim, grid_dim, shared_mem_size,>>>(args...));
/// call_cuda!(name_of_cuda_kernel<<<block_dim, grid_dim,>>>(args...));
/// call_cuda!(name_of_cuda_kernel<<<block_dim,>>>(args...));
/// ```
/// Don't omit `,` before `>>>`.
#[macro_export]
macro_rules! cuda_call {
    ($func: ident <<< $blk: expr, $grd: expr, $shr: expr, >>> ($($arg: expr),*)) => ( {
        unsafe {
            if false { $func($($arg),*); } // type check of arguments
            $crate::cuda_ffi::launch_kernel($func as *const std::os::raw::c_void,
                                     $crate::dim3::from($blk),
                                     $crate::dim3::from($grd),
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
