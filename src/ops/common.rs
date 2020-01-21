use ocl_core::CommandQueue;
use ocl_core::Event;
use ocl_core::MapFlags;
use ocl_core::Mem;
use ocl_core::OclPrm;

pub fn calc_num_blocks(size: usize, num_threads: usize) -> usize {
    (size + num_threads - 1) / num_threads
}

pub unsafe fn read_buffer<T: OclPrm>(queue: &CommandQueue, buf: &Mem, ret: &mut [T]) {
    let mem = ocl_core::enqueue_map_buffer(
        &queue,
        buf,
        true,
        MapFlags::READ,
        0,
        ret.len(),
        None::<Event>,
        None::<&mut Event>,
    )
    .unwrap();
    ret.clone_from_slice(mem.as_slice(ret.len()));
    ocl_core::enqueue_unmap_mem_object(&queue, buf, &mem, None::<Event>, None::<&mut Event>)
        .unwrap();
}

pub unsafe fn write_buffer<T: OclPrm>(queue: &CommandQueue, val: &[T], buf: &Mem) {
    let mut mem = ocl_core::enqueue_map_buffer(
        &queue,
        buf,
        true,
        MapFlags::WRITE,
        0,
        val.len(),
        None::<Event>,
        None::<&mut Event>,
    )
    .unwrap();
    mem.as_slice_mut(val.len()).clone_from_slice(val);
    ocl_core::enqueue_unmap_mem_object(&queue, buf, &mem, None::<Event>, None::<&mut Event>)
        .unwrap();
}

macro_rules! define_empty_impl {
    ( $name:ident ) => {
        pub struct $name {
            internal: std::sync::Arc<crate::OpenCLInternal>,
        }
        impl $name {
            pub fn new(internal: &std::sync::Arc<crate::OpenCLInternal>) -> $name {
                $name {
                    internal: std::sync::Arc::clone(internal),
                }
            }
        }
    };
}

macro_rules! buffer {
    ( $tensor:expr) => {
        ($tensor.handle().load(std::sync::atomic::Ordering::Acquire) as *const ocl_core::Mem)
            .as_ref()
            .unwrap()
    };
}

macro_rules! define_opencl_impl_struct {
    ( $name:ident, $kernel:ident ) => {
        pub struct $name {
            kernel: std::sync::Mutex<ocl_core::Kernel>,
            wgs: [usize; 3],
            internal: std::sync::Arc<crate::OpenCLInternal>,
        }
        impl $name {
            pub fn new(
                program: &ocl_core::Program,
                internal: &std::sync::Arc<crate::OpenCLInternal>,
            ) -> $name {
                let kernel = ocl_core::create_kernel(program, stringify!($kernel)).unwrap();
                match ocl_core::get_kernel_work_group_info(
                    &kernel,
                    internal.queue.device().unwrap(),
                    ocl_core::KernelWorkGroupInfo::CompileWorkGroupSize,
                )
                .unwrap()
                {
                    ocl_core::KernelWorkGroupInfoResult::CompileWorkGroupSize(wgs) => $name {
                        kernel: std::sync::Mutex::new(kernel),
                        wgs: wgs,
                        internal: std::sync::Arc::clone(internal),
                    },
                    _ => panic!(),
                }
            }
        }
    };
}

macro_rules! define_opencl_fw_x_impl {
    ( $name:ident, $kernel:ident ) => {
        define_opencl_impl_struct!($name, $kernel);
        impl prima_undine::device_impl::FunctionFwImpl for $name {
            fn call(
                &self,
                xs: &[&prima_undine::Tensor],
                _u32data: &[u32],
                _f32data: &[f32],
                ys: &mut [&mut prima_undine::Tensor],
            ) {
                use prima_undine::functions::BasicFunctions;
                let x = xs[0];
                let y = &mut ys[0];
                let size = y.shape().size();
                let g1 = super::common::calc_num_blocks(size as usize, self.wgs[0]);
                let kernel = self.kernel.lock().unwrap();
                unsafe {
                    ocl_core::set_kernel_arg(&kernel, 0, ocl_core::ArgVal::mem(buffer!(x)))
                        .unwrap();
                    ocl_core::set_kernel_arg(&kernel, 1, ocl_core::ArgVal::scalar(&size)).unwrap();
                    ocl_core::set_kernel_arg(&kernel, 2, ocl_core::ArgVal::mem(buffer!(y)))
                        .unwrap();
                    ocl_core::enqueue_kernel(
                        &self.internal.queue,
                        &kernel,
                        1,
                        None,
                        &[g1 * self.wgs[0], 1, 1],
                        Some([self.wgs[0], 1, 1]),
                        None::<ocl_core::Event>,
                        None::<&mut ocl_core::Event>,
                    )
                    .unwrap();
                }
            }
        }
    };
}

macro_rules! define_opencl_bw_x_impl {
    ( $name:ident, $kernel:ident ) => {
        define_opencl_impl_struct!($name, $kernel);
        impl prima_undine::device_impl::FunctionBwImpl for $name {
            fn call(
                &self,
                xs: &[&prima_undine::Tensor],
                ys: &[&prima_undine::Tensor],
                gys: &[&prima_undine::Tensor],
                _u32data: &[u32],
                _f32data: &[f32],
                gx: &mut prima_undine::Tensor,
            ) {
                use prima_undine::functions::BasicFunctions;
                let x = xs[0];
                let y = ys[0];
                let gy = gys[0];
                let size = y.shape().size();
                let g1 = super::common::calc_num_blocks(size as usize, self.wgs[0]);
                let kernel = self.kernel.lock().unwrap();
                unsafe {
                    ocl_core::set_kernel_arg(&kernel, 0, ocl_core::ArgVal::mem(buffer!(x)))
                        .unwrap();
                    ocl_core::set_kernel_arg(&kernel, 1, ocl_core::ArgVal::mem(buffer!(y)))
                        .unwrap();
                    ocl_core::set_kernel_arg(&kernel, 2, ocl_core::ArgVal::mem(buffer!(gy)))
                        .unwrap();
                    ocl_core::set_kernel_arg(&kernel, 3, ocl_core::ArgVal::scalar(&size)).unwrap();
                    ocl_core::set_kernel_arg(&kernel, 4, ocl_core::ArgVal::mem(buffer!(gx)))
                        .unwrap();
                    ocl_core::enqueue_kernel(
                        &self.internal.queue,
                        &kernel,
                        1,
                        None,
                        &[g1 * self.wgs[0], 1, 1],
                        Some([self.wgs[0], 1, 1]),
                        None::<ocl_core::Event>,
                        None::<&mut ocl_core::Event>,
                    )
                    .unwrap();
                }
            }
        }
    };
}

macro_rules! define_opencl_fw_ab_impl {
    ( $name:ident, $kernel:ident ) => {
        define_opencl_impl_struct!($name, $kernel);
        impl prima_undine::device_impl::FunctionFwImpl for $name {
            fn call(
                &self,
                xs: &[&prima_undine::Tensor],
                _u32data: &[u32],
                _f32data: &[f32],
                ys: &mut [&mut prima_undine::Tensor],
            ) {
                use prima_undine::functions::BasicFunctions;
                let a = xs[0];
                let b = xs[1];
                let y = &mut ys[0];
                let size = y.shape().volume();
                let g1 = super::common::calc_num_blocks(size as usize, self.wgs[0]);
                let g2 = y.shape().batch() as usize;
                let mba = a.shape().has_batch() as u32;
                let mbb = b.shape().has_batch() as u32;
                let kernel = self.kernel.lock().unwrap();
                unsafe {
                    ocl_core::set_kernel_arg(&kernel, 0, ocl_core::ArgVal::mem(buffer!(a)))
                        .unwrap();
                    ocl_core::set_kernel_arg(&kernel, 1, ocl_core::ArgVal::mem(buffer!(b)))
                        .unwrap();
                    ocl_core::set_kernel_arg(&kernel, 2, ocl_core::ArgVal::scalar(&size)).unwrap();
                    ocl_core::set_kernel_arg(&kernel, 3, ocl_core::ArgVal::scalar(&mba)).unwrap();
                    ocl_core::set_kernel_arg(&kernel, 4, ocl_core::ArgVal::scalar(&mbb)).unwrap();
                    ocl_core::set_kernel_arg(&kernel, 5, ocl_core::ArgVal::mem(buffer!(y)))
                        .unwrap();
                    ocl_core::enqueue_kernel(
                        &self.internal.queue,
                        &kernel,
                        2,
                        None,
                        &[g1 * self.wgs[0], g2, 1],
                        Some([self.wgs[0], 1, 1]),
                        None::<ocl_core::Event>,
                        None::<&mut ocl_core::Event>,
                    )
                    .unwrap();
                }
            }
        }
    };
}

macro_rules! define_opencl_bw_a_impl {
    ( $name:ident, $kernel:ident ) => {
        define_opencl_impl_struct!($name, $kernel);
        impl prima_undine::device_impl::FunctionBwImpl for $name {
            fn call(
                &self,
                xs: &[&prima_undine::Tensor],
                ys: &[&prima_undine::Tensor],
                gys: &[&prima_undine::Tensor],
                _u32data: &[u32],
                _f32data: &[f32],
                gx: &mut prima_undine::Tensor,
            ) {
                use prima_undine::functions::BasicFunctions;
                let a = xs[0];
                let b = xs[1];
                let y = ys[0];
                let gy = gys[0];
                let ga = gx;
                let size = y.shape().volume();
                let g1 = super::common::calc_num_blocks(size as usize, self.wgs[0]);
                let g2 = y.shape().batch() as usize;
                let mba = a.shape().has_batch() as u32;
                let mbb = b.shape().has_batch() as u32;
                let kernel = self.kernel.lock().unwrap();
                unsafe {
                    ocl_core::set_kernel_arg(&kernel, 0, ocl_core::ArgVal::mem(buffer!(a)))
                        .unwrap();
                    ocl_core::set_kernel_arg(&kernel, 1, ocl_core::ArgVal::mem(buffer!(b)))
                        .unwrap();
                    ocl_core::set_kernel_arg(&kernel, 2, ocl_core::ArgVal::mem(buffer!(y)))
                        .unwrap();
                    ocl_core::set_kernel_arg(&kernel, 3, ocl_core::ArgVal::mem(buffer!(gy)))
                        .unwrap();
                    ocl_core::set_kernel_arg(&kernel, 4, ocl_core::ArgVal::scalar(&size)).unwrap();
                    ocl_core::set_kernel_arg(&kernel, 5, ocl_core::ArgVal::scalar(&mba)).unwrap();
                    ocl_core::set_kernel_arg(&kernel, 6, ocl_core::ArgVal::scalar(&mbb)).unwrap();
                    ocl_core::set_kernel_arg(&kernel, 7, ocl_core::ArgVal::mem(buffer!(ga)))
                        .unwrap();
                    ocl_core::enqueue_kernel(
                        &self.internal.queue,
                        &kernel,
                        2,
                        None,
                        &[g1 * self.wgs[0], g2, 1],
                        Some([self.wgs[0], 1, 1]),
                        None::<ocl_core::Event>,
                        None::<&mut ocl_core::Event>,
                    )
                    .unwrap();
                }
            }
        }
    };
}

macro_rules! define_opencl_bw_b_impl {
    ( $name:ident, $kernel:ident ) => {
        define_opencl_impl_struct!($name, $kernel);
        impl prima_undine::device_impl::FunctionBwImpl for $name {
            fn call(
                &self,
                xs: &[&prima_undine::Tensor],
                ys: &[&prima_undine::Tensor],
                gys: &[&prima_undine::Tensor],
                _u32data: &[u32],
                _f32data: &[f32],
                gx: &mut prima_undine::Tensor,
            ) {
                use prima_undine::functions::BasicFunctions;
                let a = xs[0];
                let b = xs[1];
                let y = ys[0];
                let gy = gys[0];
                let gb = gx;
                let size = y.shape().volume();
                let g1 = super::common::calc_num_blocks(size as usize, self.wgs[0]);
                let g2 = y.shape().batch() as usize;
                let mba = a.shape().has_batch() as u32;
                let mbb = b.shape().has_batch() as u32;
                let kernel = self.kernel.lock().unwrap();
                unsafe {
                    ocl_core::set_kernel_arg(&*kernel, 0, ocl_core::ArgVal::mem(buffer!(a)))
                        .unwrap();
                    ocl_core::set_kernel_arg(&*kernel, 1, ocl_core::ArgVal::mem(buffer!(b)))
                        .unwrap();
                    ocl_core::set_kernel_arg(&*kernel, 2, ocl_core::ArgVal::mem(buffer!(y)))
                        .unwrap();
                    ocl_core::set_kernel_arg(&*kernel, 3, ocl_core::ArgVal::mem(buffer!(gy)))
                        .unwrap();
                    ocl_core::set_kernel_arg(&*kernel, 4, ocl_core::ArgVal::scalar(&size)).unwrap();
                    ocl_core::set_kernel_arg(&*kernel, 5, ocl_core::ArgVal::scalar(&mba)).unwrap();
                    ocl_core::set_kernel_arg(&*kernel, 6, ocl_core::ArgVal::scalar(&mbb)).unwrap();
                    ocl_core::set_kernel_arg(&*kernel, 7, ocl_core::ArgVal::mem(buffer!(gb)))
                        .unwrap();
                    ocl_core::enqueue_kernel(
                        &self.internal.queue,
                        &kernel,
                        2,
                        None,
                        &[g1 * self.wgs[0], g2, 1],
                        Some([self.wgs[0], 1, 1]),
                        None::<ocl_core::Event>,
                        None::<&mut ocl_core::Event>,
                    )
                    .unwrap();
                }
            }
        }
    };
}

macro_rules! define_opencl_fw_const_impl {
    ( $name:ident, $kernel:ident ) => {
        define_opencl_impl_struct!($name, $kernel);
        impl prima_undine::device_impl::FunctionFwImpl for $name {
            fn call(
                &self,
                xs: &[&prima_undine::Tensor],
                _u32data: &[u32],
                f32data: &[f32],
                ys: &mut [&mut prima_undine::Tensor],
            ) {
                use prima_undine::functions::BasicFunctions;
                let x = xs[0];
                let k = f32data[0];
                let y = &mut ys[0];
                let size = y.shape().size();
                let g1 = super::common::calc_num_blocks(size as usize, self.wgs[0]);
                let kernel = self.kernel.lock().unwrap();
                unsafe {
                    ocl_core::set_kernel_arg(&kernel, 0, ocl_core::ArgVal::mem(buffer!(x)))
                        .unwrap();
                    ocl_core::set_kernel_arg(&kernel, 1, ocl_core::ArgVal::scalar(&k)).unwrap();
                    ocl_core::set_kernel_arg(&kernel, 2, ocl_core::ArgVal::scalar(&size)).unwrap();
                    ocl_core::set_kernel_arg(&kernel, 3, ocl_core::ArgVal::mem(buffer!(y)))
                        .unwrap();
                    ocl_core::enqueue_kernel(
                        &self.internal.queue,
                        &kernel,
                        1,
                        None,
                        &[g1 * self.wgs[0], 1, 1],
                        Some([self.wgs[0], 1, 1]),
                        None::<ocl_core::Event>,
                        None::<&mut ocl_core::Event>,
                    )
                    .unwrap();
                }
            }
        }
    };
}

macro_rules! define_opencl_bw_const_impl {
    ( $name:ident, $kernel:ident ) => {
        define_opencl_impl_struct!($name, $kernel);
        impl prima_undine::device_impl::FunctionBwImpl for $name {
            fn call(
                &self,
                xs: &[&prima_undine::Tensor],
                ys: &[&prima_undine::Tensor],
                gys: &[&prima_undine::Tensor],
                _u32data: &[u32],
                f32data: &[f32],
                gx: &mut prima_undine::Tensor,
            ) {
                use prima_undine::functions::BasicFunctions;
                let x = xs[0];
                let y = ys[0];
                let gy = gys[0];
                let k = f32data[0];
                let size = y.shape().size();
                let g1 = super::common::calc_num_blocks(size as usize, self.wgs[0]);
                let kernel = self.kernel.lock().unwrap();
                unsafe {
                    ocl_core::set_kernel_arg(&kernel, 0, ocl_core::ArgVal::mem(buffer!(x)))
                        .unwrap();
                    ocl_core::set_kernel_arg(&kernel, 1, ocl_core::ArgVal::mem(buffer!(y)))
                        .unwrap();
                    ocl_core::set_kernel_arg(&kernel, 2, ocl_core::ArgVal::mem(buffer!(gy)))
                        .unwrap();
                    ocl_core::set_kernel_arg(&kernel, 3, ocl_core::ArgVal::scalar(&k)).unwrap();
                    ocl_core::set_kernel_arg(&kernel, 4, ocl_core::ArgVal::scalar(&size)).unwrap();
                    ocl_core::set_kernel_arg(&kernel, 5, ocl_core::ArgVal::mem(buffer!(gx)))
                        .unwrap();
                    ocl_core::enqueue_kernel(
                        &self.internal.queue,
                        &kernel,
                        1,
                        None,
                        &[g1 * self.wgs[0], 1, 1],
                        Some([self.wgs[0], 1, 1]),
                        None::<ocl_core::Event>,
                        None::<&mut ocl_core::Event>,
                    )
                    .unwrap();
                }
            }
        }
    };
}
