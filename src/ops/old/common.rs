macro_rules! buffer {
    ( $tensor:expr) => {
        ($tensor.handle() as *const ocl::Buffer<f32>)
            .as_ref()
            .unwrap()
    };
}

macro_rules! define_empty_impl {
    ( $name:ident ) => {
        pub struct $name {}
        impl $name {
            pub fn new() -> $name {
                $name {}
            }
        }
    };
}

macro_rules! define_opencl_fw_x_impl {
    ( $name:ident, $kernel:ident ) => {
        pub struct $name {
            kernel: std::sync::Mutex<ocl::Kernel>,
            wgs_x: u32,
        }
        impl $name {
            pub fn new(program: &ocl::Program, queue: ocl::Queue) -> $name {
                let device = queue.device();
                let kernel = ocl::Kernel::builder()
                    .program(program)
                    .name(stringify!($kernel))
                    .queue(queue)
                    .arg(None::<&ocl::Buffer<f32>>)
                    .arg(0)
                    .arg(None::<&ocl::Buffer<f32>>)
                    .build()
                    .unwrap();
                match kernel
                    .wg_info(
                        device,
                        ocl::enums::KernelWorkGroupInfo::CompileWorkGroupSize,
                    )
                    .unwrap()
                {
                    ocl::enums::KernelWorkGroupInfoResult::CompileWorkGroupSize([wgs_x, _, _]) => {
                        $name {
                            kernel: std::sync::Mutex::new(kernel),
                            wgs_x: wgs_x as u32,
                        }
                    }
                    _ => panic!(),
                }
            }
        }
        impl prima_undine::device_impl::FunctionFwImpl for $name {
            fn call(
                &self,
                xs: &[&prima_undine::Tensor],
                _u32data: &[u32],
                _f32data: &[f32],
                ys: &mut [&mut prima_undine::Tensor],
            ) {
                let x = xs[0];
                let y = &mut ys[0];
                let size = x.shape().size();
                let g1 = super::calc_num_blocks(size, self.wgs_x);
                let kernel = self.kernel.lock().unwrap();
                unsafe {
                    kernel.set_arg(0, buffer!(x)).unwrap();
                    kernel.set_arg(1, size).unwrap();
                    kernel.set_arg(2, buffer!(y)).unwrap();
                    kernel
                        .cmd()
                        .global_work_size([g1 * self.wgs_x, 1, 1])
                        .local_work_size([self.wgs_x, 1, 1])
                        .enq()
                        .unwrap();
                }
            }
        }
    };
}

macro_rules! define_opencl_bw_x_impl {
    ( $name:ident, $kernel:ident ) => {
        pub struct $name {
            kernel: std::sync::Mutex<ocl::Kernel>,
            wgs_x: u32,
        }
        impl $name {
            pub fn new(program: &ocl::Program, queue: ocl::Queue) -> $name {
                let device = queue.device();
                let kernel = ocl::Kernel::builder()
                    .program(program)
                    .name(stringify!($kernel))
                    .queue(queue)
                    .arg(None::<&ocl::Buffer<f32>>)
                    .arg(None::<&ocl::Buffer<f32>>)
                    .arg(None::<&ocl::Buffer<f32>>)
                    .arg(0)
                    .arg(None::<&ocl::Buffer<f32>>)
                    .build()
                    .unwrap();
                match kernel
                    .wg_info(
                        device,
                        ocl::enums::KernelWorkGroupInfo::CompileWorkGroupSize,
                    )
                    .unwrap()
                {
                    ocl::enums::KernelWorkGroupInfoResult::CompileWorkGroupSize([wgs_x, _, _]) => {
                        $name {
                            kernel: std::sync::Mutex::new(kernel),
                            wgs_x: wgs_x as u32,
                        }
                    }
                    _ => panic!(),
                }
            }
        }
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
                let x = xs[0];
                let y = ys[0];
                let gy = gys[0];
                let size = x.shape().size();
                let g1 = super::calc_num_blocks(size, self.wgs_x);
                let kernel = self.kernel.lock().unwrap();
                unsafe {
                    kernel.set_arg(0, buffer!(x)).unwrap();
                    kernel.set_arg(1, buffer!(y)).unwrap();
                    kernel.set_arg(2, buffer!(gy)).unwrap();
                    kernel.set_arg(3, size).unwrap();
                    kernel.set_arg(4, buffer!(gx)).unwrap();
                    kernel
                        .cmd()
                        .global_work_size([g1 * self.wgs_x, 1, 1])
                        .local_work_size([self.wgs_x, 1, 1])
                        .enq()
                        .unwrap();
                }
            }
        }
    };
}

macro_rules! define_opencl_fw_ab_impl {
    ( $name:ident, $kernel:ident ) => {
        pub struct $name {
            kernel: std::sync::Mutex<ocl::Kernel>,
            wgs_x: u32,
        }
        impl $name {
            pub fn new(program: &ocl::Program, queue: ocl::Queue) -> $name {
                let device = queue.device();
                let kernel = ocl::Kernel::builder()
                    .program(program)
                    .name(stringify!($kernel))
                    .queue(queue)
                    .arg(None::<&ocl::Buffer<f32>>)
                    .arg(None::<&ocl::Buffer<f32>>)
                    .arg(0)
                    .arg(0)
                    .arg(0)
                    .arg(None::<&ocl::Buffer<f32>>)
                    .build()
                    .unwrap();
                match kernel
                    .wg_info(
                        device,
                        ocl::enums::KernelWorkGroupInfo::CompileWorkGroupSize,
                    )
                    .unwrap()
                {
                    ocl::enums::KernelWorkGroupInfoResult::CompileWorkGroupSize([wgs_x, _, _]) => {
                        $name {
                            kernel: std::sync::Mutex::new(kernel),
                            wgs_x: wgs_x as u32,
                        }
                    }
                    _ => panic!(),
                }
            }
        }
        impl prima_undine::device_impl::FunctionFwImpl for $name {
            fn call(
                &self,
                xs: &[&prima_undine::Tensor],
                _u32data: &[u32],
                _f32data: &[f32],
                ys: &mut [&mut prima_undine::Tensor],
            ) {
                let a = xs[0];
                let b = xs[1];
                let y = &mut ys[0];
                let size = y.shape().volume();
                let g1 = super::calc_num_blocks(size, self.wgs_x);
                let g2 = y.shape().batch();
                let mba = a.shape().has_batch() as u32;
                let mbb = b.shape().has_batch() as u32;
                let kernel = self.kernel.lock().unwrap();
                unsafe {
                    kernel.set_arg(0, buffer!(a)).unwrap();
                    kernel.set_arg(1, buffer!(b)).unwrap();
                    kernel.set_arg(2, size).unwrap();
                    kernel.set_arg(3, mba).unwrap();
                    kernel.set_arg(4, mbb).unwrap();
                    kernel.set_arg(5, buffer!(y)).unwrap();
                    kernel
                        .cmd()
                        .global_work_size([g1 * self.wgs_x, g2, 1])
                        .local_work_size([self.wgs_x, 1, 1])
                        .enq()
                        .unwrap();
                }
            }
        }
    };
}

macro_rules! define_opencl_bw_a_impl {
    ( $name:ident, $kernel:ident ) => {
        pub struct $name {
            kernel: std::sync::Mutex<ocl::Kernel>,
            wgs_x: u32,
        }
        impl $name {
            pub fn new(program: &ocl::Program, queue: ocl::Queue) -> $name {
                let device = queue.device();
                let kernel = ocl::Kernel::builder()
                    .program(program)
                    .name(stringify!($kernel))
                    .queue(queue)
                    .arg(None::<&ocl::Buffer<f32>>)
                    .arg(None::<&ocl::Buffer<f32>>)
                    .arg(None::<&ocl::Buffer<f32>>)
                    .arg(None::<&ocl::Buffer<f32>>)
                    .arg(0)
                    .arg(0)
                    .arg(0)
                    .arg(None::<&ocl::Buffer<f32>>)
                    .build()
                    .unwrap();
                match kernel
                    .wg_info(
                        device,
                        ocl::enums::KernelWorkGroupInfo::CompileWorkGroupSize,
                    )
                    .unwrap()
                {
                    ocl::enums::KernelWorkGroupInfoResult::CompileWorkGroupSize([wgs_x, _, _]) => {
                        $name {
                            kernel: std::sync::Mutex::new(kernel),
                            wgs_x: wgs_x as u32,
                        }
                    }
                    _ => panic!(),
                }
            }
        }
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
                let a = xs[0];
                let b = xs[1];
                let y = ys[0];
                let gy = gys[0];
                let ga = gx;
                let size = y.shape().volume();
                let g1 = super::calc_num_blocks(size, self.wgs_x);
                let g2 = y.shape().batch();
                let mba = a.shape().has_batch() as u32;
                let mbb = b.shape().has_batch() as u32;
                let kernel = self.kernel.lock().unwrap();
                unsafe {
                    kernel.set_arg(0, buffer!(a)).unwrap();
                    kernel.set_arg(1, buffer!(b)).unwrap();
                    kernel.set_arg(2, buffer!(y)).unwrap();
                    kernel.set_arg(3, buffer!(gy)).unwrap();
                    kernel.set_arg(4, size).unwrap();
                    kernel.set_arg(5, mba).unwrap();
                    kernel.set_arg(6, mbb).unwrap();
                    kernel.set_arg(7, buffer!(ga)).unwrap();
                    kernel
                        .cmd()
                        .global_work_size([g1 * self.wgs_x, g2, 1])
                        .local_work_size([self.wgs_x, 1, 1])
                        .enq()
                        .unwrap();
                }
            }
        }
    };
}

macro_rules! define_opencl_bw_b_impl {
    ( $name:ident, $kernel:ident ) => {
        pub struct $name {
            kernel: std::sync::Mutex<ocl::Kernel>,
            wgs_x: u32,
        }
        impl $name {
            pub fn new(program: &ocl::Program, queue: ocl::Queue) -> $name {
                let device = queue.device();
                let kernel = ocl::Kernel::builder()
                    .program(program)
                    .name(stringify!($kernel))
                    .queue(queue)
                    .arg(None::<&ocl::Buffer<f32>>)
                    .arg(None::<&ocl::Buffer<f32>>)
                    .arg(None::<&ocl::Buffer<f32>>)
                    .arg(None::<&ocl::Buffer<f32>>)
                    .arg(0)
                    .arg(0)
                    .arg(0)
                    .arg(None::<&ocl::Buffer<f32>>)
                    .build()
                    .unwrap();
                match kernel
                    .wg_info(
                        device,
                        ocl::enums::KernelWorkGroupInfo::CompileWorkGroupSize,
                    )
                    .unwrap()
                {
                    ocl::enums::KernelWorkGroupInfoResult::CompileWorkGroupSize([wgs_x, _, _]) => {
                        $name {
                            kernel: std::sync::Mutex::new(kernel),
                            wgs_x: wgs_x as u32,
                        }
                    }
                    _ => panic!(),
                }
            }
        }
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
                let a = xs[0];
                let b = xs[1];
                let y = ys[0];
                let gy = gys[0];
                let gb = gx;
                let size = y.shape().volume();
                let g1 = super::calc_num_blocks(size, self.wgs_x);
                let g2 = y.shape().batch();
                let mba = a.shape().has_batch() as u32;
                let mbb = b.shape().has_batch() as u32;
                let kernel = self.kernel.lock().unwrap();
                unsafe {
                    kernel.set_arg(0, buffer!(a)).unwrap();
                    kernel.set_arg(1, buffer!(b)).unwrap();
                    kernel.set_arg(2, buffer!(y)).unwrap();
                    kernel.set_arg(3, buffer!(gy)).unwrap();
                    kernel.set_arg(4, size).unwrap();
                    kernel.set_arg(5, mba).unwrap();
                    kernel.set_arg(6, mbb).unwrap();
                    kernel.set_arg(7, buffer!(gb)).unwrap();
                    kernel
                        .cmd()
                        .global_work_size([g1 * self.wgs_x, g2, 1])
                        .local_work_size([self.wgs_x, 1, 1])
                        .enq()
                        .unwrap();
                }
            }
        }
    };
}

macro_rules! define_opencl_fw_const_impl {
    ( $name:ident, $kernel:ident ) => {
        pub struct $name {
            kernel: std::sync::Mutex<ocl::Kernel>,
            wgs_x: u32,
        }
        impl $name {
            pub fn new(program: &ocl::Program, queue: ocl::Queue) -> $name {
                let device = queue.device();
                let kernel = ocl::Kernel::builder()
                    .program(program)
                    .name(stringify!($kernel))
                    .queue(queue)
                    .arg(None::<&ocl::Buffer<f32>>)
                    .arg(0)
                    .arg(0)
                    .arg(None::<&ocl::Buffer<f32>>)
                    .build()
                    .unwrap();
                match kernel
                    .wg_info(
                        device,
                        ocl::enums::KernelWorkGroupInfo::CompileWorkGroupSize,
                    )
                    .unwrap()
                {
                    ocl::enums::KernelWorkGroupInfoResult::CompileWorkGroupSize([wgs_x, _, _]) => {
                        $name {
                            kernel: std::sync::Mutex::new(kernel),
                            wgs_x: wgs_x as u32,
                        }
                    }
                    _ => panic!(),
                }
            }
        }
        impl prima_undine::device_impl::FunctionFwImpl for $name {
            fn call(
                &self,
                xs: &[&prima_undine::Tensor],
                _u32data: &[u32],
                f32data: &[f32],
                ys: &mut [&mut prima_undine::Tensor],
            ) {
                let x = xs[0];
                let k = f32data[0];
                let y = &mut ys[0];
                let size = x.shape().size();
                let g1 = super::calc_num_blocks(size, self.wgs_x);
                let kernel = self.kernel.lock().unwrap();
                unsafe {
                    kernel.set_arg(0, buffer!(x)).unwrap();
                    kernel.set_arg(1, k).unwrap();
                    kernel.set_arg(2, size).unwrap();
                    kernel.set_arg(3, buffer!(y)).unwrap();
                    kernel
                        .cmd()
                        .global_work_size([g1 * self.wgs_x, 1, 1])
                        .local_work_size([self.wgs_x, 1, 1])
                        .enq()
                        .unwrap();
                }
            }
        }
    };
}

macro_rules! define_opencl_bw_const_impl {
    ( $name:ident, $kernel:ident ) => {
        pub struct $name {
            kernel: std::sync::Mutex<ocl::Kernel>,
            wgs_x: u32,
        }
        impl $name {
            pub fn new(program: &ocl::Program, queue: ocl::Queue) -> $name {
                let device = queue.device();
                let kernel = ocl::Kernel::builder()
                    .program(program)
                    .name(stringify!($kernel))
                    .queue(queue)
                    .arg(None::<&ocl::Buffer<f32>>)
                    .arg(None::<&ocl::Buffer<f32>>)
                    .arg(None::<&ocl::Buffer<f32>>)
                    .arg(0)
                    .arg(0)
                    .arg(None::<&ocl::Buffer<f32>>)
                    .build()
                    .unwrap();
                match kernel
                    .wg_info(
                        device,
                        ocl::enums::KernelWorkGroupInfo::CompileWorkGroupSize,
                    )
                    .unwrap()
                {
                    ocl::enums::KernelWorkGroupInfoResult::CompileWorkGroupSize([wgs_x, _, _]) => {
                        $name {
                            kernel: std::sync::Mutex::new(kernel),
                            wgs_x: wgs_x as u32,
                        }
                    }
                    _ => panic!(),
                }
            }
        }
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
                let x = xs[0];
                let y = ys[0];
                let gy = gys[0];
                let k = f32data[0];
                let size = x.shape().size();
                let g1 = super::calc_num_blocks(size, self.wgs_x);
                let kernel = self.kernel.lock().unwrap();
                unsafe {
                    kernel.set_arg(0, buffer!(x)).unwrap();
                    kernel.set_arg(1, buffer!(y)).unwrap();
                    kernel.set_arg(2, buffer!(gy)).unwrap();
                    kernel.set_arg(3, k).unwrap();
                    kernel.set_arg(4, size).unwrap();
                    kernel.set_arg(5, buffer!(gx)).unwrap();
                    kernel
                        .cmd()
                        .global_work_size([g1 * self.wgs_x, 1, 1])
                        .local_work_size([self.wgs_x, 1, 1])
                        .enq()
                        .unwrap();
                }
            }
        }
    };
}

macro_rules! define_opencl_fw_scalar_impl {
    ( $name:ident, $kernel:ident ) => {
        pub struct $name {
            kernel: std::sync::Mutex<ocl::Kernel>,
            wgs_x: u32,
        }
        impl $name {
            pub fn new(program: &ocl::Program, queue: ocl::Queue) -> $name {
                let device = queue.device();
                let kernel = ocl::Kernel::builder()
                    .program(program)
                    .name(stringify!($kernel))
                    .queue(queue)
                    .arg(None::<&ocl::Buffer<f32>>)
                    .arg(None::<&ocl::Buffer<f32>>)
                    .arg(0)
                    .arg(0)
                    .arg(0)
                    .arg(None::<&ocl::Buffer<f32>>)
                    .build()
                    .unwrap();
                match kernel
                    .wg_info(
                        device,
                        ocl::enums::KernelWorkGroupInfo::CompileWorkGroupSize,
                    )
                    .unwrap()
                {
                    ocl::enums::KernelWorkGroupInfoResult::CompileWorkGroupSize([wgs_x, _, _]) => {
                        $name {
                            kernel: std::sync::Mutex::new(kernel),
                            wgs_x: wgs_x as u32,
                        }
                    }
                    _ => panic!(),
                }
            }
        }
        impl prima_undine::device_impl::FunctionFwImpl for $name {
            fn call(
                &self,
                xs: &[&prima_undine::Tensor],
                _u32data: &[u32],
                _f32data: &[f32],
                ys: &mut [&mut prima_undine::Tensor],
            ) {
                let x = xs[0];
                let k = xs[1];
                let y = &mut ys[0];
                let size = y.shape().volume();
                let g1 = super::calc_num_blocks(size, self.wgs_x);
                let g2 = y.shape().batch();
                let mbx = x.shape().has_batch() as u32;
                let mbk = k.shape().has_batch() as u32;
                let kernel = self.kernel.lock().unwrap();
                unsafe {
                    kernel.set_arg(0, buffer!(x)).unwrap();
                    kernel.set_arg(1, buffer!(k)).unwrap();
                    kernel.set_arg(2, size).unwrap();
                    kernel.set_arg(3, mbx).unwrap();
                    kernel.set_arg(4, mbk).unwrap();
                    kernel.set_arg(5, buffer!(y)).unwrap();
                    kernel
                        .cmd()
                        .global_work_size([g1 * self.wgs_x, g2, 1])
                        .local_work_size([self.wgs_x, 1, 1])
                        .enq()
                        .unwrap();
                }
            }
        }
    };
}
