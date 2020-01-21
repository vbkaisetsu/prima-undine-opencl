use std::sync::Arc;
use std::sync::Mutex;

use ocl_core::ArgVal;
use ocl_core::Event;
use ocl_core::Kernel;

use rand::RngCore;

use prima_undine::device_impl::FunctionFwImpl;
use prima_undine::functions::BasicFunctions;
use prima_undine::Tensor;

pub struct XORShiftRandomizer {
    pub rand_state: ocl_core::Mem,
}

impl XORShiftRandomizer {
    pub fn new(
        program: &ocl_core::Program,
        internal: &std::sync::Arc<crate::OpenCLInternal>,
    ) -> Self {
        let initialize_kernel =
            ocl_core::create_kernel(program, "initialize_xorshift_kernel").unwrap();
        match ocl_core::get_kernel_work_group_info(
            &initialize_kernel,
            internal.queue.device().unwrap(),
            ocl_core::KernelWorkGroupInfo::CompileWorkGroupSize,
        )
        .unwrap()
        {
            ocl_core::KernelWorkGroupInfoResult::CompileWorkGroupSize(wgs) => {
                let size = wgs[0];
                let mut rng = rand::thread_rng();
                let mut seeds = vec![0; size as usize];
                for i in 0..size as usize {
                    seeds[i] = rng.next_u32();
                }
                let seeds = unsafe {
                    ocl_core::create_buffer(
                        &internal.context,
                        ocl_core::MEM_READ_WRITE | ocl_core::MEM_COPY_HOST_PTR,
                        size,
                        Some(&seeds),
                    )
                    .unwrap()
                };
                let rand_state = unsafe {
                    ocl_core::create_buffer(
                        &internal.context,
                        ocl_core::MEM_READ_WRITE,
                        size * 4,
                        None::<&[u32]>,
                    )
                    .unwrap()
                };
                ocl_core::set_kernel_arg(&initialize_kernel, 0, ArgVal::mem(&seeds)).unwrap();
                ocl_core::set_kernel_arg(&initialize_kernel, 1, ArgVal::mem(&rand_state)).unwrap();
                unsafe {
                    ocl_core::enqueue_kernel(
                        &internal.queue,
                        &initialize_kernel,
                        1,
                        None,
                        &[size, 1, 1],
                        Some([size, 1, 1]),
                        None::<Event>,
                        None::<&mut Event>,
                    )
                    .unwrap();
                }
                Self {
                    rand_state: rand_state,
                }
            }
            _ => panic!(),
        }
    }
}

pub struct RandomBernoulliImpl {
    randomizer: Arc<Mutex<XORShiftRandomizer>>,
    kernel: Mutex<Kernel>,
    wgs: [usize; 3],
    internal: Arc<crate::OpenCLInternal>,
}

impl RandomBernoulliImpl {
    pub fn new(
        randomizer: &Arc<Mutex<XORShiftRandomizer>>,
        program: &ocl_core::Program,
        internal: &Arc<crate::OpenCLInternal>,
    ) -> Self {
        let kernel = ocl_core::create_kernel(program, "xorshift_bernoulli_kernel").unwrap();
        match ocl_core::get_kernel_work_group_info(
            &kernel,
            internal.queue.device().unwrap(),
            ocl_core::KernelWorkGroupInfo::CompileWorkGroupSize,
        )
        .unwrap()
        {
            ocl_core::KernelWorkGroupInfoResult::CompileWorkGroupSize(wgs) => Self {
                randomizer: Arc::clone(randomizer),
                kernel: Mutex::new(kernel),
                wgs: wgs,
                internal: Arc::clone(internal),
            },
            _ => panic!(),
        }
    }
}

impl FunctionFwImpl for RandomBernoulliImpl {
    fn call(&self, _xs: &[&Tensor], _u32data: &[u32], f32data: &[f32], ys: &mut [&mut Tensor]) {
        let p = f32data[0];
        let y = &mut ys[0];
        let size = y.shape().size();
        let randomizer = self.randomizer.lock().unwrap();
        let kernel = self.kernel.lock().unwrap();
        let g1 = super::common::calc_num_blocks(size as usize, self.wgs[0]);
        unsafe {
            ocl_core::set_kernel_arg(&kernel, 0, ArgVal::mem(&randomizer.rand_state)).unwrap();
            ocl_core::set_kernel_arg(&kernel, 1, ArgVal::scalar(&p)).unwrap();
            ocl_core::set_kernel_arg(&kernel, 2, ArgVal::scalar(&size)).unwrap();
            ocl_core::set_kernel_arg(&kernel, 3, ArgVal::mem(&buffer!(y))).unwrap();
            ocl_core::enqueue_kernel(
                &self.internal.queue,
                &kernel,
                1,
                None,
                &[g1 * self.wgs[0], 1, 1],
                Some([self.wgs[0], 1, 1]),
                None::<Event>,
                None::<&mut Event>,
            )
            .unwrap();
        }
    }
}

pub struct RandomUniformImpl {
    randomizer: Arc<Mutex<XORShiftRandomizer>>,
    kernel: Mutex<Kernel>,
    wgs: [usize; 3],
    internal: Arc<crate::OpenCLInternal>,
}

impl RandomUniformImpl {
    pub fn new(
        randomizer: &Arc<Mutex<XORShiftRandomizer>>,
        program: &ocl_core::Program,
        internal: &Arc<crate::OpenCLInternal>,
    ) -> Self {
        let kernel = ocl_core::create_kernel(program, "xorshift_uniform_kernel").unwrap();
        match ocl_core::get_kernel_work_group_info(
            &kernel,
            internal.queue.device().unwrap(),
            ocl_core::KernelWorkGroupInfo::CompileWorkGroupSize,
        )
        .unwrap()
        {
            ocl_core::KernelWorkGroupInfoResult::CompileWorkGroupSize(wgs) => Self {
                randomizer: Arc::clone(randomizer),
                kernel: Mutex::new(kernel),
                wgs: wgs,
                internal: Arc::clone(internal),
            },
            _ => panic!(),
        }
    }
}

impl FunctionFwImpl for RandomUniformImpl {
    fn call(&self, _xs: &[&Tensor], _u32data: &[u32], f32data: &[f32], ys: &mut [&mut Tensor]) {
        let lower = f32data[0];
        let upper = f32data[1];
        let y = &mut ys[0];
        let size = y.shape().size();
        let randomizer = self.randomizer.lock().unwrap();
        let kernel = self.kernel.lock().unwrap();
        let g1 = super::common::calc_num_blocks(size as usize, self.wgs[0]);
        unsafe {
            ocl_core::set_kernel_arg(&kernel, 0, ArgVal::mem(&randomizer.rand_state)).unwrap();
            ocl_core::set_kernel_arg(&kernel, 1, ArgVal::scalar(&lower)).unwrap();
            ocl_core::set_kernel_arg(&kernel, 2, ArgVal::scalar(&upper)).unwrap();
            ocl_core::set_kernel_arg(&kernel, 3, ArgVal::scalar(&size)).unwrap();
            ocl_core::set_kernel_arg(&kernel, 4, ArgVal::mem(&buffer!(y))).unwrap();
            ocl_core::enqueue_kernel(
                &self.internal.queue,
                &kernel,
                1,
                None,
                &[g1 * self.wgs[0], 1, 1],
                Some([self.wgs[0], 1, 1]),
                None::<Event>,
                None::<&mut Event>,
            )
            .unwrap();
        }
    }
}

pub struct RandomNormalImpl {
    randomizer: Arc<Mutex<XORShiftRandomizer>>,
    kernel: Mutex<Kernel>,
    wgs: [usize; 3],
    internal: Arc<crate::OpenCLInternal>,
}

impl RandomNormalImpl {
    pub fn new(
        randomizer: &Arc<Mutex<XORShiftRandomizer>>,
        program: &ocl_core::Program,
        internal: &Arc<crate::OpenCLInternal>,
    ) -> Self {
        let kernel = ocl_core::create_kernel(program, "xorshift_normal_kernel").unwrap();
        match ocl_core::get_kernel_work_group_info(
            &kernel,
            internal.queue.device().unwrap(),
            ocl_core::KernelWorkGroupInfo::CompileWorkGroupSize,
        )
        .unwrap()
        {
            ocl_core::KernelWorkGroupInfoResult::CompileWorkGroupSize(wgs) => Self {
                randomizer: Arc::clone(randomizer),
                kernel: Mutex::new(kernel),
                wgs: wgs,
                internal: Arc::clone(internal),
            },
            _ => panic!(),
        }
    }
}

impl FunctionFwImpl for RandomNormalImpl {
    fn call(&self, _xs: &[&Tensor], _u32data: &[u32], f32data: &[f32], ys: &mut [&mut Tensor]) {
        let mean = f32data[0];
        let sd = f32data[1];
        let y = &mut ys[0];
        let size = y.shape().size();
        let randomizer = self.randomizer.lock().unwrap();
        let kernel = self.kernel.lock().unwrap();
        let g1 = super::common::calc_num_blocks(size as usize, self.wgs[0]);
        unsafe {
            ocl_core::set_kernel_arg(&kernel, 0, ArgVal::mem(&randomizer.rand_state)).unwrap();
            ocl_core::set_kernel_arg(&kernel, 1, ArgVal::scalar(&mean)).unwrap();
            ocl_core::set_kernel_arg(&kernel, 2, ArgVal::scalar(&sd)).unwrap();
            ocl_core::set_kernel_arg(&kernel, 3, ArgVal::scalar(&size)).unwrap();
            ocl_core::set_kernel_arg(&kernel, 4, ArgVal::mem(&buffer!(y))).unwrap();
            ocl_core::enqueue_kernel(
                &self.internal.queue,
                &kernel,
                1,
                None,
                &[g1 * self.wgs[0], 1, 1],
                Some([self.wgs[0], 1, 1]),
                None::<Event>,
                None::<&mut Event>,
            )
            .unwrap();
        }
    }
}
