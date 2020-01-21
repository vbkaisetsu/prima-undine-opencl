use std::sync::Arc;
use std::sync::Mutex;

use ocl_core::ArgVal;
use ocl_core::Event;
use ocl_core::Kernel;
use ocl_core::KernelWorkGroupInfo;
use ocl_core::KernelWorkGroupInfoResult;
use ocl_core::Program;

use prima_undine::device_impl::FunctionFwU32Impl;
use prima_undine::functions::BasicFunctions;
use prima_undine::Tensor;

pub struct ArgsortImpl {
    init_kernel: Mutex<Kernel>,
    main_kernel: Mutex<Kernel>,
    wgs: [usize; 3],
    internal: Arc<crate::OpenCLInternal>,
}

impl ArgsortImpl {
    pub fn new(program: &Program, internal: &Arc<crate::OpenCLInternal>) -> Self {
        let init_kernel = ocl_core::create_kernel(program, "init_argsort_kernel").unwrap();
        let main_kernel = ocl_core::create_kernel(program, "argsort_kernel").unwrap();
        match ocl_core::get_kernel_work_group_info(
            &main_kernel,
            internal.queue.device().unwrap(),
            KernelWorkGroupInfo::CompileWorkGroupSize,
        )
        .unwrap()
        {
            KernelWorkGroupInfoResult::CompileWorkGroupSize(wgs) => Self {
                init_kernel: Mutex::new(init_kernel),
                main_kernel: Mutex::new(main_kernel),
                wgs: wgs,
                internal: Arc::clone(internal),
            },
            _ => panic!(),
        }
    }
}

impl FunctionFwU32Impl for ArgsortImpl {
    fn call(&self, xs: &[&Tensor], u32data: &[u32], _f32data: &[f32], ys: &mut [u32]) {
        let x = xs[0];
        let dim = u32data[0];
        let s = x.shape();
        let skip = s.lower_volume(dim);
        let len = s[dim];
        let idx_len = {
            let mut idx_len = 1;
            while idx_len < len {
                idx_len <<= 1;
            }
            idx_len
        };
        let size = s.size();
        let idx_size = size / len * idx_len;
        let ret = unsafe {
            ocl_core::create_buffer(
                &self.internal.context,
                ocl_core::MEM_READ_WRITE,
                idx_size as usize,
                None::<&[u32]>,
            )
            .unwrap()
        };
        {
            let g1 = super::common::calc_num_blocks(idx_size as usize, self.wgs[0]);
            let kernel = self.init_kernel.lock().unwrap();
            unsafe {
                ocl_core::set_kernel_arg(&kernel, 0, ArgVal::scalar(&idx_size)).unwrap();
                ocl_core::set_kernel_arg(&kernel, 1, ArgVal::mem(&ret)).unwrap();
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
        {
            let g1 = super::common::calc_num_blocks(idx_size as usize / 2, self.wgs[0]);
            let kernel = self.main_kernel.lock().unwrap();
            let mut block_size = 2;
            while block_size <= idx_len {
                let mut dist = block_size >> 1;
                while dist >= 1 {
                    unsafe {
                        ocl_core::set_kernel_arg(&kernel, 0, ArgVal::mem(buffer!(x))).unwrap();
                        ocl_core::set_kernel_arg(&kernel, 1, ArgVal::scalar(&block_size)).unwrap();
                        ocl_core::set_kernel_arg(&kernel, 2, ArgVal::scalar(&dist)).unwrap();
                        ocl_core::set_kernel_arg(&kernel, 3, ArgVal::scalar(&skip)).unwrap();
                        ocl_core::set_kernel_arg(&kernel, 4, ArgVal::scalar(&len)).unwrap();
                        ocl_core::set_kernel_arg(&kernel, 5, ArgVal::scalar(&idx_len)).unwrap();
                        ocl_core::set_kernel_arg(&kernel, 6, ArgVal::scalar(&size)).unwrap();
                        ocl_core::set_kernel_arg(&kernel, 7, ArgVal::scalar(&idx_size)).unwrap();
                        ocl_core::set_kernel_arg(&kernel, 8, ArgVal::mem(&ret)).unwrap();
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
                    dist >>= 1;
                }
                block_size <<= 1;
            }
        }
        unsafe {
            super::common::read_buffer(&self.internal.queue, &ret, ys);
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::test_utils::get_device;
    use prima_undine::functions::BasicFunctions;
    use prima_undine::shape;
    use rand::seq::SliceRandom;

    #[test]
    fn check_argsort_dims() {
        let x_data = vec![
            10., 17., 6., 8., 16., 18., 14., 15., 4., 3., 13., 11., 9., 12., 2., 1., 5., 7.,
        ];
        let expected = vec![
            vec![
                6., 10., 17., 8., 16., 18., 4., 14., 15., 3., 11., 13., 2., 9., 12., 1., 5., 7.,
            ],
            vec![
                8., 15., 4., 10., 16., 6., 14., 17., 18., 1., 5., 2., 3., 12., 7., 9., 13., 11.,
            ],
            vec![
                10., 17., 6., 8., 16., 18., 14., 15., 4., 3., 13., 11., 9., 12., 2., 1., 5., 7.,
            ],
        ];
        let dev = get_device();
        let x = dev.new_tensor_by_slice(shape![3, 3; 2], &x_data);
        for &i in &[0, 1, 2] {
            let mut result = vec![0; x.shape().size() as usize];
            dev.call_fw_u32_impl("argsort_impl", &[&x], &[i], &[], &mut result);
            let sorted = result
                .iter()
                .map(|&i| x_data[i as usize])
                .collect::<Vec<f32>>();
            assert_eq!(expected[i as usize], sorted);
        }
    }

    #[test]
    fn check_index_sort_large() {
        let ns = vec![
            1, 2, 3, 15, 16, 17, 255, 256, 257, 1023, 1024, 1025, 2047, 2048, 2049, 65534, 65535,
            65536,
        ];
        let mut rng = rand::thread_rng();
        let dev = get_device();
        for &n in &ns {
            let mut x_data = (0..n).map(|x| x as f32).collect::<Vec<f32>>();
            let expected = x_data.clone();
            x_data.shuffle(&mut rng);
            let x = dev.new_tensor_by_slice(shape![n], &x_data);
            let mut result = vec![0; n as usize];
            dev.call_fw_u32_impl("argsort_impl", &[&x], &[0], &[], &mut result);
            let sorted = result
                .iter()
                .map(|&i| x_data[i as usize])
                .collect::<Vec<f32>>();
            assert_eq!(expected, sorted);
        }
    }
}
