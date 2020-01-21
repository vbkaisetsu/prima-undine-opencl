use std::sync::Arc;
use std::sync::Mutex;

use ocl_core::ArgVal;
use ocl_core::Event;
use ocl_core::Kernel;
use ocl_core::Program;

use prima_undine::device_impl::FunctionFwU32Impl;
use prima_undine::functions::BasicFunctions;
use prima_undine::Tensor;

pub struct ArgminImpl {
    kernels: Vec<Mutex<Kernel>>,
    internal: Arc<crate::OpenCLInternal>,
}

impl ArgminImpl {
    pub fn new(program: &Program, internal: &Arc<crate::OpenCLInternal>) -> ArgminImpl {
        let kernels = (0..=10)
            .map(|i| {
                Mutex::new(
                    ocl_core::create_kernel(
                        program,
                        "argmin_kernel_".to_string() + &(1 << i).to_string(),
                    )
                    .unwrap(),
                )
            })
            .collect();
        ArgminImpl {
            kernels: kernels,
            internal: Arc::clone(internal),
        }
    }
}

impl FunctionFwU32Impl for ArgminImpl {
    fn call(&self, xs: &[&Tensor], u32data: &[u32], _f32data: &[f32], ys: &mut [u32]) {
        let x = xs[0];
        let dim = u32data[0];
        let n = x.shape()[dim];
        let r = x.shape().size() / n;
        let s = x.shape().lower_volume(dim);
        // TODO
        let mut group_size = 256;
        while group_size >> 1 >= n {
            group_size >>= 1;
        }
        let ret = unsafe {
            ocl_core::create_buffer(
                &self.internal.context,
                ocl_core::MEM_READ_WRITE,
                r as usize,
                None::<&[u32]>,
            )
            .unwrap()
        };
        let case = |k, m: usize| {
            let kernel = self.kernels[m].lock().unwrap();
            unsafe {
                ocl_core::set_kernel_arg(&kernel, 0, ArgVal::mem(buffer!(x))).unwrap();
                ocl_core::set_kernel_arg(&kernel, 1, ArgVal::scalar(&s)).unwrap();
                ocl_core::set_kernel_arg(&kernel, 2, ArgVal::scalar(&n)).unwrap();
                ocl_core::set_kernel_arg(&kernel, 3, ArgVal::mem(&ret)).unwrap();
                ocl_core::enqueue_kernel(
                    &self.internal.queue,
                    &kernel,
                    1,
                    None,
                    &[r as usize * k, 1, 1],
                    Some([k, 1, 1]),
                    None::<Event>,
                    None::<&mut Event>,
                )
                .unwrap();
            }
        };
        match group_size {
            1024 => case(1024, 10),
            512 => case(512, 9),
            256 => case(256, 8),
            128 => case(128, 7),
            64 => case(64, 6),
            32 => case(32, 5),
            16 => case(16, 4),
            8 => case(8, 3),
            4 => case(4, 2),
            2 => case(2, 1),
            1 => case(1, 0),
            _ => panic!(),
        }
        unsafe {
            super::common::read_buffer(&self.internal.queue, &ret, ys);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::cmp;

    use crate::test_utils::get_device;
    use prima_undine::functions::BasicFunctions;
    use prima_undine::shape;
    use rand::seq::SliceRandom;

    #[test]
    fn check_argmin_dims() {
        let x_data = vec![
            3., 4., 5., 0., 1., 2., 6., 7., 8., 0., -1., -2., -6., -7., -8., -3., -4., -5.,
        ];
        let expected = vec![
            vec![0, 0, 0, 2, 2, 2],
            vec![1, 1, 1, 1, 1, 1],
            vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ];
        let dev = get_device();
        let x = dev.new_tensor_by_slice(shape![3, 3; 2], &x_data);
        for &i in &[0, 1, 2] {
            let mut result = vec![0; (x.shape().size() / x.shape()[i]) as usize];
            dev.call_fw_u32_impl("argmin_impl", &[&x], &[i], &[], &mut result);
            assert_eq!(expected[i as usize], result);
        }
    }

    #[test]
    fn check_argmin_large() {
        let ns = vec![
            1, 2, 3, 15, 16, 17, 255, 256, 257, 1023, 1024, 1025, 2047, 2048, 2049, 65534, 65535,
            65536,
        ];
        let mut rng = rand::thread_rng();
        let dev = get_device();
        for &n in &ns {
            let mut x_data = (0..n).map(|x| x as f32).collect::<Vec<f32>>();
            x_data.shuffle(&mut rng);
            let pos = x_data.iter().position(|&x| x == 0.).unwrap() as u32;
            let expected = vec![pos];
            let mut result = vec![0];
            let x = dev.new_tensor_by_slice(shape![n], &x_data);
            dev.call_fw_u32_impl("argmin_impl", &[&x], &[0], &[], &mut result);
            assert_eq!(expected, result);
        }
    }

    #[test]
    fn check_argmin_multiple_large() {
        let ns = vec![
            1, 2, 3, 15, 16, 17, 255, 256, 257, 1023, 1024, 1025, 2047, 2048, 2049, 65534, 65535,
            65536,
        ];
        let mut rng = rand::thread_rng();
        let dev = get_device();
        for &n in &ns {
            let mut x_data = (0..n).map(|x| x as f32).collect::<Vec<f32>>();
            for i in 0..cmp::min(10, n as usize) {
                x_data[i] = 0.;
            }
            x_data.shuffle(&mut rng);
            let pos = x_data.iter().position(|&x| x == 0.).unwrap() as u32;
            let expected = vec![pos];
            let mut result = vec![0];
            let x = dev.new_tensor_by_slice(shape![n], &x_data);
            dev.call_fw_u32_impl("argmin_impl", &[&x], &[0], &[], &mut result);
            assert_eq!(expected, result);
        }
    }
}
