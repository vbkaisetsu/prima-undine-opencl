use std::sync::Arc;
use std::sync::Mutex;

use ocl_core::ArgVal;
use ocl_core::Event;
use ocl_core::Kernel;
use ocl_core::Program;

use prima_undine::device_impl::FunctionFwImpl;
use prima_undine::functions::BasicFunctions;
use prima_undine::Tensor;

pub struct LogsumexpFwImpl {
    kernels: Vec<Mutex<Kernel>>,
    internal: Arc<crate::OpenCLInternal>,
}

impl LogsumexpFwImpl {
    pub fn new(program: &Program, internal: &Arc<crate::OpenCLInternal>) -> LogsumexpFwImpl {
        let kernels = (0..=10)
            .map(|i| {
                Mutex::new(
                    ocl_core::create_kernel(
                        program,
                        "logsumexp_fw_kernel_".to_string() + &(1 << i).to_string(),
                    )
                    .unwrap(),
                )
            })
            .collect();
        LogsumexpFwImpl {
            kernels: kernels,
            internal: Arc::clone(internal),
        }
    }
}

impl FunctionFwImpl for LogsumexpFwImpl {
    fn call(&self, xs: &[&Tensor], u32data: &[u32], _f32data: &[f32], ys: &mut [&mut Tensor]) {
        let x = xs[0];
        let dim = u32data[0];
        let y = &mut ys[0];
        let n = x.shape()[dim];
        let r = y.shape().size();
        let s = y.shape().lower_volume(dim);
        // TODO
        let mut group_size = 256;
        while group_size >> 1 >= n {
            group_size >>= 1;
        }
        let case = |k, m: usize| {
            let kernel = self.kernels[m].lock().unwrap();
            unsafe {
                ocl_core::set_kernel_arg(&kernel, 0, ArgVal::mem(buffer!(x))).unwrap();
                ocl_core::set_kernel_arg(&kernel, 1, ArgVal::scalar(&s)).unwrap();
                ocl_core::set_kernel_arg(&kernel, 2, ArgVal::scalar(&n)).unwrap();
                ocl_core::set_kernel_arg(&kernel, 3, ArgVal::mem(buffer!(y))).unwrap();
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
    }
}

#[cfg(test)]
mod tests {
    use crate::test_utils::get_device;
    use prima_undine::functions::BasicFunctions;
    use prima_undine::shape;

    #[test]
    fn check_logsumexp_fw() {
        let x_data = vec![
            1., 2., 3., 4., 5., 6., 7., 8., -1., -2., -3., -4., -5., -6., -7., -8.,
        ];
        let y_data = vec![
            vec![
                2.31326169,
                4.31326169,
                6.31326169,
                8.31326169,
                -0.68673831,
                -2.68673831,
                -4.68673831,
                -6.68673831,
            ],
            vec![
                3.12692801,
                4.12692801,
                7.12692801,
                8.12692801,
                -0.87307199,
                -1.87307199,
                -4.87307199,
                -5.87307199,
            ],
            vec![
                5.01814993,
                6.01814993,
                7.01814993,
                8.01814993,
                -0.98185007,
                -1.98185007,
                -2.98185007,
                -3.98185007,
            ],
            vec![
                1., 2., 3., 4., 5., 6., 7., 8., -1., -2., -3., -4., -5., -6., -7., -8.,
            ],
        ];
        let dev = get_device();
        let x = dev.new_tensor_by_slice(shape![2, 2, 2; 2], &x_data);
        for i in 0..4 {
            let y_shape = shape![2, 2, 2; 2].resize_dim(i, 1);
            let mut y = dev.new_tensor(y_shape);
            y.alloc();
            dev.call_fw_impl("logsumexp_fw_impl", &[&x], &[i], &[], &mut [&mut y]);
            assert_vector_ulps_eq!(y_data[i as usize], y.to_vec());
        }
    }

    #[test]
    fn check_logsumexp_fw_large() {
        let ns = vec![
            1, 2, 3, 15, 16, 17, 255, 256, 257, 1023, 1024, 1025, 2047, 2048, 2049, 65535, 65536,
            65537,
        ];
        let dev = get_device();
        for &n in &ns {
            for &k in &[-5., -1., 0., 1., 5.] {
                let x = dev.new_tensor_by_constant(shape![n], k);
                let mut y = dev.new_tensor(shape![]);
                y.alloc();
                dev.call_fw_impl("logsumexp_fw_impl", &[&x], &[0], &[], &mut [&mut y]);
                assert_vector_ulps_eq!(vec![k + (n as f32).ln()], y.to_vec());
            }
        }
    }
}
