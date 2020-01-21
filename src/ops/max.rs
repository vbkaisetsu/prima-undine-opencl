use std::sync::Arc;
use std::sync::Mutex;

use ocl_core::ArgVal;
use ocl_core::Event;
use ocl_core::Kernel;
use ocl_core::Program;

use prima_undine::device_impl::FunctionBwImpl;
use prima_undine::device_impl::FunctionFwImpl;
use prima_undine::functions::BasicFunctions;
use prima_undine::Tensor;

pub struct MaxFwImpl {
    kernels: Vec<Mutex<Kernel>>,
    internal: Arc<crate::OpenCLInternal>,
}

impl MaxFwImpl {
    pub fn new(program: &Program, internal: &Arc<crate::OpenCLInternal>) -> MaxFwImpl {
        let kernels = (0..=10)
            .map(|i| {
                Mutex::new(
                    ocl_core::create_kernel(
                        program,
                        "max_fw_kernel_".to_string() + &(1 << i).to_string(),
                    )
                    .unwrap(),
                )
            })
            .collect();
        MaxFwImpl {
            kernels: kernels,
            internal: Arc::clone(internal),
        }
    }
}

impl FunctionFwImpl for MaxFwImpl {
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

pub struct MaxBwImpl {
    kernels: Vec<Mutex<Kernel>>,
    internal: Arc<crate::OpenCLInternal>,
}

impl MaxBwImpl {
    pub fn new(program: &Program, internal: &Arc<crate::OpenCLInternal>) -> MaxBwImpl {
        let kernels = (0..=10)
            .map(|i| {
                Mutex::new(
                    ocl_core::create_kernel(
                        program,
                        "max_bw_kernel_".to_string() + &(1 << i).to_string(),
                    )
                    .unwrap(),
                )
            })
            .collect();
        MaxBwImpl {
            kernels: kernels,
            internal: Arc::clone(internal),
        }
    }
}

impl FunctionBwImpl for MaxBwImpl {
    fn call(
        &self,
        xs: &[&Tensor],
        ys: &[&Tensor],
        gys: &[&Tensor],
        u32data: &[u32],
        _f32data: &[f32],
        gx: &mut Tensor,
    ) {
        let x = xs[0];
        let y = ys[0];
        let gy = gys[0];
        let dim = u32data[0];
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
                ocl_core::set_kernel_arg(&kernel, 1, ArgVal::mem(buffer!(y))).unwrap();
                ocl_core::set_kernel_arg(&kernel, 2, ArgVal::mem(buffer!(gy))).unwrap();
                ocl_core::set_kernel_arg(&kernel, 3, ArgVal::scalar(&s)).unwrap();
                ocl_core::set_kernel_arg(&kernel, 4, ArgVal::scalar(&n)).unwrap();
                ocl_core::set_kernel_arg(&kernel, 5, ArgVal::mem(buffer!(gx))).unwrap();
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
    use std::cmp;

    use crate::test_utils::get_device;
    use prima_undine::functions::BasicFunctions;
    use prima_undine::shape;
    use prima_undine::Shape;
    use rand::seq::SliceRandom;

    #[test]
    fn check_max_fw_dims() {
        struct TestCase(u32, Shape, Vec<f32>);
        let test_cases = vec![
            TestCase(0, shape![1, 3; 2], vec![2., 8., 5., -3., 0., -6.]),
            TestCase(1, shape![3, 1; 2], vec![6., 7., 8., 0., -1., -2.]),
            TestCase(
                2,
                shape![3, 3; 2],
                vec![
                    0., 1., 2., 6., 7., 8., 3., 4., 5., -3., -4., -5., 0., -1., -2., -6., -7., -8.,
                ],
            ),
        ];
        let x_data = vec![
            0., 1., 2., 6., 7., 8., 3., 4., 5., -3., -4., -5., 0., -1., -2., -6., -7., -8.,
        ];
        let dev = get_device();
        let x = dev.new_tensor_by_slice(shape![3, 3; 2], &x_data);
        for tc in &test_cases {
            let mut y = dev.new_tensor(tc.1);
            y.alloc();
            dev.call_fw_impl("max_fw_impl", &[&x], &[tc.0], &[], &mut [&mut y]);
            assert_vector_ulps_eq!(tc.2, y.to_vec());
        }
    }

    #[test]
    fn check_max_fw_large() {
        let ns = vec![
            1, 2, 3, 15, 16, 17, 255, 256, 257, 1023, 1024, 1025, 2047, 2048, 2049, 65535, 65536,
            65537,
        ];
        let mut rng = rand::thread_rng();
        let dev = get_device();
        for &n in &ns {
            let mut x_data = (0..n).map(|x| x as f32).collect::<Vec<f32>>();
            x_data.shuffle(&mut rng);
            let x = dev.new_tensor_by_slice(shape![n], &x_data);
            let mut y = dev.new_tensor(shape![]);
            y.alloc();
            dev.call_fw_impl("max_fw_impl", &[&x], &[0], &[], &mut [&mut y]);
            assert_vector_ulps_eq!(vec![(n - 1) as f32], y.to_vec());
        }
    }

    #[test]
    fn check_max_bw_dims() {
        struct TestCase(u32, Vec<f32>, Vec<f32>, Vec<f32>);
        let test_cases = vec![
            TestCase(
                0,
                vec![2., 8., 5., -3., 0., -6.],
                vec![1., 2., 6., 5., 3., 4.],
                vec![
                    1., 1., 2., 1., 1., 3., 1., 1., 7., 6., 1., 1., 4., 1., 1., 5., 1., 1.,
                ],
            ),
            TestCase(
                1,
                vec![6., 7., 8., 0., -1., -2.],
                vec![-1., 1., -2., 2., -3., 3.],
                vec![
                    1., 1., 1., 0., 2., -1., 1., 1., 1., 1., 1., 1., 3., -2., 4., 1., 1., 1.,
                ],
            ),
            TestCase(
                2,
                vec![
                    0., 1., 2., 6., 7., 8., 3., 4., 5., -3., -4., -5., 0., -1., -2., -6., -7., -8.,
                ],
                vec![
                    0., 1., 0., -1., 0., 1., 0., -1., 2., 1., 0., -1., 0., 1., 2., 3., 4., 6.,
                ],
                vec![
                    1., 2., 1., 0., 1., 2., 1., 0., 3., 2., 1., 0., 1., 2., 3., 4., 5., 7.,
                ],
            ),
        ];
        let x_data = vec![
            0., 1., 2., 6., 7., 8., 3., 4., 5., -3., -4., -5., 0., -1., -2., -6., -7., -8.,
        ];
        let r = shape![3, 3; 2];
        let dev = get_device();
        let x = dev.new_tensor_by_slice(r, &x_data);
        for tc in &test_cases {
            let s = r.resize_dim(tc.0, 1);
            let y = dev.new_tensor_by_slice(s, &tc.1);
            let gy = dev.new_tensor_by_slice(s, &tc.2);
            let mut gx = dev.new_tensor_by_constant(r, 1.);
            dev.call_bw_impl("max_bw_impl", &[&x], &[&y], &[&gy], &[tc.0], &[], &mut gx);
            assert_vector_ulps_eq!(tc.3, gx.to_vec());
        }
    }

    #[test]
    fn check_max_bw_large() {
        let ns = vec![
            1, 2, 3, 15, 16, 17, 255, 256, 257, 1023, 1024, 1025, 2047, 2048, 2049, 65534, 65535,
            65536,
        ];
        let mut rng = rand::thread_rng();
        let dev = get_device();
        for &n in &ns {
            let mut x_data = (0..n).map(|x| x as f32).collect::<Vec<f32>>();
            let mut gx_data = vec![1.; n as usize];
            x_data.shuffle(&mut rng);
            let pos = x_data.iter().position(|&x| x == (n - 1) as f32).unwrap();
            gx_data[pos] = 2.;
            let x = dev.new_tensor_by_slice(shape![n], &x_data);
            let y = dev.new_tensor_by_constant(shape![], (n - 1) as f32);
            let gy = dev.new_tensor_by_constant(shape![], 1.);
            let mut gx = dev.new_tensor_by_constant(shape![n], 1.);
            dev.call_bw_impl("max_bw_impl", &[&x], &[&y], &[&gy], &[0], &[], &mut gx);
            assert_vector_ulps_eq!(gx_data, gx.to_vec());
        }
    }

    #[test]
    fn check_max_bw_multiple_large() {
        let ns = vec![
            1, 2, 3, 15, 16, 17, 255, 256, 257, 1023, 1024, 1025, 2047, 2048, 2049, 65534, 65535,
            65536,
        ];
        let mut rng = rand::thread_rng();
        let dev = get_device();
        for &n in &ns {
            let mut x_data = (0..n).map(|x| x as f32).collect::<Vec<f32>>();
            for i in 0..cmp::min(10, n as usize) {
                x_data[i] = (n - 1) as f32;
            }
            let mut gx_data = vec![1.; n as usize];
            x_data.shuffle(&mut rng);
            let pos = x_data.iter().position(|&x| x == (n - 1) as f32).unwrap();
            gx_data[pos] = 2.;
            let x = dev.new_tensor_by_slice(shape![n], &x_data);
            let y = dev.new_tensor_by_constant(shape![], (n - 1) as f32);
            let gy = dev.new_tensor_by_constant(shape![], 1.);
            let mut gx = dev.new_tensor_by_constant(shape![n], 1.);
            dev.call_bw_impl("max_bw_impl", &[&x], &[&y], &[&gy], &[0], &[], &mut gx);
            assert_vector_ulps_eq!(gx_data, gx.to_vec());
        }
    }
}
