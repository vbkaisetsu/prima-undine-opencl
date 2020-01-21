use std::cmp;
use std::sync::Mutex;

use ocl::Buffer;
use ocl::Kernel;
use ocl::Program;
use ocl::Queue;

use prima_undine::device_impl::FunctionFwImpl;
use prima_undine::Tensor;

pub struct SumFwImpl {
    kernels: Vec<Mutex<Kernel>>,
}

impl SumFwImpl {
    pub fn new(program: &Program, queue: Queue) -> SumFwImpl {
        let kernels = (0..=10)
            .map(|i| {
                Mutex::new(
                    Kernel::builder()
                        .program(program)
                        .name("sum_fw_kernel_".to_string() + &(1 << i).to_string())
                        .queue(queue.clone())
                        .arg(None::<&Buffer<f32>>)
                        .arg(0)
                        .arg(0)
                        .arg(None::<&Buffer<f32>>)
                        .build()
                        .unwrap(),
                )
            })
            .collect();
        SumFwImpl { kernels: kernels }
    }
}

impl FunctionFwImpl for SumFwImpl {
    fn call(&self, xs: &[&Tensor], u32data: &[u32], _f32data: &[f32], ys: &mut [&mut Tensor]) {
        let x = xs[0];
        let dim = u32data[0];
        let y = &mut ys[0];
        let n = x.shape()[dim];
        let r = y.shape().size();
        let s = y.shape().lower_volume(dim);
        let mut group_size = cmp::min(1024, 1024);
        while group_size >> 1 >= n {
            group_size >>= 1;
        }
        let case = |k, m: usize| unsafe {
            let kernel = self.kernels[m].lock().unwrap();
            kernel.set_arg(0, buffer!(x)).unwrap();
            kernel.set_arg(1, s).unwrap();
            kernel.set_arg(2, n).unwrap();
            kernel.set_arg(3, buffer!(y)).unwrap();
            kernel
                .cmd()
                .global_work_size([r * k, 1, 1])
                .local_work_size([k, 1, 1])
                .enq()
                .unwrap();
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
    use prima_undine::shape;
    use prima_undine::Shape;

    #[test]
    fn sum_fw_test() {
        struct TestCase(u32, Shape, Vec<f32>, Shape, Vec<f32>);
        let test_cases = vec![
            TestCase(
                0,
                shape![2, 2, 3; 2],
                vec![
                    -12., -11., -10., -9., -8., -7., -6., -5., -4., -3., -2., -1., 0., 1., 2., 3.,
                    4., 5., 6., 7., 8., 9., 10., 11.,
                ],
                shape![1, 2, 3; 2],
                vec![-23., -19., -15., -11., -7., -3., 1., 5., 9., 13., 17., 21.],
            ),
            TestCase(
                1,
                shape![2, 3, 3],
                vec![
                    -9., -8., -7., -6., -5., -4., -3., -2., -1., 0., 1., 2., 3., 4., 5., 6., 7., 8.,
                ],
                shape![2, 1, 3],
                vec![-21., -18., -3., 0., 15., 18.],
            ),
            TestCase(
                2,
                shape![2, 3, 2],
                vec![-6., -5., -4., -3., -2., -1., 0., 1., 2., 3., 4., 5.],
                shape![2, 3],
                vec![-6., -4., -2., 0., 2., 4.],
            ),
        ];
        let dev = get_device();
        for tc in &test_cases {
            let dim = tc.0;
            let x = dev.new_tensor_by_slice(tc.1, &tc.2);
            let mut y = dev.new_tensor(tc.3);
            y.alloc();
            dev.call_fw_impl("sum_fw_impl", &[&x], &[dim], &[], &mut [&mut y]);
            assert_vector_ulps_eq!(&tc.4, &y.to_vec());
        }
    }
}
