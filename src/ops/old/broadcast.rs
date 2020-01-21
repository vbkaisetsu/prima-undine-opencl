use std::sync::Mutex;

use ocl::enums::KernelWorkGroupInfo::CompileWorkGroupSize;
use ocl::enums::KernelWorkGroupInfoResult::CompileWorkGroupSize as CompileWorkGroupSizeResult;
use ocl::Buffer;
use ocl::Kernel;
use ocl::Program;
use ocl::Queue;

use prima_undine::device_impl::FunctionFwImpl;
use prima_undine::Tensor;

pub struct BroadcastFwImpl {
    kernel: Mutex<Kernel>,
    wgs_x: u32,
}

impl BroadcastFwImpl {
    pub fn new(program: &Program, queue: Queue) -> BroadcastFwImpl {
        let device = queue.device();
        let kernel = Kernel::builder()
            .program(program)
            .name("broadcast_fw_kernel")
            .queue(queue)
            .arg(None::<&Buffer<f32>>)
            .arg(0)
            .arg(0)
            .arg(0)
            .arg(None::<&Buffer<f32>>)
            .build()
            .unwrap();
        match kernel.wg_info(device, CompileWorkGroupSize).unwrap() {
            CompileWorkGroupSizeResult([wgs_x, _, _]) => BroadcastFwImpl {
                kernel: Mutex::new(kernel),
                wgs_x: wgs_x as u32,
            },
            _ => panic!(),
        }
    }
}

impl FunctionFwImpl for BroadcastFwImpl {
    fn call(&self, xs: &[&Tensor], u32data: &[u32], _f32data: &[f32], ys: &mut [&mut Tensor]) {
        let x = xs[0];
        let dim = u32data[0];
        let size = u32data[1];
        let y = &mut ys[0];
        let skip1 = y.shape().lower_volume(dim);
        let skip2 = skip1 * size;
        let total = y.shape().size();
        let g1 = super::calc_num_blocks(size, self.wgs_x);
        let kernel = self.kernel.lock().unwrap();
        unsafe {
            kernel.set_arg(0, buffer!(x)).unwrap();
            kernel.set_arg(1, skip1).unwrap();
            kernel.set_arg(2, skip2).unwrap();
            kernel.set_arg(3, total).unwrap();
            kernel.set_arg(4, buffer!(y)).unwrap();
            kernel
                .cmd()
                .global_work_size([g1 * self.wgs_x, 1, 1])
                .local_work_size([self.wgs_x, 1, 1])
                .enq()
                .unwrap();
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::test_utils::get_device;
    use prima_undine::shape;
    use prima_undine::Shape;

    #[test]
    fn broadcast_fw_test() {
        struct TestCase(u32, u32, Shape, Vec<f32>, Shape, Vec<f32>);
        let test_cases = vec![
            TestCase(
                0,
                2,
                shape![1, 2, 3; 2],
                vec![-6., -5., -4., -3., -2., -1., 0., 1., 2., 3., 4., 5.],
                shape![2, 2, 3; 2],
                vec![
                    -6., -6., -5., -5., -4., -4., -3., -3., -2., -2., -1., -1., 0., 0., 1., 1., 2.,
                    2., 3., 3., 4., 4., 5., 5.,
                ],
            ),
            TestCase(
                1,
                3,
                shape![2, 1, 3],
                vec![-6., -5., -4., -3., -2., -1.],
                shape![2, 3, 3],
                vec![
                    -6., -5., -6., -5., -6., -5., -4., -3., -4., -3., -4., -3., -2., -1., -2., -1.,
                    -2., -1.,
                ],
            ),
            TestCase(
                2,
                2,
                shape![2, 3],
                vec![-6., -5., -4., -3., -2., -1.],
                shape![2, 3, 2],
                vec![-6., -5., -4., -3., -2., -1., -6., -5., -4., -3., -2., -1.],
            ),
        ];
        let dev = get_device();
        for tc in &test_cases {
            let dim = tc.0;
            let size = tc.1;
            let x = dev.new_tensor_by_slice(tc.2, &tc.3);
            let mut y = dev.new_tensor(tc.4);
            y.alloc();
            dev.call_fw_impl("broadcast_fw_impl", &[&x], &[dim, size], &[], &mut [&mut y]);
            assert_vector_ulps_eq!(&tc.5, &y.to_vec());
        }
    }
}
