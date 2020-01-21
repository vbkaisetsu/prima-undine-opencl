use std::cmp;
use std::sync::Mutex;

use ocl::enums::KernelWorkGroupInfo::CompileWorkGroupSize;
use ocl::enums::KernelWorkGroupInfoResult::CompileWorkGroupSize as CompileWorkGroupSizeResult;
use ocl::Buffer;
use ocl::Kernel;
use ocl::Program;
use ocl::Queue;

use prima_undine::device_impl::FunctionFwImpl;
use prima_undine::Tensor;

pub struct SubAssignImpl {
    kernel: Mutex<Kernel>,
    wgs_x: u32,
}

impl SubAssignImpl {
    pub fn new(program: &Program, queue: Queue) -> SubAssignImpl {
        let device = queue.device();
        let kernel = Kernel::builder()
            .program(program)
            .name("sub_assign_kernel")
            .queue(queue)
            .arg(None::<&Buffer<f32>>)
            .arg(0)
            .arg(0)
            .arg(0)
            .arg(None::<&Buffer<f32>>)
            .build()
            .unwrap();
        match kernel.wg_info(device, CompileWorkGroupSize).unwrap() {
            CompileWorkGroupSizeResult([wgs_x, _, _]) => SubAssignImpl {
                kernel: Mutex::new(kernel),
                wgs_x: wgs_x as u32,
            },
            _ => panic!(),
        }
    }
}

impl FunctionFwImpl for SubAssignImpl {
    fn call(&self, xs: &[&Tensor], _u32data: &[u32], _f32data: &[f32], ys: &mut [&mut Tensor]) {
        let x = xs[0];
        let y = &mut ys[0];
        let size = y.shape().volume();
        let g1 = super::calc_num_blocks(size, self.wgs_x);
        let mbx = x.shape().has_batch() as u32;
        let mby = y.shape().has_batch() as u32;
        let bs = cmp::max(x.shape().batch(), y.shape().batch());
        let kernel = self.kernel.lock().unwrap();
        unsafe {
            kernel.set_arg(0, buffer!(x)).unwrap();
            kernel.set_arg(1, size).unwrap();
            kernel.set_arg(2, mbx).unwrap();
            kernel.set_arg(3, mby).unwrap();
            kernel.set_arg(4, buffer!(y)).unwrap();
            kernel
                .cmd()
                .global_work_size([g1 * self.wgs_x, bs, 1])
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
    fn sub_assign_test() {
        // y -= x
        struct TestCase(Shape, Vec<f32>, Shape, Vec<f32>);
        let test_cases = vec![
            TestCase(
                shape![2, 3; 2],
                vec![-6., -5., -4., -3., -2., -1., 0., 1., 2., 3., 4., 5.],
                shape![2, 3; 2],
                vec![7., 6., 5., 4., 3., 2., 1., 0., -1., -2., -3., -4.],
            ),
            TestCase(
                shape![2, 3],
                vec![-6., -5., -4., -3., -2., -1.],
                shape![2, 3; 2],
                vec![7., 6., 5., 4., 3., 2., 7., 6., 5., 4., 3., 2.],
            ),
            TestCase(
                shape![2, 3; 2],
                vec![-6., -5., -4., -3., -2., -1., 0., 1., 2., 3., 4., 5.],
                shape![2, 3],
                vec![7., 5., 3., 1., -1., -3.],
            ),
        ];
        let dev = get_device();
        for tc in &test_cases {
            let x = dev.new_tensor_by_slice(tc.0, &tc.1);
            let mut y = dev.new_tensor_by_constant(tc.2, 1.);
            dev.call_fw_impl("sub_assign_impl", &[&x], &[], &[], &mut [&mut y]);
            assert_vector_ulps_eq!(&tc.3, &y.to_vec());
        }
    }
}
