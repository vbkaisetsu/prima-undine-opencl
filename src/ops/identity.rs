use ocl_core::ArgVal;
use ocl_core::Event;

use prima_undine::device_impl::FunctionFwImpl;
use prima_undine::functions::BasicFunctions;
use prima_undine::Tensor;

define_opencl_impl_struct!(IdentityImpl, set_identity_kernel);
impl FunctionFwImpl for IdentityImpl {
    fn call(&self, _xs: &[&Tensor], _u32data: &[u32], _f32data: &[f32], ys: &mut [&mut Tensor]) {
        let y = &mut ys[0];
        let size = y.shape().volume();
        let skip = y.shape()[0] + 1;
        let g1 = super::common::calc_num_blocks(size as usize, self.wgs[0]);
        let queue = &self.internal.queue;
        let kernel = self.kernel.lock().unwrap();
        unsafe {
            ocl_core::set_kernel_arg(&kernel, 0, ArgVal::scalar(&size)).unwrap();
            ocl_core::set_kernel_arg(&kernel, 1, ArgVal::scalar(&skip)).unwrap();
            ocl_core::set_kernel_arg(&kernel, 2, ArgVal::mem(buffer!(y))).unwrap();
            ocl_core::enqueue_kernel(
                &queue,
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

#[cfg(test)]
mod tests {
    use crate::test_utils::get_device;
    use prima_undine::functions::BasicFunctions;
    use prima_undine::shape;
    use prima_undine::Shape;

    #[test]
    fn check_identity() {
        struct TestCase(Shape, Vec<f32>);
        let test_cases = vec![
            TestCase(shape![], vec![1.]),
            TestCase(shape![2, 2], vec![1., 0., 0., 1.]),
            TestCase(shape![3, 3], vec![1., 0., 0., 0., 1., 0., 0., 0., 1.]),
            TestCase(
                shape![4, 4],
                vec![
                    1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.,
                ],
            ),
        ];
        let dev = get_device();
        for tc in &test_cases {
            let mut y = dev.new_tensor(tc.0);
            y.alloc();
            dev.call_fw_impl("identity_impl", &[], &[], &[], &mut [&mut y]);
            assert_vector_ulps_eq!(tc.1, y.to_vec());
        }
    }
}
