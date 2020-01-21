use prima_undine::device_impl::FunctionFwImpl;
use prima_undine::Tensor;

define_empty_impl!(ResetTensorImpl);
impl FunctionFwImpl for ResetTensorImpl {
    fn call(&self, _xs: &[&Tensor], _u32data: &[u32], f32data: &[f32], ys: &mut [&mut Tensor]) {
        let k = f32data[0];
        let y = &mut ys[0];
        unsafe {
            buffer!(y).cmd().fill(k, None).enq().unwrap();
        }
    }
}

define_empty_impl!(ResetTensorBySliceImpl);
impl FunctionFwImpl for ResetTensorBySliceImpl {
    fn call(&self, _xs: &[&Tensor], _u32data: &[u32], f32data: &[f32], ys: &mut [&mut Tensor]) {
        let values = f32data;
        let y = &mut ys[0];
        unsafe {
            buffer!(y).write(values).enq().unwrap();
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::test_utils::get_device;
    use prima_undine::shape;
    use prima_undine::Shape;

    #[test]
    fn reset_tensor_test() {
        struct TestCase(f32, Shape, Vec<f32>);
        let test_cases = vec![
            TestCase(
                5.,
                shape![2, 3; 3],
                vec![
                    5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5.,
                ],
            ),
            TestCase(
                3.,
                shape![2, 3; 2],
                vec![3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3.],
            ),
            TestCase(2., shape![2, 3], vec![2., 2., 2., 2., 2., 2.]),
        ];
        let dev = get_device();
        for tc in &test_cases {
            let k = tc.0;
            let mut y = dev.new_tensor(tc.1);
            y.alloc();
            dev.call_fw_impl("reset_tensor_impl", &[], &[], &[k], &mut [&mut y]);
            assert_vector_ulps_eq!(&tc.2, &y.to_vec());
        }
    }

    #[test]
    fn reset_tensor_by_slice_test() {
        struct TestCase(Shape, Vec<f32>);
        let test_cases = vec![
            TestCase(
                shape![2, 3; 3],
                vec![
                    -9., -8., -7., -6., -5., -4., -3., -2., -1., 0., 1., 2., 3., 4., 5., 6., 7., 8.,
                ],
            ),
            TestCase(
                shape![2, 3; 2],
                vec![-6., -5., -4., -3., -2., -1., 0., 1., 2., 3., 4., 5.],
            ),
            TestCase(shape![2, 3], vec![-3., -2., -1., 0., 1., 2.]),
        ];
        let dev = get_device();
        for tc in &test_cases {
            let ks = &tc.1;
            let mut y = dev.new_tensor(tc.0);
            y.alloc();
            dev.call_fw_impl("reset_tensor_by_slice_impl", &[], &[], ks, &mut [&mut y]);
            assert_vector_ulps_eq!(&tc.1, &y.to_vec());
        }
    }
}
