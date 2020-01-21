define_opencl_fw_x_impl!(TanhFwImpl, tanh_fw_kernel);
define_opencl_bw_x_impl!(TanhBwImpl, tanh_bw_kernel);

#[cfg(test)]
mod tests {
    use crate::test_utils::get_device;
    use prima_undine::shape;
    use prima_undine::Shape;

    #[test]
    fn tanh_fw_test() {
        // y = tanh(x)
        let y_f = |x: f64| x.tanh();
        struct TestCase(Shape, Vec<f32>);
        let test_cases = vec![
            TestCase(
                shape![1, 2, 3; 2],
                vec![-6., -5., -4., -3., -2., -1., 0., 1., 2., 3., 4., 5.],
            ),
            TestCase(
                shape![2, 3, 3],
                vec![
                    -9., -8., -7., -6., -5., -4., -3., -2., -1., 0., 1., 2., 3., 4., 5., 6., 7., 8.,
                ],
            ),
            TestCase(
                shape![2, 3, 2],
                vec![-6., -5., -4., -3., -2., -1., 0., 1., 2., 3., 4., 5.],
            ),
        ];
        let dev = get_device();
        for tc in &test_cases {
            let y_data = generate_fw_testset!(tc.1, y_f);
            let x = dev.new_tensor_by_slice(tc.0, &tc.1);
            let mut y = dev.new_tensor(tc.0);
            y.alloc();
            dev.call_fw_impl("tanh_fw_impl", &[&x], &[], &[], &mut [&mut y]);
            assert_vector_ulps_eq!(&y_data, &y.to_vec());
        }
    }

    #[test]
    fn tanh_bw_test() {
        // y = tanh(x)
        // dy/dx = 1 - y^2
        let y_f = |x: f64| x.tanh();
        let gx_f = |_x: f64, y: f64, gy: f64| 1. + (1. - y * y) * gy;
        struct TestCase(Shape, Vec<f32>, Vec<f32>);
        let test_cases = vec![
            TestCase(
                shape![1, 2, 3; 2],
                vec![-6., -5., -4., -3., -2., -1., 0., 1., 2., 3., 4., 5.],
                vec![-1., -2., 1., 2., -1., -2., 1., 2., -1., -2., 1., 2.],
            ),
            TestCase(
                shape![2, 3, 3],
                vec![
                    -9., -8., -7., -6., -5., -4., -3., -2., -1., 0., 1., 2., 3., 4., 5., 6., 7., 8.,
                ],
                vec![
                    -1., -2., 1., 2., -1., -2., 1., 2., -1., -2., 1., 2., -1., -2., 1., 2., -1.,
                    -2.,
                ],
            ),
            TestCase(
                shape![2, 3, 2],
                vec![-6., -5., -4., -3., -2., -1., 0., 1., 2., 3., 4., 5.],
                vec![-1., -2., 1., 2., -1., -2., 1., 2., -1., -2., 1., 2.],
            ),
        ];
        let dev = get_device();
        for tc in &test_cases {
            let (y_data, gx_data) = generate_bw_testset!(tc.1, tc.2, y_f, gx_f);
            let x = dev.new_tensor_by_slice(tc.0, &tc.1);
            let y = dev.new_tensor_by_slice(tc.0, &y_data);
            let gy = dev.new_tensor_by_slice(tc.0, &tc.2);
            let mut gx = dev.new_tensor_by_constant(tc.0, 1.);
            dev.call_bw_impl("tanh_bw_impl", &[&x], &[&y], &[&gy], &[], &[], &mut gx);
            assert_vector_ulps_eq!(&gx_data, &gx.to_vec());
        }
    }
}
