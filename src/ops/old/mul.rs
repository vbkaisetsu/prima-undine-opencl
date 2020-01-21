define_opencl_fw_ab_impl!(MulFwImpl, mul_fw_kernel);
define_opencl_bw_a_impl!(MulBwAImpl, mul_bw_a_kernel);
define_opencl_bw_b_impl!(MulBwBImpl, mul_bw_b_kernel);
define_opencl_fw_const_impl!(MulConstFwImpl, mul_const_fw_kernel);
define_opencl_bw_const_impl!(MulConstBwImpl, mul_const_bw_kernel);
define_opencl_fw_scalar_impl!(MulScalarFwImpl, mul_scalar_fw_kernel);

#[cfg(test)]
mod tests {
    use crate::test_utils::get_device;
    use prima_undine::shape;
    use prima_undine::Shape;

    #[test]
    fn mul_fw_test() {
        // y = a * b
        struct TestCase(Shape, Vec<f32>, Shape, Vec<f32>, Shape, Vec<f32>);
        let test_cases = vec![
            TestCase(
                shape![2, 3; 2],
                vec![-6., -5., -4., -3., -2., -1., 0., 1., 2., 3., 4., 5.],
                shape![2, 3; 2],
                vec![-1., 1., -1., 1., -1., 1., -2., 2., -2., 2., -2., 2.],
                shape![2, 3; 2],
                vec![6., -5., 4., -3., 2., -1., 0., 2., -4., 6., -8., 10.],
            ),
            TestCase(
                shape![2, 3],
                vec![-6., -5., -4., -3., -2., -1.],
                shape![2, 3; 2],
                vec![-1., 1., -1., 1., -1., 1., -2., 2., -2., 2., -2., 2.],
                shape![2, 3; 2],
                vec![6., -5., 4., -3., 2., -1., 12., -10., 8., -6., 4., -2.],
            ),
            TestCase(
                shape![2, 3; 2],
                vec![-6., -5., -4., -3., -2., -1., 0., 1., 2., 3., 4., 5.],
                shape![2, 3],
                vec![-1., 1., -1., 1., -1., 1.],
                shape![2, 3; 2],
                vec![6., -5., 4., -3., 2., -1., 0., 1., -2., 3., -4., 5.],
            ),
        ];
        let dev = get_device();
        for tc in &test_cases {
            let a = dev.new_tensor_by_slice(tc.0, &tc.1);
            let b = dev.new_tensor_by_slice(tc.2, &tc.3);
            let mut y = dev.new_tensor(tc.4);
            y.alloc();
            dev.call_fw_impl("mul_fw_impl", &[&a, &b], &[], &[], &mut [&mut y]);
            assert_vector_ulps_eq!(&tc.5, &y.to_vec());
        }
    }

    #[test]
    fn mul_bw_a_test() {
        // y = a * b
        // dy/da = b
        struct TestCase(Shape, Vec<f32>, Shape, Vec<f32>, Shape, Vec<f32>);
        let test_cases = vec![
            TestCase(
                shape![2, 3; 2],
                vec![-1., 1., -1., 1., -1., 1., -2., 2., -2., 2., -2., 2.],
                shape![2, 3; 2],
                vec![7., -4., 5., -2., 3., 0., 1., 3., -3., 7., -7., 11.],
                shape![2, 3; 2],
                vec![-6., -5., -4., -3., -2., -1., 0., 1., 2., 3., 4., 5.],
            ),
            TestCase(
                shape![2, 3; 2],
                vec![-1., 1., -1., 1., -1., 1., -2., 2., -2., 2., -2., 2.],
                shape![2, 3],
                vec![7., -2., 1., 4., -5., 10.],
                shape![2, 3; 2],
                vec![-6., -5., -4., -3., -2., -1., 0., 1., 2., 3., 4., 5.],
            ),
            TestCase(
                shape![2, 3],
                vec![-1., 1., -1., 1., -1., 1.],
                shape![2, 3; 2],
                vec![7., -4., 5., -2., 3., 0., 1., 2., -1., 4., -3., 6.],
                shape![2, 3; 2],
                vec![-6., -5., -4., -3., -2., -1., 0., 1., 2., 3., 4., 5.],
            ),
        ];
        let dev = get_device();
        for tc in &test_cases {
            let a = dev.new_tensor_by_constant(tc.2, std::f32::INFINITY);
            let b = dev.new_tensor_by_slice(tc.0, &tc.1);
            let y = dev.new_tensor_by_constant(tc.4, std::f32::INFINITY);
            let gy = dev.new_tensor_by_slice(tc.4, &tc.5);
            let mut ga = dev.new_tensor_by_constant(tc.2, 1.);
            dev.call_bw_impl("mul_bw_a_impl", &[&a, &b], &[&y], &[&gy], &[], &[], &mut ga);
            assert_vector_ulps_eq!(&tc.3, &ga.to_vec());
        }
    }

    #[test]
    fn mul_bw_b_test() {
        // y = a * b
        // dy/db = a
        struct TestCase(Shape, Vec<f32>, Shape, Vec<f32>, Shape, Vec<f32>);
        let test_cases = vec![
            TestCase(
                shape![2, 3; 2],
                vec![-1., 1., -1., 1., -1., 1., -2., 2., -2., 2., -2., 2.],
                shape![2, 3; 2],
                vec![7., -4., 5., -2., 3., 0., 1., 3., -3., 7., -7., 11.],
                shape![2, 3; 2],
                vec![-6., -5., -4., -3., -2., -1., 0., 1., 2., 3., 4., 5.],
            ),
            TestCase(
                shape![2, 3; 2],
                vec![-1., 1., -1., 1., -1., 1., -2., 2., -2., 2., -2., 2.],
                shape![2, 3],
                vec![7., -2., 1., 4., -5., 10.],
                shape![2, 3; 2],
                vec![-6., -5., -4., -3., -2., -1., 0., 1., 2., 3., 4., 5.],
            ),
            TestCase(
                shape![2, 3],
                vec![-1., 1., -1., 1., -1., 1.],
                shape![2, 3; 2],
                vec![7., -4., 5., -2., 3., 0., 1., 2., -1., 4., -3., 6.],
                shape![2, 3; 2],
                vec![-6., -5., -4., -3., -2., -1., 0., 1., 2., 3., 4., 5.],
            ),
        ];
        let dev = get_device();
        for tc in &test_cases {
            let a = dev.new_tensor_by_slice(tc.0, &tc.1);
            let b = dev.new_tensor_by_constant(tc.2, std::f32::INFINITY);
            let y = dev.new_tensor_by_constant(tc.4, std::f32::INFINITY);
            let gy = dev.new_tensor_by_slice(tc.4, &tc.5);
            let mut gb = dev.new_tensor_by_constant(tc.2, 1.);
            dev.call_bw_impl("mul_bw_b_impl", &[&a, &b], &[&y], &[&gy], &[], &[], &mut gb);
            assert_vector_ulps_eq!(&tc.3, &gb.to_vec());
        }
    }

    #[test]
    fn mul_const_fw_test() {
        // y = x * k
        struct TestCase(Shape, Vec<f32>, f32, Vec<f32>);
        let test_cases = vec![
            TestCase(
                shape![2, 3; 2],
                vec![-5., -4., -3., -2., -1., 0., 1., 2., 3., 4., 5., 6.],
                1.,
                vec![-5., -4., -3., -2., -1., 0., 1., 2., 3., 4., 5., 6.],
            ),
            TestCase(
                shape![2, 3],
                vec![-1., 3., -1., 3., -1., 3.],
                2.,
                vec![-2., 6., -2., 6., -2., 6.],
            ),
        ];
        let dev = get_device();
        for tc in &test_cases {
            let x = dev.new_tensor_by_slice(tc.0, &tc.1);
            let k = tc.2;
            let mut y = dev.new_tensor(tc.0);
            y.alloc();
            dev.call_fw_impl("mul_const_fw_impl", &[&x], &[], &[k], &mut [&mut y]);
            assert_vector_ulps_eq!(&tc.3, &y.to_vec());
        }
    }

    #[test]
    fn mul_const_bw_test() {
        // y = x * k
        // gy/gx = k
        struct TestCase(Shape, Vec<f32>, f32, Vec<f32>);
        let test_cases = vec![
            TestCase(
                shape![2, 3; 2],
                vec![-4., -3., -2., -1., 0., 1., 2., 3., 4., 5., 6., 7.],
                1.,
                vec![-5., -4., -3., -2., -1., 0., 1., 2., 3., 4., 5., 6.],
            ),
            TestCase(
                shape![2, 3],
                vec![-3., 13., -3., 13., -3., 13.],
                2.,
                vec![-2., 6., -2., 6., -2., 6.],
            ),
        ];
        let dev = get_device();
        for tc in &test_cases {
            let x = dev.new_tensor_by_slice(tc.0, &tc.1);
            let k = tc.2;
            let y = dev.new_tensor_by_constant(tc.0, std::f32::INFINITY);
            let gy = dev.new_tensor_by_slice(tc.0, &tc.3);
            let mut gx = dev.new_tensor_by_constant(tc.0, 1.);
            dev.call_bw_impl(
                "mul_const_bw_impl",
                &[&x],
                &[&y],
                &[&gy],
                &[],
                &[k],
                &mut gx,
            );
            assert_vector_ulps_eq!(&tc.1, &gx.to_vec());
        }
    }

    #[test]
    fn mul_scalar_fw_test() {
        // y = x * k
        struct TestCase(Shape, Vec<f32>, Shape, Vec<f32>, Shape, Vec<f32>);
        let test_cases = vec![
            TestCase(
                shape![2, 3; 2],
                vec![-6., -5., -4., -3., -2., -1., 0., 1., 2., 3., 4., 5.],
                shape![; 2],
                vec![-1., 1.],
                shape![2, 3; 2],
                vec![6., 5., 4., 3., 2., 1., 0., 1., 2., 3., 4., 5.],
            ),
            TestCase(
                shape![2, 3],
                vec![-6., -5., -4., -3., -2., -1.],
                shape![; 2],
                vec![-1., 1.],
                shape![2, 3; 2],
                vec![6., 5., 4., 3., 2., 1., -6., -5., -4., -3., -2., -1.],
            ),
            TestCase(
                shape![2, 3; 2],
                vec![-6., -5., -4., -3., -2., -1., 0., 1., 2., 3., 4., 5.],
                shape![],
                vec![2.],
                shape![2, 3; 2],
                vec![-12., -10., -8., -6., -4., -2., 0., 2., 4., 6., 8., 10.],
            ),
        ];
        let dev = get_device();
        for tc in &test_cases {
            let x = dev.new_tensor_by_slice(tc.0, &tc.1);
            let k = dev.new_tensor_by_slice(tc.2, &tc.3);
            let mut y = dev.new_tensor(tc.4);
            y.alloc();
            dev.call_fw_impl("mul_scalar_fw_impl", &[&x, &k], &[], &[], &mut [&mut y]);
            assert_vector_ulps_eq!(&tc.5, &y.to_vec());
        }
    }
}
