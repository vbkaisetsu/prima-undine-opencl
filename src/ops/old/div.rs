define_opencl_fw_ab_impl!(DivFwImpl, div_fw_kernel);
define_opencl_bw_a_impl!(DivBwAImpl, div_bw_a_kernel);
define_opencl_bw_b_impl!(DivBwBImpl, div_bw_b_kernel);
define_opencl_fw_const_impl!(DivConstLFwImpl, div_const_l_fw_kernel);
define_opencl_bw_const_impl!(DivConstLBwImpl, div_const_l_bw_kernel);
define_opencl_fw_const_impl!(DivConstRFwImpl, div_const_r_fw_kernel);
define_opencl_bw_const_impl!(DivConstRBwImpl, div_const_r_bw_kernel);
define_opencl_fw_scalar_impl!(DivScalarLFwImpl, div_scalar_l_fw_kernel);
define_opencl_fw_scalar_impl!(DivScalarRFwImpl, div_scalar_r_fw_kernel);

#[cfg(test)]
mod tests {
    use crate::test_utils::get_device;
    use prima_undine::shape;
    use prima_undine::Shape;

    #[test]
    fn div_fw_test() {
        // y = a / b
        struct TestCase(Shape, Vec<f32>, Shape, Vec<f32>, Shape, Vec<f32>);
        let test_cases = vec![
            TestCase(
                shape![2, 3; 2],
                vec![-6., -5., -4., -3., -2., -1., 0., 1., 2., 3., 4., 5.],
                shape![2, 3; 2],
                vec![-1., 1., -1., 1., -1., 1., -1., 1., -1., 1., -1., 1.],
                shape![2, 3; 2],
                vec![6., -5., 4., -3., 2., -1., 0., 1., -2., 3., -4., 5.],
            ),
            TestCase(
                shape![2, 3],
                vec![-6., -5., -4., -3., -2., -1.],
                shape![2, 3; 2],
                vec![-1., 1., -1., 1., -1., 1., -2., 2., -2., 2., -2., 2.],
                shape![2, 3; 2],
                vec![6., -5., 4., -3., 2., -1., 3., -2.5, 2., -1.5, 1., -0.5],
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
            dev.call_fw_impl("div_fw_impl", &[&a, &b], &[], &[], &mut [&mut y]);
            assert_vector_ulps_eq!(&tc.5, &y.to_vec());
        }
    }

    #[test]
    fn div_bw_a_test() {
        // y = a / b
        // dy/da = 1 / b
        struct TestCase(Shape, Vec<f32>, Shape, Vec<f32>, Shape, Vec<f32>);
        let test_cases = vec![
            TestCase(
                shape![2, 3; 2],
                vec![-1., 1., -1., 1., -1., 1., -2., 2., -2., 2., -2., 2.],
                shape![2, 3; 2],
                vec![7., -4., 5., -2., 3., 0., 1., 1.5, 0., 2.5, -1., 3.5],
                shape![2, 3; 2],
                vec![-6., -5., -4., -3., -2., -1., 0., 1., 2., 3., 4., 5.],
            ),
            TestCase(
                shape![2, 3; 2],
                vec![-1., 1., -1., 1., -1., 1., -2., 2., -2., 2., -2., 2.],
                shape![2, 3],
                vec![7., -3.5, 4., -0.5, 1., 2.5],
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
            dev.call_bw_impl("div_bw_a_impl", &[&a, &b], &[&y], &[&gy], &[], &[], &mut ga);
            assert_vector_ulps_eq!(&tc.3, &ga.to_vec());
        }
    }

    #[test]
    fn div_bw_b_test() {
        // y = a / b
        // dy/db = -y / b
        struct TestCase(Shape, Shape, Vec<f32>, Vec<f32>, Shape, Vec<f32>, Vec<f32>);
        let test_cases = vec![
            TestCase(
                shape![2, 3; 2],
                shape![2, 3; 2],
                vec![-1., 1., -1., 1., -1., 1., -2., 2., -2., 2., -2., 2.],
                vec![
                    -35., -24., -15., -8., -3., 0., 1., 0.5, -1., -3.5, -7., -11.5,
                ],
                shape![2, 3; 2],
                vec![6., -5., 4., -3., 2., -1., 0., 1., -2., 3., -4., 5.],
                vec![-6., -5., -4., -3., -2., -1., 0., 1., 2., 3., 4., 5.],
            ),
            TestCase(
                shape![2, 3; 2],
                shape![2, 3],
                vec![-1., 2., -1., 2., -1., 2.],
                vec![-35., -12., -19., -8., -19., -12.],
                shape![2, 3; 2],
                vec![6., -5., 4., -3., 2., -1., 0., 1., -2., 3., -4., 5.],
                vec![-6., -5., -4., -3., -2., -1., 0., 1., 2., 3., 4., 5.],
            ),
        ];
        let dev = get_device();
        for tc in &test_cases {
            let a = dev.new_tensor_by_constant(tc.0, std::f32::INFINITY);
            let b = dev.new_tensor_by_slice(tc.1, &tc.2);
            let y = dev.new_tensor_by_slice(tc.4, &tc.5);
            let gy = dev.new_tensor_by_slice(tc.4, &tc.6);
            let mut gb = dev.new_tensor_by_constant(tc.1, 1.);
            dev.call_bw_impl("div_bw_b_impl", &[&a, &b], &[&y], &[&gy], &[], &[], &mut gb);
            assert_vector_ulps_eq!(&tc.3, &gb.to_vec());
        }
    }

    #[test]
    fn div_const_l_fw_test() {
        // y = k / x
        struct TestCase(Shape, Vec<f32>, f32, Vec<f32>);
        let test_cases = vec![
            TestCase(
                shape![2, 3; 2],
                vec![-5., -4., -3., -2., -1., 0.5, 1., 2., 3., 4., 5., 6.],
                1.,
                vec![
                    -0.2,
                    -0.25,
                    -1. / 3.,
                    -0.5,
                    -1.,
                    2.,
                    1.,
                    0.5,
                    1. / 3.,
                    0.25,
                    0.2,
                    1. / 6.,
                ],
            ),
            TestCase(
                shape![2, 3],
                vec![-1., 3., -1., 3., -1., 3.],
                2.,
                vec![-2., 2. / 3., -2., 2. / 3., -2., 2. / 3.],
            ),
        ];
        let dev = get_device();
        for tc in &test_cases {
            let x = dev.new_tensor_by_slice(tc.0, &tc.1);
            let k = tc.2;
            let mut y = dev.new_tensor(tc.0);
            y.alloc();
            dev.call_fw_impl("div_const_l_fw_impl", &[&x], &[], &[k], &mut [&mut y]);
            assert_vector_ulps_eq!(&tc.3, &y.to_vec());
        }
    }

    #[test]
    fn div_const_l_bw_test() {
        // y = k / x
        // dy/dx = -y / x
        struct TestCase(Shape, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>);
        let test_cases = vec![
            TestCase(
                shape![2, 3; 2],
                vec![-5., -4., -3., -2., -1., 0.5, 1., 2., 3., 4., 5., 6.],
                vec![
                    26. / 25.,
                    15. / 16.,
                    10. / 9.,
                    3. / 4.,
                    2.,
                    -3.,
                    3.,
                    0.5,
                    11. / 9.,
                    7. / 8.,
                    27. / 25.,
                    17. / 18.,
                ],
                vec![
                    -0.2,
                    -1. / 4.,
                    -1. / 3.,
                    -0.5,
                    -1.,
                    2.,
                    1.,
                    0.5,
                    1. / 3.,
                    1. / 4.,
                    0.2,
                    1. / 6.,
                ],
                vec![-1., 1., -1., 1., -1., 1., -2., 2., -2., 2., -2., 2.],
            ),
            TestCase(
                shape![2, 3],
                vec![-1., 3., -1., 3., -1., 3.],
                vec![3., 7. / 9., 3., 7. / 9., 3., 7. / 9.],
                vec![-2., 2. / 3., -2., 2. / 3., -2., 2. / 3.],
                vec![-1., 1., -1., 1., -1., 1.],
            ),
        ];
        let dev = get_device();
        for tc in &test_cases {
            let x = dev.new_tensor_by_slice(tc.0, &tc.1);
            let k = std::f32::INFINITY;
            let y = dev.new_tensor_by_slice(tc.0, &tc.3);
            let gy = dev.new_tensor_by_slice(tc.0, &tc.4);
            let mut gx = dev.new_tensor_by_constant(tc.0, 1.);
            dev.call_bw_impl(
                "div_const_l_bw_impl",
                &[&x],
                &[&y],
                &[&gy],
                &[],
                &[k],
                &mut gx,
            );
            assert_vector_ulps_eq!(&tc.2, &gx.to_vec());
        }
    }

    #[test]
    fn div_const_r_fw_test() {
        // y = x / k
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
                vec![-1. / 2., 3. / 2., -1. / 2., 3. / 2., -1. / 2., 3. / 2.],
            ),
        ];
        let dev = get_device();
        for tc in &test_cases {
            let x = dev.new_tensor_by_slice(tc.0, &tc.1);
            let k = tc.2;
            let mut y = dev.new_tensor(tc.0);
            y.alloc();
            dev.call_fw_impl("div_const_r_fw_impl", &[&x], &[], &[k], &mut [&mut y]);
            assert_vector_ulps_eq!(&tc.3, &y.to_vec());
        }
    }

    #[test]
    fn div_const_r_bw_test() {
        // y = x / k
        // dy/dx = 1 / k
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
                vec![3. / 4., 7. / 4., 3. / 4., 7. / 4., 3. / 4., 7. / 4.],
                2.,
                vec![-1. / 2., 3. / 2., -1. / 2., 3. / 2., -1. / 2., 3. / 2.],
            ),
        ];
        let dev = get_device();
        for tc in &test_cases {
            let x = dev.new_tensor_by_constant(tc.0, std::f32::INFINITY);
            let k = tc.2;
            let y = dev.new_tensor_by_constant(tc.0, std::f32::INFINITY);
            let gy = dev.new_tensor_by_slice(tc.0, &tc.3);
            let mut gx = dev.new_tensor_by_constant(tc.0, 1.);
            dev.call_bw_impl(
                "div_const_r_bw_impl",
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
    fn div_scalar_l_fw_test() {
        // y = k / x
        struct TestCase(Shape, Vec<f32>, Shape, Vec<f32>, Shape, Vec<f32>);
        let test_cases = vec![
            TestCase(
                shape![2, 3; 2],
                vec![-6., -5., -4., -3., -2., -1., 0.5, 1., 2., 3., 4., 5.],
                shape![; 2],
                vec![-1., 1.],
                shape![2, 3; 2],
                vec![
                    1. / 6.,
                    0.2,
                    0.25,
                    1. / 3.,
                    0.5,
                    1.,
                    2.,
                    1.,
                    0.5,
                    1. / 3.,
                    0.25,
                    0.2,
                ],
            ),
            TestCase(
                shape![2, 3],
                vec![-6., -5., -4., -3., -2., -1.],
                shape![; 2],
                vec![-1., 1.],
                shape![2, 3; 2],
                vec![
                    1. / 6.,
                    0.2,
                    0.25,
                    1. / 3.,
                    0.5,
                    1.,
                    -1. / 6.,
                    -0.2,
                    -0.25,
                    -1. / 3.,
                    -0.5,
                    -1.,
                ],
            ),
            TestCase(
                shape![2, 3; 2],
                vec![-6., -5., -4., -3., -2., -1., 0.5, 1., 2., 3., 4., 5.],
                shape![],
                vec![0.5],
                shape![2, 3; 2],
                vec![
                    -1. / 12.,
                    -0.1,
                    -1. / 8.,
                    -1. / 6.,
                    -0.25,
                    -0.5,
                    1.,
                    0.5,
                    0.25,
                    1. / 6.,
                    1. / 8.,
                    0.1,
                ],
            ),
        ];
        let dev = get_device();
        for tc in &test_cases {
            let x = dev.new_tensor_by_slice(tc.0, &tc.1);
            let k = dev.new_tensor_by_slice(tc.2, &tc.3);
            let mut y = dev.new_tensor(tc.4);
            y.alloc();
            dev.call_fw_impl("div_scalar_l_fw_impl", &[&x, &k], &[], &[], &mut [&mut y]);
            assert_vector_ulps_eq!(&tc.5, &y.to_vec());
        }
    }

    #[test]
    fn div_scalar_r_fw_test() {
        // y = x / k
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
                vec![0.5],
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
            dev.call_fw_impl("div_scalar_r_fw_impl", &[&x, &k], &[], &[], &mut [&mut y]);
            assert_vector_ulps_eq!(&tc.5, &y.to_vec());
        }
    }
}
