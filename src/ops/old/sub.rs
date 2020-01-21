define_opencl_fw_ab_impl!(SubFwImpl, sub_fw_kernel);
define_opencl_bw_a_impl!(SubBwAImpl, sub_bw_a_kernel);
define_opencl_bw_b_impl!(SubBwBImpl, sub_bw_b_kernel);
define_opencl_fw_const_impl!(SubConstLFwImpl, sub_const_l_fw_kernel);
define_opencl_bw_const_impl!(SubConstLBwImpl, sub_const_l_bw_kernel);
define_opencl_fw_const_impl!(SubConstRFwImpl, sub_const_r_fw_kernel);
define_opencl_bw_const_impl!(SubConstRBwImpl, sub_const_r_bw_kernel);
define_opencl_fw_scalar_impl!(SubScalarLFwImpl, sub_scalar_l_fw_kernel);
define_opencl_fw_scalar_impl!(SubScalarRFwImpl, sub_scalar_r_fw_kernel);

#[cfg(test)]
mod tests {
    use crate::test_utils::get_device;
    use prima_undine::shape;
    use prima_undine::Shape;

    #[test]
    fn sub_fw_test() {
        // y = a - b
        struct TestCase(Shape, Vec<f32>, Shape, Vec<f32>, Shape, Vec<f32>);
        let test_cases = vec![
            TestCase(
                shape![2, 3; 2],
                vec![-6., -5., -4., -3., -2., -1., 0., 1., 2., 3., 4., 5.],
                shape![2, 3; 2],
                vec![-1., 1., -1., 1., -1., 1., -1., 1., -1., 1., -1., 1.],
                shape![2, 3; 2],
                vec![-5., -6., -3., -4., -1., -2., 1., 0., 3., 2., 5., 4.],
            ),
            TestCase(
                shape![2, 3],
                vec![-6., -5., -4., -3., -2., -1.],
                shape![2, 3; 2],
                vec![-1., 1., -1., 1., -1., 1., -2., 2., -2., 2., -2., 2.],
                shape![2, 3; 2],
                vec![-5., -6., -3., -4., -1., -2., -4., -7., -2., -5., 0., -3.],
            ),
            TestCase(
                shape![2, 3; 2],
                vec![-6., -5., -4., -3., -2., -1., 0., 1., 2., 3., 4., 5.],
                shape![2, 3],
                vec![-1., 1., -1., 1., -1., 1.],
                shape![2, 3; 2],
                vec![-5., -6., -3., -4., -1., -2., 1., 0., 3., 2., 5., 4.],
            ),
        ];
        let dev = get_device();
        for tc in &test_cases {
            let a = dev.new_tensor_by_slice(tc.0, &tc.1);
            let b = dev.new_tensor_by_slice(tc.2, &tc.3);
            let mut y = dev.new_tensor(tc.4);
            y.alloc();
            dev.call_fw_impl("sub_fw_impl", &[&a, &b], &[], &[], &mut [&mut y]);
            assert_vector_ulps_eq!(&tc.5, &y.to_vec());
        }
    }

    #[test]
    fn sub_bw_a_test() {
        // y = a - b
        // dy/da = 1
        struct TestCase(Shape, Vec<f32>, Shape, Shape, Vec<f32>);
        let test_cases = vec![
            TestCase(
                shape![2, 3; 2],
                vec![-5., -4., -3., -2., -1., 0., 1., 2., 3., 4., 5., 6.],
                shape![2, 3],
                shape![2, 3; 2],
                vec![-6., -5., -4., -3., -2., -1., 0., 1., 2., 3., 4., 5.],
            ),
            TestCase(
                shape![2, 3],
                vec![-2., 4., -2., 4., -2., 4.],
                shape![2, 3; 2],
                shape![2, 3; 2],
                vec![-1., 1., -1., 1., -1., 1., -2., 2., -2., 2., -2., 2.],
            ),
        ];
        let dev = get_device();
        for tc in &test_cases {
            let a = dev.new_tensor_by_constant(tc.0, std::f32::INFINITY);
            let b = dev.new_tensor_by_constant(tc.2, std::f32::INFINITY);
            let y = dev.new_tensor_by_constant(tc.3, std::f32::INFINITY);
            let gy = dev.new_tensor_by_slice(tc.3, &tc.4);
            let mut ga = dev.new_tensor_by_constant(tc.0, 1.);
            dev.call_bw_impl("sub_bw_a_impl", &[&a, &b], &[&y], &[&gy], &[], &[], &mut ga);
            assert_vector_ulps_eq!(&tc.1, &ga.to_vec());
        }
    }

    #[test]
    fn sub_bw_b_test() {
        // y = a - b
        // dy/db = -1
        struct TestCase(Shape, Shape, Vec<f32>, Shape, Vec<f32>);
        let test_cases = vec![
            TestCase(
                shape![2, 3],
                shape![2, 3; 2],
                vec![7., 6., 5., 4., 3., 2., 1., 0., -1., -2., -3., -4.],
                shape![2, 3; 2],
                vec![-6., -5., -4., -3., -2., -1., 0., 1., 2., 3., 4., 5.],
            ),
            TestCase(
                shape![2, 3; 2],
                shape![2, 3],
                vec![4., -2., 4., -2., 4., -2.],
                shape![2, 3; 2],
                vec![-1., 1., -1., 1., -1., 1., -2., 2., -2., 2., -2., 2.],
            ),
        ];
        let dev = get_device();
        for tc in &test_cases {
            let a = dev.new_tensor_by_constant(tc.0, std::f32::INFINITY);
            let b = dev.new_tensor_by_constant(tc.1, std::f32::INFINITY);
            let y = dev.new_tensor_by_constant(tc.3, std::f32::INFINITY);
            let gy = dev.new_tensor_by_slice(tc.3, &tc.4);
            let mut gb = dev.new_tensor_by_constant(tc.1, 1.);
            dev.call_bw_impl("sub_bw_b_impl", &[&a, &b], &[&y], &[&gy], &[], &[], &mut gb);
            assert_vector_ulps_eq!(&tc.2, &gb.to_vec());
        }
    }

    #[test]
    fn sub_const_l_fw_test() {
        // y = k - x
        struct TestCase(Shape, Vec<f32>, f32, Vec<f32>);
        let test_cases = vec![
            TestCase(
                shape![2, 3; 2],
                vec![-5., -4., -3., -2., -1., 0., 1., 2., 3., 4., 5., 6.],
                1.,
                vec![6., 5., 4., 3., 2., 1., 0., -1., -2., -3., -4., -5.],
            ),
            TestCase(
                shape![2, 3],
                vec![-1., 3., -1., 3., -1., 3.],
                2.,
                vec![3., -1., 3., -1., 3., -1.],
            ),
        ];
        let dev = get_device();
        for tc in &test_cases {
            let x = dev.new_tensor_by_slice(tc.0, &tc.1);
            let k = tc.2;
            let mut y = dev.new_tensor(tc.0);
            y.alloc();
            dev.call_fw_impl("sub_const_l_fw_impl", &[&x], &[], &[k], &mut [&mut y]);
            assert_vector_ulps_eq!(&tc.3, &y.to_vec());
        }
    }

    #[test]
    fn sub_const_l_bw_test() {
        // y = k - x
        // dy/dx = -1
        struct TestCase(Shape, Vec<f32>, Vec<f32>);
        let test_cases = vec![
            TestCase(
                shape![2, 3; 2],
                vec![6., 5., 4., 3., 2., 1., 0., -1., -2., -3., -4., -5.],
                vec![-5., -4., -3., -2., -1., 0., 1., 2., 3., 4., 5., 6.],
            ),
            TestCase(
                shape![2, 3],
                vec![2., -2., 2., -2., 2., -2.],
                vec![-1., 3., -1., 3., -1., 3.],
            ),
        ];
        let dev = get_device();
        for tc in &test_cases {
            let x = dev.new_tensor_by_constant(tc.0, std::f32::INFINITY);
            let k = std::f32::INFINITY;
            let y = dev.new_tensor_by_constant(tc.0, std::f32::INFINITY);
            let gy = dev.new_tensor_by_slice(tc.0, &tc.2);
            let mut gx = dev.new_tensor_by_constant(tc.0, 1.);
            dev.call_bw_impl(
                "sub_const_l_bw_impl",
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
    fn sub_const_r_fw_test() {
        // y = x - k
        struct TestCase(Shape, Vec<f32>, f32, Vec<f32>);
        let test_cases = vec![
            TestCase(
                shape![2, 3; 2],
                vec![-5., -4., -3., -2., -1., 0., 1., 2., 3., 4., 5., 6.],
                1.,
                vec![-6., -5., -4., -3., -2., -1., 0., 1., 2., 3., 4., 5.],
            ),
            TestCase(
                shape![2, 3],
                vec![-1., 3., -1., 3., -1., 3.],
                2.,
                vec![-3., 1., -3., 1., -3., 1.],
            ),
        ];
        let dev = get_device();
        for tc in &test_cases {
            let x = dev.new_tensor_by_slice(tc.0, &tc.1);
            let k = tc.2;
            let mut y = dev.new_tensor(tc.0);
            y.alloc();
            dev.call_fw_impl("sub_const_r_fw_impl", &[&x], &[], &[k], &mut [&mut y]);
            assert_vector_ulps_eq!(&tc.3, &y.to_vec());
        }
    }

    #[test]
    fn sub_const_r_bw_test() {
        // y = x - k
        // dy/dx = 1
        struct TestCase(Shape, Vec<f32>, Vec<f32>);
        let test_cases = vec![
            TestCase(
                shape![2, 3; 2],
                vec![-4., -3., -2., -1., 0., 1., 2., 3., 4., 5., 6., 7.],
                vec![-5., -4., -3., -2., -1., 0., 1., 2., 3., 4., 5., 6.],
            ),
            TestCase(
                shape![2, 3],
                vec![0., 4., 0., 4., 0., 4.],
                vec![-1., 3., -1., 3., -1., 3.],
            ),
        ];
        let dev = get_device();
        for tc in &test_cases {
            let x = dev.new_tensor_by_constant(tc.0, std::f32::INFINITY);
            let k = std::f32::INFINITY;
            let y = dev.new_tensor_by_constant(tc.0, std::f32::INFINITY);
            let gy = dev.new_tensor_by_slice(tc.0, &tc.2);
            let mut gx = dev.new_tensor_by_constant(tc.0, 1.);
            dev.call_bw_impl(
                "sub_const_r_bw_impl",
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
    fn sub_scalar_l_fw_test() {
        // y = k - x
        struct TestCase(Shape, Vec<f32>, Shape, Vec<f32>, Shape, Vec<f32>);
        let test_cases = vec![
            TestCase(
                shape![2, 3; 2],
                vec![-6., -5., -4., -3., -2., -1., 0., 1., 2., 3., 4., 5.],
                shape![; 2],
                vec![-1., 1.],
                shape![2, 3; 2],
                vec![5., 4., 3., 2., 1., 0., 1., 0., -1., -2., -3., -4.],
            ),
            TestCase(
                shape![2, 3],
                vec![-6., -5., -4., -3., -2., -1.],
                shape![; 2],
                vec![-1., 1.],
                shape![2, 3; 2],
                vec![5., 4., 3., 2., 1., 0., 7., 6., 5., 4., 3., 2.],
            ),
            TestCase(
                shape![2, 3; 2],
                vec![-6., -5., -4., -3., -2., -1., 0., 1., 2., 3., 4., 5.],
                shape![],
                vec![1.],
                shape![2, 3; 2],
                vec![7., 6., 5., 4., 3., 2., 1., 0., -1., -2., -3., -4.],
            ),
        ];
        let dev = get_device();
        for tc in &test_cases {
            let x = dev.new_tensor_by_slice(tc.0, &tc.1);
            let k = dev.new_tensor_by_slice(tc.2, &tc.3);
            let mut y = dev.new_tensor(tc.4);
            y.alloc();
            dev.call_fw_impl("sub_scalar_l_fw_impl", &[&x, &k], &[], &[], &mut [&mut y]);
            assert_vector_ulps_eq!(&tc.5, &y.to_vec());
        }
    }

    #[test]
    fn sub_scalar_r_fw_test() {
        // y = x - k
        struct TestCase(Shape, Vec<f32>, Shape, Vec<f32>, Shape, Vec<f32>);
        let test_cases = vec![
            TestCase(
                shape![2, 3; 2],
                vec![-6., -5., -4., -3., -2., -1., 0., 1., 2., 3., 4., 5.],
                shape![; 2],
                vec![-1., 1.],
                shape![2, 3; 2],
                vec![-5., -4., -3., -2., -1., 0., -1., 0., 1., 2., 3., 4.],
            ),
            TestCase(
                shape![2, 3],
                vec![-6., -5., -4., -3., -2., -1.],
                shape![; 2],
                vec![-1., 1.],
                shape![2, 3; 2],
                vec![-5., -4., -3., -2., -1., 0., -7., -6., -5., -4., -3., -2.],
            ),
            TestCase(
                shape![2, 3; 2],
                vec![-6., -5., -4., -3., -2., -1., 0., 1., 2., 3., 4., 5.],
                shape![],
                vec![1.],
                shape![2, 3; 2],
                vec![-7., -6., -5., -4., -3., -2., -1., 0., 1., 2., 3., 4.],
            ),
        ];
        let dev = get_device();
        for tc in &test_cases {
            let x = dev.new_tensor_by_slice(tc.0, &tc.1);
            let k = dev.new_tensor_by_slice(tc.2, &tc.3);
            let mut y = dev.new_tensor(tc.4);
            y.alloc();
            dev.call_fw_impl("sub_scalar_r_fw_impl", &[&x, &k], &[], &[], &mut [&mut y]);
            assert_vector_ulps_eq!(&tc.5, &y.to_vec());
        }
    }
}
