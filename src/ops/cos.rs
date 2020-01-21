define_opencl_fw_x_impl!(CosFwImpl, cos_fw_kernel);
define_opencl_bw_x_impl!(CosBwImpl, cos_bw_kernel);

#[cfg(test)]
mod tests {
    use crate::test_utils::get_device;
    use prima_undine::functions::BasicFunctions;
    use prima_undine::shape;

    #[test]
    fn check_cos_fw() {
        let y_f = |x: f64| x.cos();
        let x_data = vec![0., 0.5, 1., 2., 3., 4., 0., -0.5, -1., -2., -3., -4.];
        let y_data = generate_fw_testset!(x_data, y_f);
        let dev = get_device();
        let x = dev.new_tensor_by_slice(shape![2, 3; 2], &x_data);
        let mut y = dev.new_tensor(shape![2, 3; 2]);
        y.alloc();
        dev.call_fw_impl("cos_fw_impl", &[&x], &[], &[], &mut [&mut y]);
        assert_vector_ulps_eq!(y_data, y.to_vec());
    }

    #[test]
    fn check_cos_bw() {
        let y_f = |x: f64| x.cos();
        let gx_f = |x: f64, _y: f64, gy: f64| 1. - gy * x.sin();
        let x_data = vec![0., 1., 2., 3., 0., -1., -2., -3.];
        let gy_data = vec![1., -1., 2., -2., 2., -2., 1., -1.];
        let (y_data, gx_data) = generate_bw_testset!(x_data, gy_data, y_f, gx_f);
        let dev = get_device();
        let x = dev.new_tensor_by_slice(shape![2, 2; 2], &x_data);
        let y = dev.new_tensor_by_slice(shape![2, 2; 2], &y_data);
        let gy = dev.new_tensor_by_slice(shape![2, 2; 2], &gy_data);
        let mut gx = dev.new_tensor_by_constant(shape![2, 2; 2], 1.);
        dev.call_bw_impl("cos_bw_impl", &[&x], &[&y], &[&gy], &[], &[], &mut gx);
        assert_vector_ulps_eq!(gx_data, gx.to_vec());
    }
}
