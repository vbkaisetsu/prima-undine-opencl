define_opencl_fw_x_impl!(SoftplusFwImpl, softplus_fw_kernel);

#[cfg(test)]
mod tests {
    use crate::test_utils::get_device;
    use prima_undine::functions::BasicFunctions;
    use prima_undine::shape;

    #[test]
    fn check_softfplus_fw() {
        let y_f = |x: f64| (1. + x.exp()).ln();
        let x_data = vec![0., 0.5, 1., 2., 3., 4., 0., -0.5, -1., -2., -3., -4.];
        let y_data = generate_fw_testset!(x_data, y_f);
        let dev = get_device();
        let x = dev.new_tensor_by_slice(shape![2, 3; 2], &x_data);
        let mut y = dev.new_tensor(shape![2, 3; 2]);
        y.alloc();
        dev.call_fw_impl("softplus_fw_impl", &[&x], &[], &[], &mut [&mut y]);
        assert_vector_ulps_eq!(y_data, y.to_vec());
    }
}
