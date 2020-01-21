use ocl_core::Event;
use ocl_core::MapFlags;
use ocl_core::MemMap;

use prima_undine::device_impl::FunctionFwImpl;
use prima_undine::functions::BasicFunctions;
use prima_undine::Tensor;

define_empty_impl!(ResetTensorImpl);
impl FunctionFwImpl for ResetTensorImpl {
    fn call(&self, _xs: &[&Tensor], _u32data: &[u32], f32data: &[f32], ys: &mut [&mut Tensor]) {
        let k = f32data[0];
        let y = &mut ys[0];
        unsafe {
            ocl_core::enqueue_fill_buffer(
                &self.internal.queue,
                buffer!(y),
                k,
                0,
                y.shape().size() as usize,
                None::<Event>,
                None::<&mut Event>,
                None,
            )
            .unwrap();
        }
    }
}

define_empty_impl!(ResetTensorBySliceImpl);
impl FunctionFwImpl for ResetTensorBySliceImpl {
    fn call(&self, _xs: &[&Tensor], _u32data: &[u32], f32data: &[f32], ys: &mut [&mut Tensor]) {
        let y = &mut ys[0];
        unsafe {
            super::common::write_buffer(&self.internal.queue, f32data, buffer!(y));
        }
    }
}

define_empty_impl!(ResetTensorByTensorImpl);
impl FunctionFwImpl for ResetTensorByTensorImpl {
    fn call(&self, xs: &[&Tensor], _u32data: &[u32], _f32data: &[f32], ys: &mut [&mut Tensor]) {
        let x = xs[0];
        let y = &mut ys[0];
        let size = x.shape().size() as usize;
        let x_devid = x.device().identifier();
        unsafe {
            if x_devid == y.device().identifier() {
                ocl_core::enqueue_copy_buffer::<f32, &ocl_core::Mem, &mut Event, Event>(
                    &self.internal.queue,
                    buffer!(x),
                    buffer!(y),
                    0,
                    0,
                    size,
                    None::<Event>,
                    None::<&mut Event>,
                )
                .unwrap();
            } else {
                if x_devid.starts_with("OpenCL,") {
                    let mem: MemMap<f32> = ocl_core::enqueue_map_buffer(
                        &self.internal.queue,
                        buffer!(x),
                        true,
                        MapFlags::READ,
                        0,
                        size,
                        None::<Event>,
                        None::<&mut Event>,
                    )
                    .unwrap();
                    super::common::write_buffer(
                        &self.internal.queue,
                        mem.as_slice(size),
                        buffer!(y),
                    );
                } else {
                    super::common::write_buffer(&self.internal.queue, &x.to_vec(), buffer!(y));
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::test_utils::get_device;
    use prima_undine::devices as D;
    use prima_undine::functions::BasicFunctions;
    use prima_undine::shape;

    #[test]
    fn check_reset_tensor_by_tensor() {
        let x_data = vec![
            -8., -7., -6., -5., -4., -3., -2., -1., 0., 1., 2., 3., 4., 5., 6., 7.,
        ];
        let ocl_dev = get_device();
        let naive_dev = D::Naive::new();
        let x1 = ocl_dev.new_tensor_by_slice(shape![4, 2; 2], &x_data);
        let x2 = naive_dev.new_tensor_by_slice(shape![4, 2; 2], &x_data);
        let mut y1 = ocl_dev.new_tensor(shape![4, 2; 2]);
        let mut y2 = ocl_dev.new_tensor(shape![4, 2; 2]);
        y1.alloc();
        y2.alloc();
        ocl_dev.call_fw_impl(
            "reset_tensor_by_tensor_impl",
            &[&x1],
            &[],
            &[],
            &mut [&mut y1],
        );
        ocl_dev.call_fw_impl(
            "reset_tensor_by_tensor_impl",
            &[&x2],
            &[],
            &[],
            &mut [&mut y2],
        );
        assert_vector_ulps_eq!(x_data, y1.to_vec());
        assert_vector_ulps_eq!(x_data, y2.to_vec());
    }
}
