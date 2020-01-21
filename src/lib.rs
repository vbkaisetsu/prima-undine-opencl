#[cfg(test)]
#[macro_use]
mod test_utils;

mod clblast;
mod ops;

use std::ffi::{c_void, CString};
use std::sync::atomic::{AtomicPtr, Ordering};
use std::sync::{Arc, Mutex};

use ocl_core::types::abs::{CommandQueue, Context, Mem, Program};
use ocl_core::ContextProperties;

use prima_undine::{Device, DeviceImpl};

macro_rules! kernel_string {
    ( $kernel_name:ident ) => {
        String::from_utf8(include!(concat!(
            env!("OUT_DIR"),
            "/kernel_",
            stringify!($kernel_name),
            ".in"
        )))
        .unwrap();
    };
}

pub struct OpenCLInternal {
    context: Context,
    queue: CommandQueue,
}

impl OpenCLInternal {
    fn new(platform_id: usize, device_id: usize) -> OpenCLInternal {
        let platforms = ocl_core::get_platform_ids().unwrap();
        let platform = platforms[platform_id];
        let devices = ocl_core::get_device_ids(&platform, None, None).unwrap();
        let device = devices[device_id];
        let context_properties = ContextProperties::new().platform(platform);

        let context =
            ocl_core::create_context(Some(&context_properties), &[device], None, None).unwrap();
        let queue = ocl_core::create_command_queue(&context, &device, None).unwrap();

        OpenCLInternal {
            context: context,
            queue: queue,
        }
    }

    fn build_program(&self, src: &str) -> Program {
        let src_cstring = CString::new(src).unwrap();
        let program = ocl_core::create_program_with_source(&self.context, &[src_cstring]).unwrap();
        ocl_core::build_program(
            &program,
            None::<&[()]>,
            &CString::new("").unwrap(),
            None,
            None,
        )
        .unwrap();
        program
    }
}

pub struct OpenCL {
    internal: Arc<OpenCLInternal>,
}

impl OpenCL {
    pub fn new<'dev>(platform_id: usize, device_id: usize) -> Device<'dev> {
        let internal = Arc::new(OpenCLInternal::new(platform_id, device_id));
        let mut dev = Device::new(OpenCL {
            internal: Arc::clone(&internal),
        });

        // initializers

        dev.register_fw_impl(
            "reset_tensor_impl",
            ops::reset_tensor::ResetTensorImpl::new(&internal),
        );
        dev.register_fw_impl(
            "reset_tensor_by_slice_impl",
            ops::reset_tensor::ResetTensorBySliceImpl::new(&internal),
        );
        dev.register_fw_impl(
            "reset_tensor_by_tensor_impl",
            ops::reset_tensor::ResetTensorByTensorImpl::new(&internal),
        );

        dev.register_fw_f32_impl(
            "tensor_to_vector_impl",
            ops::tensor_to_vector::TensorToVectorImpl::new(&internal),
        );

        let identity_source = kernel_string!(identity);
        let identity_program = internal.build_program(&identity_source);
        dev.register_fw_impl(
            "identity_impl",
            ops::identity::IdentityImpl::new(&identity_program, &internal),
        );

        // TODO: random

        let xorshift_source = kernel_string!(xorshift);
        let xorshift_program = internal.build_program(&xorshift_source);
        let randomizer = Arc::new(Mutex::new(ops::random::XORShiftRandomizer::new(
            &xorshift_program,
            &internal,
        )));
        dev.register_fw_impl(
            "random_bernoulli_impl",
            ops::random::RandomBernoulliImpl::new(&randomizer, &xorshift_program, &internal),
        );
        dev.register_fw_impl(
            "random_normal_impl",
            ops::random::RandomNormalImpl::new(&randomizer, &xorshift_program, &internal),
        );
        dev.register_fw_impl(
            "random_uniform_impl",
            ops::random::RandomUniformImpl::new(&randomizer, &xorshift_program, &internal),
        );

        // assign

        let add_assign_source = kernel_string!(common) + &kernel_string!(add_assign);
        let add_assign_program = internal.build_program(&add_assign_source);
        dev.register_fw_impl(
            "add_assign_impl",
            ops::add_assign::AddAssignImpl::new(&add_assign_program, &internal),
        );

        let sub_assign_source = kernel_string!(common) + &kernel_string!(sub_assign);
        let sub_assign_program = internal.build_program(&sub_assign_source);
        dev.register_fw_impl(
            "sub_assign_impl",
            ops::sub_assign::SubAssignImpl::new(&sub_assign_program, &internal),
        );

        let mul_assign_source = kernel_string!(common) + &kernel_string!(mul_assign);
        let mul_assign_program = internal.build_program(&mul_assign_source);
        dev.register_fw_impl(
            "mul_assign_const_impl",
            ops::mul_assign::MulAssignConstImpl::new(&mul_assign_program, &internal),
        );

        // utility

        let argmax_source = kernel_string!(common) + &kernel_string!(argmax);
        let argmax_program = internal.build_program(&argmax_source);
        dev.register_fw_u32_impl(
            "argmax_impl",
            ops::argmax::ArgmaxImpl::new(&argmax_program, &internal),
        );

        let argmin_source = kernel_string!(common) + &kernel_string!(argmin);
        let argmin_program = internal.build_program(&argmin_source);
        dev.register_fw_u32_impl(
            "argmin_impl",
            ops::argmin::ArgminImpl::new(&argmin_program, &internal),
        );

        let argsort_source = kernel_string!(common) + &kernel_string!(argsort);
        let argsort_program = internal.build_program(&argsort_source);
        dev.register_fw_u32_impl(
            "argsort_impl",
            ops::argsort::ArgsortImpl::new(&argsort_program, &internal),
        );

        // arithmetic

        let neg_source = kernel_string!(common) + &kernel_string!(neg);
        let neg_program = internal.build_program(&neg_source);
        dev.register_fw_impl(
            "neg_fw_impl",
            ops::neg::NegFwImpl::new(&neg_program, &internal),
        );

        let add_source = kernel_string!(common) + &kernel_string!(add);
        let add_program = internal.build_program(&add_source);
        dev.register_fw_impl(
            "add_fw_impl",
            ops::add::AddFwImpl::new(&add_program, &internal),
        );
        dev.register_bw_impl(
            "add_bw_a_impl",
            ops::add::AddBwAImpl::new(&add_program, &internal),
        );
        dev.register_bw_impl(
            "add_bw_b_impl",
            ops::add::AddBwBImpl::new(&add_program, &internal),
        );
        dev.register_fw_impl(
            "add_const_fw_impl",
            ops::add::AddConstFwImpl::new(&add_program, &internal),
        );
        dev.register_bw_impl(
            "add_const_bw_impl",
            ops::add::AddConstBwImpl::new(&add_program, &internal),
        );
        dev.register_fw_impl(
            "add_scalar_fw_impl",
            ops::add::AddScalarFwImpl::new(&add_program, &internal),
        );

        let sub_source = kernel_string!(common) + &kernel_string!(sub);
        let sub_program = internal.build_program(&sub_source);
        dev.register_fw_impl(
            "sub_fw_impl",
            ops::sub::SubFwImpl::new(&sub_program, &internal),
        );
        dev.register_bw_impl(
            "sub_bw_a_impl",
            ops::sub::SubBwAImpl::new(&sub_program, &internal),
        );
        dev.register_bw_impl(
            "sub_bw_b_impl",
            ops::sub::SubBwBImpl::new(&sub_program, &internal),
        );
        dev.register_fw_impl(
            "sub_const_l_fw_impl",
            ops::sub::SubConstLFwImpl::new(&sub_program, &internal),
        );
        dev.register_bw_impl(
            "sub_const_l_bw_impl",
            ops::sub::SubConstLBwImpl::new(&sub_program, &internal),
        );
        dev.register_fw_impl(
            "sub_const_r_fw_impl",
            ops::sub::SubConstRFwImpl::new(&sub_program, &internal),
        );
        dev.register_bw_impl(
            "sub_const_r_bw_impl",
            ops::sub::SubConstRBwImpl::new(&sub_program, &internal),
        );
        dev.register_fw_impl(
            "sub_scalar_l_fw_impl",
            ops::sub::SubScalarLFwImpl::new(&sub_program, &internal),
        );
        dev.register_fw_impl(
            "sub_scalar_r_fw_impl",
            ops::sub::SubScalarRFwImpl::new(&sub_program, &internal),
        );

        let mul_source = kernel_string!(common) + &kernel_string!(mul);
        let mul_program = internal.build_program(&mul_source);
        dev.register_fw_impl(
            "mul_fw_impl",
            ops::mul::MulFwImpl::new(&mul_program, &internal),
        );
        dev.register_bw_impl(
            "mul_bw_a_impl",
            ops::mul::MulBwAImpl::new(&mul_program, &internal),
        );
        dev.register_bw_impl(
            "mul_bw_b_impl",
            ops::mul::MulBwBImpl::new(&mul_program, &internal),
        );
        dev.register_fw_impl(
            "mul_const_fw_impl",
            ops::mul::MulConstFwImpl::new(&mul_program, &internal),
        );
        dev.register_bw_impl(
            "mul_const_bw_impl",
            ops::mul::MulConstBwImpl::new(&mul_program, &internal),
        );
        dev.register_fw_impl(
            "mul_scalar_fw_impl",
            ops::mul::MulScalarFwImpl::new(&mul_program, &internal),
        );

        let div_source = kernel_string!(common) + &kernel_string!(div);
        let div_program = internal.build_program(&div_source);
        dev.register_fw_impl(
            "div_fw_impl",
            ops::div::DivFwImpl::new(&div_program, &internal),
        );
        dev.register_bw_impl(
            "div_bw_a_impl",
            ops::div::DivBwAImpl::new(&div_program, &internal),
        );
        dev.register_bw_impl(
            "div_bw_b_impl",
            ops::div::DivBwBImpl::new(&div_program, &internal),
        );
        dev.register_fw_impl(
            "div_const_l_fw_impl",
            ops::div::DivConstLFwImpl::new(&div_program, &internal),
        );
        dev.register_bw_impl(
            "div_const_l_bw_impl",
            ops::div::DivConstLBwImpl::new(&div_program, &internal),
        );
        dev.register_fw_impl(
            "div_const_r_fw_impl",
            ops::div::DivConstRFwImpl::new(&div_program, &internal),
        );
        dev.register_bw_impl(
            "div_const_r_bw_impl",
            ops::div::DivConstRBwImpl::new(&div_program, &internal),
        );
        dev.register_fw_impl(
            "div_scalar_l_fw_impl",
            ops::div::DivScalarLFwImpl::new(&div_program, &internal),
        );
        dev.register_fw_impl(
            "div_scalar_r_fw_impl",
            ops::div::DivScalarRFwImpl::new(&div_program, &internal),
        );

        // basic

        let powf_source = kernel_string!(common) + &kernel_string!(powf);
        let powf_program = internal.build_program(&powf_source);
        dev.register_fw_impl(
            "powf_fw_impl",
            ops::powf::PowfFwImpl::new(&powf_program, &internal),
        );
        dev.register_bw_impl(
            "powf_bw_a_impl",
            ops::powf::PowfBwAImpl::new(&powf_program, &internal),
        );
        dev.register_bw_impl(
            "powf_bw_b_impl",
            ops::powf::PowfBwBImpl::new(&powf_program, &internal),
        );
        dev.register_fw_impl(
            "powf_const_l_fw_impl",
            ops::powf::PowfConstLFwImpl::new(&powf_program, &internal),
        );
        dev.register_bw_impl(
            "powf_const_l_bw_impl",
            ops::powf::PowfConstLBwImpl::new(&powf_program, &internal),
        );
        dev.register_fw_impl(
            "powf_const_r_fw_impl",
            ops::powf::PowfConstRFwImpl::new(&powf_program, &internal),
        );
        dev.register_bw_impl(
            "powf_const_r_bw_impl",
            ops::powf::PowfConstRBwImpl::new(&powf_program, &internal),
        );
        dev.register_fw_impl(
            "powf_scalar_l_fw_impl",
            ops::powf::PowfScalarLFwImpl::new(&powf_program, &internal),
        );
        dev.register_fw_impl(
            "powf_scalar_r_fw_impl",
            ops::powf::PowfScalarRFwImpl::new(&powf_program, &internal),
        );

        let sqrt_source = kernel_string!(common) + &kernel_string!(sqrt);
        let sqrt_program = internal.build_program(&sqrt_source);
        dev.register_fw_impl(
            "sqrt_fw_impl",
            ops::sqrt::SqrtFwImpl::new(&sqrt_program, &internal),
        );
        dev.register_bw_impl(
            "sqrt_bw_impl",
            ops::sqrt::SqrtBwImpl::new(&sqrt_program, &internal),
        );

        let abs_source = kernel_string!(common) + &kernel_string!(abs);
        let abs_program = internal.build_program(&abs_source);
        dev.register_fw_impl(
            "abs_fw_impl",
            ops::abs::AbsFwImpl::new(&abs_program, &internal),
        );
        dev.register_bw_impl(
            "abs_bw_impl",
            ops::abs::AbsBwImpl::new(&abs_program, &internal),
        );

        let powi_source = kernel_string!(common) + &kernel_string!(powi);
        let powi_program = internal.build_program(&powi_source);
        dev.register_fw_impl(
            "powi_fw_impl",
            ops::powi::PowiFwImpl::new(&powi_program, &internal),
        );
        dev.register_bw_impl(
            "powi_bw_impl",
            ops::powi::PowiBwImpl::new(&powi_program, &internal),
        );

        // trigonometric

        let sin_source = kernel_string!(common) + &kernel_string!(sin);
        let sin_program = internal.build_program(&sin_source);
        dev.register_fw_impl(
            "sin_fw_impl",
            ops::sin::SinFwImpl::new(&sin_program, &internal),
        );
        dev.register_bw_impl(
            "sin_bw_impl",
            ops::sin::SinBwImpl::new(&sin_program, &internal),
        );

        let cos_source = kernel_string!(common) + &kernel_string!(cos);
        let cos_program = internal.build_program(&cos_source);
        dev.register_fw_impl(
            "cos_fw_impl",
            ops::cos::CosFwImpl::new(&cos_program, &internal),
        );
        dev.register_bw_impl(
            "cos_bw_impl",
            ops::cos::CosBwImpl::new(&cos_program, &internal),
        );

        let tan_source = kernel_string!(common) + &kernel_string!(tan);
        let tan_program = internal.build_program(&tan_source);
        dev.register_fw_impl(
            "tan_fw_impl",
            ops::tan::TanFwImpl::new(&tan_program, &internal),
        );
        dev.register_bw_impl(
            "tan_bw_impl",
            ops::tan::TanBwImpl::new(&tan_program, &internal),
        );

        // exp

        let exp_source = kernel_string!(common) + &kernel_string!(exp);
        let exp_program = internal.build_program(&exp_source);
        dev.register_fw_impl(
            "exp_fw_impl",
            ops::exp::ExpFwImpl::new(&exp_program, &internal),
        );
        dev.register_bw_impl(
            "exp_bw_impl",
            ops::exp::ExpBwImpl::new(&exp_program, &internal),
        );

        let ln_source = kernel_string!(common) + &kernel_string!(ln);
        let ln_program = internal.build_program(&ln_source);
        dev.register_fw_impl("ln_fw_impl", ops::ln::LnFwImpl::new(&ln_program, &internal));
        dev.register_bw_impl("ln_bw_impl", ops::ln::LnBwImpl::new(&ln_program, &internal));

        let tanh_source = kernel_string!(common) + &kernel_string!(tanh);
        let tanh_program = internal.build_program(&tanh_source);
        dev.register_fw_impl(
            "tanh_fw_impl",
            ops::tanh::TanhFwImpl::new(&tanh_program, &internal),
        );
        dev.register_bw_impl(
            "tanh_bw_impl",
            ops::tanh::TanhBwImpl::new(&tanh_program, &internal),
        );

        let sigmoid_source = kernel_string!(common) + &kernel_string!(sigmoid);
        let sigmoid_program = internal.build_program(&sigmoid_source);
        dev.register_fw_impl(
            "sigmoid_fw_impl",
            ops::sigmoid::SigmoidFwImpl::new(&sigmoid_program, &internal),
        );
        dev.register_bw_impl(
            "sigmoid_bw_impl",
            ops::sigmoid::SigmoidBwImpl::new(&sigmoid_program, &internal),
        );

        let softplus_source = kernel_string!(common) + &kernel_string!(softplus);
        let softplus_program = internal.build_program(&softplus_source);
        dev.register_fw_impl(
            "softplus_fw_impl",
            ops::softplus::SoftplusFwImpl::new(&softplus_program, &internal),
        );

        // reduction

        let sum_source = kernel_string!(common) + &kernel_string!(sum);
        let sum_program = internal.build_program(&sum_source);
        dev.register_fw_impl(
            "sum_fw_impl",
            ops::sum::SumFwImpl::new(&sum_program, &internal),
        );

        let logsumexp_source = kernel_string!(common) + &kernel_string!(logsumexp);
        let logsumexp_program = internal.build_program(&logsumexp_source);
        dev.register_fw_impl(
            "logsumexp_fw_impl",
            ops::logsumexp::LogsumexpFwImpl::new(&logsumexp_program, &internal),
        );

        let max_source = kernel_string!(common) + &kernel_string!(max);
        let max_program = internal.build_program(&max_source);
        dev.register_fw_impl(
            "max_fw_impl",
            ops::max::MaxFwImpl::new(&max_program, &internal),
        );
        dev.register_bw_impl(
            "max_bw_impl",
            ops::max::MaxBwImpl::new(&max_program, &internal),
        );

        let min_source = kernel_string!(common) + &kernel_string!(min);
        let min_program = internal.build_program(&min_source);
        dev.register_fw_impl(
            "min_fw_impl",
            ops::min::MinFwImpl::new(&min_program, &internal),
        );
        dev.register_bw_impl(
            "min_bw_impl",
            ops::min::MinBwImpl::new(&min_program, &internal),
        );

        let broadcast_source = kernel_string!(common) + &kernel_string!(broadcast);
        let broadcast_program = internal.build_program(&broadcast_source);
        dev.register_fw_impl(
            "broadcast_fw_impl",
            ops::broadcast::BroadcastFwImpl::new(&broadcast_program, &internal),
        );

        // matrix

        dev.register_fw_impl("matmul_fw_impl", ops::matmul::MatmulFwImpl::new(&internal));
        dev.register_bw_impl(
            "matmul_bw_a_impl",
            ops::matmul::MatmulBwAImpl::new(&internal),
        );
        dev.register_bw_impl(
            "matmul_bw_b_impl",
            ops::matmul::MatmulBwBImpl::new(&internal),
        );

        let transpose_source = kernel_string!(common) + &kernel_string!(transpose);
        let transpose_program = internal.build_program(&transpose_source);
        dev.register_fw_impl(
            "transpose_fw_impl",
            ops::transpose::TransposeFwImpl::new(&transpose_program, &internal),
        );
        dev.register_bw_impl(
            "transpose_bw_impl",
            ops::transpose::TransposeBwImpl::new(&transpose_program, &internal),
        );

        let permute_dims_source = kernel_string!(common) + &kernel_string!(permute_dims);
        let permute_dims_program = internal.build_program(&permute_dims_source);
        dev.register_fw_impl(
            "permute_dims_fw_impl",
            ops::permute_dims::PermuteDimsFwImpl::new(&permute_dims_program, &internal),
        );
        dev.register_bw_impl(
            "permute_dims_bw_impl",
            ops::permute_dims::PermuteDimsBwImpl::new(&permute_dims_program, &internal),
        );

        let flip_source = kernel_string!(common) + &kernel_string!(flip);
        let flip_program = internal.build_program(&flip_source);
        dev.register_fw_impl(
            "flip_fw_impl",
            ops::flip::FlipFwImpl::new(&flip_program, &internal),
        );
        dev.register_bw_impl(
            "flip_bw_impl",
            ops::flip::FlipBwImpl::new(&flip_program, &internal),
        );

        let triangular_l_source = kernel_string!(common) + &kernel_string!(triangular_l);
        let triangular_l_program = internal.build_program(&triangular_l_source);
        dev.register_fw_impl(
            "triangular_l_fw_impl",
            ops::triangular_l::TriangularLFwImpl::new(&triangular_l_program, &internal),
        );
        dev.register_bw_impl(
            "triangular_l_bw_impl",
            ops::triangular_l::TriangularLBwImpl::new(&triangular_l_program, &internal),
        );

        let triangular_u_source = kernel_string!(common) + &kernel_string!(triangular_u);
        let triangular_u_program = internal.build_program(&triangular_u_source);
        dev.register_fw_impl(
            "triangular_u_fw_impl",
            ops::triangular_u::TriangularUFwImpl::new(&triangular_u_program, &internal),
        );
        dev.register_bw_impl(
            "triangular_u_bw_impl",
            ops::triangular_u::TriangularUBwImpl::new(&triangular_u_program, &internal),
        );

        // ramp

        let prelu_source = kernel_string!(common) + &kernel_string!(prelu);
        let prelu_program = internal.build_program(&prelu_source);
        dev.register_fw_impl(
            "prelu_fw_impl",
            ops::prelu::PReLUFwImpl::new(&prelu_program, &internal),
        );
        dev.register_bw_impl(
            "prelu_bw_impl",
            ops::prelu::PReLUBwImpl::new(&prelu_program, &internal),
        );

        let elu_source = kernel_string!(common) + &kernel_string!(elu);
        let elu_program = internal.build_program(&elu_source);
        dev.register_fw_impl(
            "elu_fw_impl",
            ops::elu::EluFwImpl::new(&elu_program, &internal),
        );
        dev.register_bw_impl(
            "elu_bw_impl",
            ops::elu::EluBwImpl::new(&elu_program, &internal),
        );

        // manipulation

        let slice_source = kernel_string!(common) + &kernel_string!(slice);
        let slice_program = internal.build_program(&slice_source);
        dev.register_fw_impl(
            "slice_fw_impl",
            ops::slice::SliceFwImpl::new(&slice_program, &internal),
        );
        dev.register_bw_impl(
            "slice_bw_impl",
            ops::slice::SliceBwImpl::new(&slice_program, &internal),
        );

        let pick_source = kernel_string!(common) + &kernel_string!(pick);
        let pick_program = internal.build_program(&pick_source);
        dev.register_fw_impl(
            "pick_fw_impl",
            ops::pick::PickFwImpl::new(&pick_program, &internal),
        );
        dev.register_bw_impl(
            "pick_bw_impl",
            ops::pick::PickBwImpl::new(&pick_program, &internal),
        );

        let concat_source = kernel_string!(common) + &kernel_string!(concat);
        let concat_program = internal.build_program(&concat_source);
        dev.register_fw_impl(
            "concat_fw_impl",
            ops::concat::ConcatFwImpl::new(&concat_program, &internal),
        );

        // batch

        let batch_concat_source = kernel_string!(common) + &kernel_string!(batch_concat);
        let batch_concat_program = internal.build_program(&batch_concat_source);
        dev.register_fw_impl(
            "batch_concat_fw_impl",
            ops::batch_concat::BatchConcatFwImpl::new(&batch_concat_program, &internal),
        );

        let batch_pick_source = kernel_string!(common) + &kernel_string!(batch_pick);
        let batch_pick_program = internal.build_program(&batch_pick_source);
        dev.register_fw_impl(
            "batch_pick_fw_impl",
            ops::batch_pick::BatchPickFwImpl::new(&batch_pick_program, &internal),
        );
        dev.register_bw_impl(
            "batch_pick_bw_impl",
            ops::batch_pick::BatchPickBwImpl::new(&batch_pick_program, &internal),
        );

        let batch_slice_source = kernel_string!(common) + &kernel_string!(batch_slice);
        let batch_slice_program = internal.build_program(&batch_slice_source);
        dev.register_fw_impl(
            "batch_slice_fw_impl",
            ops::batch_slice::BatchSliceFwImpl::new(&batch_slice_program, &internal),
        );
        dev.register_bw_impl(
            "batch_slice_bw_impl",
            ops::batch_slice::BatchSliceBwImpl::new(&batch_slice_program, &internal),
        );

        let batch_sum_source = kernel_string!(common) + &kernel_string!(batch_sum);
        let batch_sum_program = internal.build_program(&batch_sum_source);
        dev.register_fw_impl(
            "batch_sum_fw_impl",
            ops::batch_sum::BatchSumFwImpl::new(&batch_sum_program, &internal),
        );

        dev
    }
}

impl DeviceImpl for OpenCL {
    fn identifier(&self) -> String {
        "OpenCL,".to_string() + &(self.internal.context.as_ptr() as usize).to_string()
    }

    fn new_handle(&self, size: u32) -> AtomicPtr<c_void> {
        let buffer = unsafe {
            ocl_core::create_buffer(
                &self.internal.context,
                ocl_core::MEM_READ_WRITE,
                size as usize,
                None::<&[f32]>,
            )
            .unwrap()
        };
        AtomicPtr::new(Box::into_raw(Box::new(buffer)) as *mut c_void)
    }

    fn drop_handle(&self, handle: &AtomicPtr<c_void>) {
        unsafe {
            Box::from_raw(handle.load(Ordering::Acquire) as *mut Mem);
        }
    }
}
