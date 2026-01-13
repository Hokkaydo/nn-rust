use crate::backend::autograd::{Autograd, GradOp};
use crate::backend::backend::{ActivationOps, Backend};
use crate::linalg::tensor::Tensor;

impl<B: Backend> ActivationOps<Self> for Autograd<B> {
    fn sigmoid<const NDIM: usize>(tensor: &Tensor<Self, NDIM>) -> Tensor<Self, NDIM> {
        let result = B::sigmoid(&tensor.into());
        Self::record_op(GradOp::Sigmoid {
            input_id: tensor.id,
            output_id: result.id,
        });
        result.into()
    }

    fn softmax<const NDIM: usize>(tensor: &Tensor<Self, NDIM>) -> Tensor<Self, NDIM> {
        let result = B::softmax(&tensor.into());
        Self::record_op(GradOp::Softmax {
            input_id: tensor.id,
            output_id: result.id,
        });
        result.into()
    }

    fn log_softmax<const NDIM: usize>(tensor: &Tensor<Self, NDIM>) -> Tensor<Self, NDIM> {
        let result = B::log_softmax(&tensor.into());
        Self::record_op(GradOp::LogSoftmax {
            input_id: tensor.id,
            output_id: result.id,
        });
        result.into()
    }

    fn relu<const NDIM: usize>(tensor: &Tensor<Self, NDIM>) -> Tensor<Self, NDIM> {
        let result = B::relu(&tensor.into());
        Self::record_op(GradOp::ReLU {
            input_id: tensor.id,
            output_id: result.id,
        });
        result.into()
    }
}
