use crate::backend::autograd::{Autograd, GradOp};
use crate::backend::backend::{Backend, MatMulOps};
use crate::linalg::tensor::Tensor;

impl<B: Backend> MatMulOps<Self> for Autograd<B> {
    fn matmul_11(a: &Tensor<Self, 1>, b: &Tensor<Self, 1>) -> Tensor<Self, 1> {
        let result = B::matmul_11(&a.into(), &b.into());
        Self::record_op(GradOp::MatMul {
            input_ids: vec![a.id, b.id],
            output_id: result.id,
        });
        result.into()
    }

    fn matmul_12(a: &Tensor<Self, 1>, b: &Tensor<Self, 2>) -> Tensor<Self, 1> {
        let result = B::matmul_12(&a.into(), &b.into());
        Self::record_op(GradOp::MatMul {
            input_ids: vec![a.id, b.id],
            output_id: result.id,
        });
        result.into()
    }

    fn matmul_21(a: &Tensor<Self, 2>, b: &Tensor<Self, 1>) -> Tensor<Self, 1> {
        let result = B::matmul_21(&a.into(), &b.into());
        Self::record_op(GradOp::MatMul {
            input_ids: vec![a.id, b.id],
            output_id: result.id,
        });
        result.into()
    }

    fn matmul_22(a: &Tensor<Self, 2>, b: &Tensor<Self, 2>) -> Tensor<Self, 2> {
        let result = B::matmul_22(&a.into(), &b.into());
        Self::record_op(GradOp::MatMul {
            input_ids: vec![a.id, b.id],
            output_id: result.id,
        });
        result.into()
    }
}
