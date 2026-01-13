use crate::backend::autograd::{Autograd, GradOp};
use crate::backend::backend::{Backend, ViewOps};
use crate::linalg::tensor::Tensor;

impl<B: Backend> ViewOps<Self> for Autograd<B> {
    fn slice<const NDIM: usize>(
        tensor: &Tensor<Self, NDIM>,
        axis: usize,
        start: usize,
        len: usize,
    ) -> Tensor<Self, NDIM> {
        let result = B::slice(&tensor.into(), axis, start, len);
        Self::record_op(GradOp::Slice {
            input_id: tensor.id,
            output_id: result.id,
            axis,
            start,
            len,
        });
        result.into()
    }

    fn gather<const NDIM: usize>(
        tensor: &Tensor<Self, NDIM>,
        axis: usize,
        indices: &[usize],
    ) -> Tensor<Self, NDIM> {
        let result = B::gather(&tensor.into(), axis, indices);
        Self::record_op(GradOp::Gather {
            input_id: tensor.id,
            output_id: result.id,
            axis,
            indices: indices.to_vec(),
        });
        result.into()
    }
}
