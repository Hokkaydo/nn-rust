use crate::backend::autograd::{Autograd, GradOp};
use crate::backend::backend::{Backend, ReductionOps};
use crate::linalg::tensor::Tensor;

impl<B: Backend> ReductionOps<Self> for Autograd<B> {
    fn sum<const NDIM: usize>(
        tensor: &Tensor<Self, NDIM>,
        axes: Option<&[usize]>,
    ) -> Tensor<Self, NDIM> {
        let result = B::sum(&tensor.into(), axes);
        Self::record_op(GradOp::Sum {
            input_id: tensor.id,
            output_id: result.id,
            axes: axes.map(|a| a.to_vec()),
        });
        result.into()
    }

    fn mean<const NDIM: usize>(
        tensor: &Tensor<Self, NDIM>,
        axes: Option<&[usize]>,
    ) -> Tensor<Self, NDIM> {
        let result = B::mean(&tensor.into(), axes);
        Self::record_op(GradOp::Mean {
            input_id: tensor.id,
            output_id: result.id,
            axes: axes.map(|a| a.to_vec()),
        });
        result.into()
    }

    fn max<const NDIM: usize>(
        tensor: &Tensor<Self, NDIM>,
        axes: Option<&[usize]>,
    ) -> Tensor<Self, NDIM> {
        let result = B::max(&tensor.into(), axes);
        Self::record_op(GradOp::Max {
            input_id: tensor.id,
            output_id: result.id,
            axes: axes.map(|a| a.to_vec()),
        });
        result.into()
    }

    fn min<const NDIM: usize>(
        tensor: &Tensor<Self, NDIM>,
        axes: Option<&[usize]>,
    ) -> Tensor<Self, NDIM> {
        let result = B::min(&tensor.into(), axes);
        Self::record_op(GradOp::Min {
            input_id: tensor.id,
            output_id: result.id,
            axes: axes.map(|a| a.to_vec()),
        });
        result.into()
    }

    fn argmax<const NDIM: usize>(
        tensor: &Tensor<Self, NDIM>,
        axis: usize,
    ) -> Tensor<Self, { NDIM - 1 }> {
        B::argmax(&tensor.into(), axis).into()
    }
}
