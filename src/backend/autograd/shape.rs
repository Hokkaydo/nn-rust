use crate::backend::autograd::{Autograd, GradOp};
use crate::backend::backend::{Backend, ShapeOps};
use crate::linalg::tensor::Tensor;

impl<B: Backend> ShapeOps<Self> for Autograd<B> {
    fn reshape<const NDIM: usize>(
        tensor: &Tensor<Self, NDIM>,
        new_shape: [usize; NDIM],
    ) -> Tensor<Self, NDIM> {
        let result = B::reshape(&tensor.into(), new_shape);
        Self::record_op(GradOp::Reshape {
            input_id: tensor.id,
            output_id: result.id,
            new_shape: new_shape.to_vec(),
        });
        result.into()
    }

    fn transpose<const NDIM: usize>(
        tensor: &Tensor<Self, NDIM>,
        axes: Option<[usize; NDIM]>,
    ) -> Tensor<Self, NDIM> {
        let result = B::transpose(&tensor.into(), axes);
        Self::record_op(GradOp::Transpose {
            input_id: tensor.id,
            output_id: result.id,
            axes: axes.map(|a| a.to_vec()),
        });
        result.into()
    }

    fn squeeze<const NDIM: usize>(
        tensor: &Tensor<Self, NDIM>,
        axis: usize,
    ) -> Tensor<Self, { NDIM - 1 }> {
        let result = B::squeeze(&tensor.into(), axis);
        Self::record_op(GradOp::Squeeze {
            input_id: tensor.id,
            output_id: result.id,
            new_shape: B::shape(&result).to_vec(),
            axis,
        });
        result.into()
    }

    fn unsqueeze<const NDIM: usize>(
        tensor: &Tensor<Self, NDIM>,
        axis: usize,
    ) -> Tensor<Self, { NDIM + 1 }> {
        let result = B::unsqueeze(&tensor.into(), axis);
        Self::record_op(GradOp::Unsqueeze {
            input_id: tensor.id,
            output_id: result.id,
            axis,
        });
        result.into()
    }

    fn broadcast<const OLD_NDIM: usize, const NEW_NDIM: usize>(
        tensor: &Tensor<Self, OLD_NDIM>,
        new_shape: [usize; NEW_NDIM],
    ) -> Tensor<Self, NEW_NDIM> {
        let result = B::broadcast(&tensor.into(), new_shape);
        Self::record_op(GradOp::Broadcast {
            input_id: tensor.id,
            output_id: result.id,
            new_shape: new_shape.to_vec(),
        });
        result.into()
    }
}
