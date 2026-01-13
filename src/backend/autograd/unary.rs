use crate::backend::autograd::{Autograd, GradOp};
use crate::backend::backend::{Backend, UnaryOps};
use crate::linalg::tensor::{Scalar, Tensor};

impl<B: Backend> UnaryOps<Self> for Autograd<B> {
    fn neg<const NDIM: usize>(tensor: &Tensor<Self, NDIM>) -> Tensor<Self, NDIM> {
        let result = B::neg(&tensor.into());
        Self::record_op(GradOp::Neg {
            input_id: tensor.id,
            output_id: result.id,
        });
        result.into()
    }

    fn sqrt<const NDIM: usize>(tensor: &Tensor<Self, NDIM>) -> Tensor<Self, NDIM> {
        let result = B::sqrt(&tensor.into());
        Self::record_op(GradOp::Pow {
            input_id: tensor.id,
            exponent: 0.5,
            output_id: result.id,
        });
        result.into()
    }

    fn exp<const NDIM: usize>(tensor: &Tensor<Self, NDIM>) -> Tensor<Self, NDIM> {
        let result = B::exp(&tensor.into());
        Self::record_op(GradOp::Exp {
            input_id: tensor.id,
            output_id: result.id,
        });
        result.into()
    }

    fn log<const NDIM: usize>(tensor: &Tensor<Self, NDIM>) -> Tensor<Self, NDIM> {
        let result = B::log(&tensor.into());
        Self::record_op(GradOp::Log {
            input_id: tensor.id,
            output_id: result.id,
        });
        result.into()
    }

    fn abs<const NDIM: usize>(tensor: &Tensor<Self, NDIM>) -> Tensor<Self, NDIM> {
        let result = B::abs(&tensor.into());
        Self::record_op(GradOp::Abs {
            input_id: tensor.id,
            output_id: result.id,
        });
        result.into()
    }

    fn sign<const NDIM: usize>(tensor: &Tensor<Self, NDIM>) -> Tensor<Self, NDIM> {
        let result = B::sign(&tensor.into());
        Self::record_op(GradOp::Sign {
            input_id: tensor.id,
            output_id: result.id,
        });
        result.into()
    }

    fn clamp<const NDIM: usize>(
        tensor: &Tensor<Self, NDIM>,
        min: Scalar,
        max: Scalar,
    ) -> Tensor<Self, NDIM> {
        let result = B::clamp(&tensor.into(), min, max);
        Self::record_op(GradOp::Clamp {
            input_id: tensor.id,
            min,
            max,
            output_id: result.id,
        });
        result.into()
    }

    fn pow<const NDIM: usize>(tensor: &Tensor<Self, NDIM>, exponent: Scalar) -> Tensor<Self, NDIM> {
        let result = B::pow(&tensor.into(), exponent);
        Self::record_op(GradOp::Pow {
            input_id: tensor.id,
            exponent,
            output_id: result.id,
        });
        result.into()
    }
}
