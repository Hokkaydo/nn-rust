use crate::backend::autograd::{Autograd, GradOp};
use crate::backend::backend::{Backend, BinaryOps, ReverseScalarOps, ScalarOps};
use crate::linalg::tensor::{Scalar, Tensor};

impl<B: Backend> BinaryOps<Self> for Autograd<B> {
    fn add<const NDIM: usize>(
        tensor: &Tensor<Self, NDIM>,
        other: &Tensor<Self, NDIM>,
    ) -> Tensor<Self, NDIM> {
        let result = B::add(&tensor.into(), &other.into());
        Self::record_op(GradOp::Add {
            input_ids: vec![tensor.id, other.id],
            output_id: result.id,
        });
        result.into()
    }

    fn sub<const NDIM: usize>(
        tensor: &Tensor<Self, NDIM>,
        other: &Tensor<Self, NDIM>,
    ) -> Tensor<Self, NDIM> {
        let result = B::sub(&tensor.into(), &other.into());
        Self::record_op(GradOp::Sub {
            input_ids: vec![tensor.id, other.id],
            output_id: result.id,
        });
        result.into()
    }

    fn mul<const NDIM: usize>(
        tensor: &Tensor<Self, NDIM>,
        other: &Tensor<Self, NDIM>,
    ) -> Tensor<Self, NDIM> {
        let result = B::mul(&tensor.into(), &other.into());
        Self::record_op(GradOp::Mul {
            input_ids: vec![tensor.id, other.id],
            output_id: result.id,
        });
        result.into()
    }

    fn div<const NDIM: usize>(
        tensor: &Tensor<Self, NDIM>,
        other: &Tensor<Self, NDIM>,
    ) -> Tensor<Self, NDIM> {
        let result = B::div(&tensor.into(), &other.into());
        Self::record_op(GradOp::Div {
            input_ids: vec![tensor.id, other.id],
            output_id: result.id,
        });
        result.into()
    }
}

impl<B: Backend> ScalarOps<Self> for Autograd<B> {
    fn add_scalar<const NDIM: usize>(
        tensor: &Tensor<Self, NDIM>,
        scalar: Scalar,
    ) -> Tensor<Self, NDIM> {
        let result = B::add_scalar(&tensor.into(), scalar);
        Self::record_op(GradOp::AddScalar {
            input_id: tensor.id,
            scalar,
            output_id: result.id,
        });
        result.into()
    }

    fn sub_scalar<const NDIM: usize>(
        tensor: &Tensor<Self, NDIM>,
        scalar: Scalar,
    ) -> Tensor<Self, NDIM> {
        let result = B::sub_scalar(&tensor.into(), scalar);
        Self::record_op(GradOp::SubScalar {
            input_id: tensor.id,
            scalar,
            output_id: result.id,
        });
        result.into()
    }

    fn mul_scalar<const NDIM: usize>(
        tensor: &Tensor<Self, NDIM>,
        scalar: Scalar,
    ) -> Tensor<Self, NDIM> {
        let result = B::mul_scalar(&tensor.into(), scalar);
        Self::record_op(GradOp::MulScalar {
            input_id: tensor.id,
            scalar,
            output_id: result.id,
        });
        result.into()
    }

    fn div_scalar<const NDIM: usize>(
        tensor: &Tensor<Self, NDIM>,
        scalar: Scalar,
    ) -> Tensor<Self, NDIM> {
        let result = B::div_scalar(&tensor.into(), scalar);
        Self::record_op(GradOp::DivScalar {
            input_id: tensor.id,
            scalar,
            output_id: result.id,
        });
        result.into()
    }
}

impl<B: Backend> ReverseScalarOps<Self> for Autograd<B> {
    fn scalar_add<const NDIM: usize>(
        scalar: Scalar,
        tensor: &Tensor<Self, NDIM>,
    ) -> Tensor<Self, NDIM> {
        Self::add_scalar(tensor, scalar)
    }

    fn scalar_sub<const NDIM: usize>(
        scalar: Scalar,
        tensor: &Tensor<Self, NDIM>,
    ) -> Tensor<Self, NDIM> {
        let result = B::scalar_sub(scalar, &tensor.into());
        Self::record_op(GradOp::ScalarSub {
            scalar,
            input_id: tensor.id,
            output_id: result.id,
        });
        result.into()
    }

    fn scalar_mul<const NDIM: usize>(
        scalar: Scalar,
        tensor: &Tensor<Self, NDIM>,
    ) -> Tensor<Self, NDIM> {
        Self::mul_scalar(tensor, scalar)
    }

    fn scalar_div<const NDIM: usize>(
        scalar: Scalar,
        tensor: &Tensor<Self, NDIM>,
    ) -> Tensor<Self, NDIM> {
        let result = B::scalar_div(scalar, &tensor.into());
        Self::record_op(GradOp::ScalarDiv {
            scalar,
            input_id: tensor.id,
            output_id: result.id,
        });
        result.into()
    }
}
