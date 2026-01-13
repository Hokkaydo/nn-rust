mod activation;
mod binary;
mod engine;
mod matmul;
mod reduction;
mod shape;
mod unary;
mod view;

use crate::backend::backend::{
    ActivationOps, Backend, BinaryOps, MatMulOps, ReductionOps, ReverseScalarOps, ScalarOps,
    ShapeOps, TensorOps, UnaryOps, ViewOps,
};
use crate::linalg::tensor::{Scalar, Tensor, TensorId};
use std::cell::RefCell;
use std::marker::PhantomData;

#[derive(Clone)]
pub struct Autograd<B: Backend> {
    _backend: PhantomData<B>,
}

enum GradOp {
    // Unary operations
    Neg {
        input_id: TensorId,
        output_id: TensorId,
    },
    Pow {
        input_id: TensorId,
        exponent: Scalar,
        output_id: TensorId,
    },
    Exp {
        input_id: TensorId,
        output_id: TensorId,
    },
    Log {
        input_id: TensorId,
        output_id: TensorId,
    },
    Abs {
        input_id: TensorId,
        output_id: TensorId,
    },
    Sign {
        input_id: TensorId,
        output_id: TensorId,
    },
    Clamp {
        input_id: TensorId,
        min: Scalar,
        max: Scalar,
        output_id: TensorId,
    },

    // Binary operations
    Add {
        input_ids: Vec<TensorId>,
        output_id: TensorId,
    },
    AddScalar {
        input_id: TensorId,
        scalar: Scalar,
        output_id: TensorId,
    },
    Sub {
        input_ids: Vec<TensorId>,
        output_id: TensorId,
    },
    SubScalar {
        input_id: TensorId,
        scalar: Scalar,
        output_id: TensorId,
    },
    ScalarSub {
        scalar: Scalar,
        input_id: TensorId,
        output_id: TensorId,
    },
    Mul {
        input_ids: Vec<TensorId>,
        output_id: TensorId,
    },
    MulScalar {
        input_id: TensorId,
        scalar: Scalar,
        output_id: TensorId,
    },
    Div {
        input_ids: Vec<TensorId>,
        output_id: TensorId,
    },
    DivScalar {
        input_id: TensorId,
        scalar: Scalar,
        output_id: TensorId,
    },
    ScalarDiv {
        scalar: Scalar,
        input_id: TensorId,
        output_id: TensorId,
    },
    MatMul {
        input_ids: Vec<TensorId>,
        output_id: TensorId,
    },

    // Reduction operations
    Sum {
        input_id: TensorId,
        output_id: TensorId,
        axes: Option<Vec<usize>>,
    },
    Mean {
        input_id: TensorId,
        output_id: TensorId,
        axes: Option<Vec<usize>>,
    },
    Max {
        input_id: TensorId,
        output_id: TensorId,
        axes: Option<Vec<usize>>,
    },
    Min {
        input_id: TensorId,
        output_id: TensorId,
        axes: Option<Vec<usize>>,
    },

    // Shape operations
    Reshape {
        input_id: TensorId,
        output_id: TensorId,
        new_shape: Vec<usize>,
    },
    Transpose {
        input_id: TensorId,
        output_id: TensorId,
        axes: Option<Vec<usize>>,
    },
    Squeeze {
        input_id: TensorId,
        output_id: TensorId,
        new_shape: Vec<usize>,
        axis: usize,
    },
    Unsqueeze {
        input_id: TensorId,
        output_id: TensorId,
        axis: usize,
    },
    Broadcast {
        input_id: TensorId,
        output_id: TensorId,
        new_shape: Vec<usize>,
    },

    // View operations
    Slice {
        input_id: TensorId,
        output_id: TensorId,
        axis: usize,
        start: usize,
        len: usize,
    },
    Gather {
        input_id: TensorId,
        output_id: TensorId,
        axis: usize,
        indices: Vec<usize>,
    },

    // Activation functions
    Sigmoid {
        input_id: TensorId,
        output_id: TensorId,
    },
    ReLU {
        input_id: TensorId,
        output_id: TensorId,
    },
    Softmax {
        input_id: TensorId,
        output_id: TensorId,
    },
    LogSoftmax {
        input_id: TensorId,
        output_id: TensorId,
    },
}

thread_local! {
    static AUTOGRAD_TAPE: RefCell<Vec<GradOp>> = RefCell::new(Vec::new());
}

impl<B: Backend> Autograd<B> {
    fn record_op(op: GradOp) {
        AUTOGRAD_TAPE.with(|tape| {
            tape.borrow_mut().push(op);
        });
    }
}

impl<B: Backend> Default for Autograd<B> {
    fn default() -> Self {
        Autograd {
            _backend: PhantomData,
        }
    }
}

impl<B: Backend, const NDIM: usize> Into<Tensor<Autograd<B>, NDIM>> for &Tensor<B, NDIM> {
    fn into(self) -> Tensor<Autograd<B>, NDIM> {
        Tensor::<Autograd<B>, NDIM> {
            id: self.id,
            _backend: PhantomData,
        }
    }
}

impl<B: Backend, const NDIM: usize> Into<Tensor<Autograd<B>, NDIM>> for Tensor<B, NDIM> {
    fn into(self) -> Tensor<Autograd<B>, NDIM> {
        Tensor::<Autograd<B>, NDIM> {
            id: self.id,
            _backend: PhantomData,
        }
    }
}

impl<B: Backend, const NDIM: usize> Into<Tensor<B, NDIM>> for &Tensor<Autograd<B>, NDIM> {
    fn into(self) -> Tensor<B, NDIM> {
        Tensor::<B, NDIM> {
            id: self.id,
            _backend: PhantomData,
        }
    }
}

impl<B: Backend> TensorOps<Self> for Autograd<B> {}

impl<B: Backend> Backend for Autograd<B> {
    fn name() -> &'static str {
        "Autograd"
    }

    fn tensor<const NDIM: usize>(data: Vec<Scalar>, shape: [usize; NDIM]) -> Tensor<Self, NDIM> {
        let registered = B::tensor(data, shape);
        Tensor::<Self, NDIM> {
            id: registered.id,
            _backend: PhantomData,
        }
    }

    fn tensor_from_raw_parts<const NDIM: usize>(
        data: Vec<Scalar>,
        shape: [usize; NDIM],
        strides: [usize; NDIM],
    ) -> Tensor<Self, NDIM> {
        let registered = B::tensor_from_raw_parts(data, shape, strides);
        Tensor::<Self, NDIM> {
            id: registered.id,
            _backend: PhantomData,
        }
    }

    fn shape<const NDIM: usize>(tensor: &Tensor<Self, NDIM>) -> [usize; NDIM] {
        B::shape(&tensor.into())
    }

    fn data<const NDIM: usize>(tensor: &Tensor<Self, NDIM>) -> Vec<Scalar> {
        B::data(&tensor.into())
    }

    fn with_mut_data<F, const NDIM: usize>(tensor: &Tensor<Self, NDIM>, f: F)
    where
        F: FnOnce(&mut [Scalar]),
    {
        B::with_mut_data(&tensor.into(), f)
    }

    fn strides<const NDIM: usize>(tensor: &Tensor<Self, NDIM>) -> [usize; NDIM] {
        B::strides(&tensor.into())
    }

    fn offset<const NDIM: usize>(tensor: &Tensor<Self, NDIM>) -> usize {
        B::offset(&tensor.into())
    }

    fn internal_debug<const NDIM: usize>(tensor: &Tensor<Self, NDIM>) -> String {
        B::internal_debug(&tensor.into())
    }
}
