mod activation;
mod binary;
mod matmul;
mod reduction;
mod shape;
mod unary;
mod view;

use crate::backend::allocator::{Allocator, ArenaAllocator};
use crate::backend::backend::{
    ActivationOps, Backend, BinaryOps, MatMulOps, ReductionOps, ReverseScalarOps, ScalarOps,
    ShapeOps, TensorOps, UnaryOps, ViewOps,
};
use crate::linalg::tensor::{Scalar, Tensor};
use std::cell::RefCell;

#[derive(Clone, Default)]
pub struct CPUBackend;

impl TensorOps<Self> for CPUBackend {}

thread_local! {
    static ALLOCATOR: RefCell<ArenaAllocator> = RefCell::new(ArenaAllocator::default());
}

impl Backend for CPUBackend {
    fn name() -> &'static str {
        "CPU"
    }

    fn tensor<const NDIM: usize>(data: Vec<Scalar>, shape: [usize; NDIM]) -> Tensor<Self, NDIM> {
        let id = ALLOCATOR.with(|allocator| allocator.borrow_mut().allocate(data, shape.to_vec()));

        Tensor::<Self, NDIM> {
            id,
            _backend: std::marker::PhantomData,
        }
    }

    fn tensor_from_raw_parts<const NDIM: usize>(
        data: Vec<Scalar>,
        shape: [usize; NDIM],
        strides: [usize; NDIM],
    ) -> Tensor<Self, NDIM> {
        let id = ALLOCATOR.with(|allocator| {
            allocator
                .borrow_mut()
                .allocate_with_strides(data, shape.to_vec(), strides.to_vec())
        });

        Tensor::<Self, NDIM> {
            id,
            _backend: std::marker::PhantomData,
        }
    }

    fn shape<const NDIM: usize>(tensor: &Tensor<CPUBackend, NDIM>) -> [usize; NDIM] {
        ALLOCATOR.with(|allocator| {
            allocator
                .borrow()
                .shape(tensor.id)
                .expect("Tensor not found")
                .try_into()
                .expect("Shape dimension mismatch")
        })
    }

    fn data<const NDIM: usize>(tensor: &Tensor<CPUBackend, NDIM>) -> Vec<Scalar> {
        ALLOCATOR.with(|allocator| {
            allocator
                .borrow()
                .data(tensor.id)
                .expect("Tensor not found")
                .to_vec()
        })
    }

    fn with_mut_data<F, const NDIM: usize>(tensor: &Tensor<CPUBackend, NDIM>, f: F)
    where
        F: FnOnce(&mut [Scalar]),
    {
        ALLOCATOR.with(|allocator| {
            if let Some(data) = allocator.borrow_mut().data_mut(tensor.id) {
                f(data);
            } else {
                panic!("Tensor not found");
            }
        });
    }

    fn strides<const NDIM: usize>(tensor: &Tensor<CPUBackend, NDIM>) -> [usize; NDIM] {
        ALLOCATOR.with(|allocator| {
            allocator
                .borrow()
                .strides(tensor.id)
                .expect("Tensor not found")
                .try_into()
                .expect("Strides dimension mismatch")
        })
    }

    fn offset<const NDIM: usize>(tensor: &Tensor<CPUBackend, NDIM>) -> usize {
        ALLOCATOR.with(|allocator| {
            allocator
                .borrow()
                .offset(tensor.id)
                .expect("Tensor not found")
        })
    }

    fn internal_debug<const NDIM: usize>(tensor: &Tensor<CPUBackend, NDIM>) -> String {
        ALLOCATOR.with(|allocator| {
            let tensor_internal = allocator.borrow();
            let shape = tensor_internal.shape(tensor.id).expect("Tensor not found");
            let strides = tensor_internal
                .strides(tensor.id)
                .expect("Tensor not found");
            let offset = tensor_internal.offset(tensor.id).expect("Tensor not found");
            format!(
                "shape: {:?}, strides: {:?}, offset: {}",
                shape, strides, offset
            )
        })
    }
}
