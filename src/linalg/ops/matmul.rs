use crate::linalg::autograd::grad_fn::TensorMatMulTensorFn;
use crate::linalg::tensor_grad::{InternalTensor, Storage, Tensor};
use std::cell::RefCell;
use std::rc::Rc;

impl Tensor {
    pub fn matmul(&self, other: &Tensor) -> Tensor {
        assert_eq!(
            self.shape.len(),
            2,
            "Left tensor must be 2D, got shape {:?}",
            self.shape
        );
        assert_eq!(
            other.shape.len(),
            2,
            "Right tensor must be 2D, got shape {:?}",
            other.shape
        );
        assert_eq!(
            self.shape[1], other.shape[0],
            "Inner dimensions must match: got {} vs {}",
            self.shape[1], other.shape[0]
        );

        let m = self.shape[0];
        let k = self.shape[1];
        let n = other.shape[1];

        let mut result_data = vec![0.0; m * n];

        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0;
                for kk in 0..k {
                    let a_idx = self.offset + i * self.strides[0] + kk * self.strides[1];
                    let b_idx = other.offset + kk * other.strides[0] + j * other.strides[1];
                    sum += self.storage.data[a_idx] * other.storage.data[b_idx];
                }
                result_data[i * n + j] = sum;
            }
        }

        let requires_grad = self.requires_grad || other.requires_grad;

        InternalTensor {
            storage: Rc::new(Storage::new(result_data)),
            shape: vec![m, n],
            strides: vec![n, 1],
            offset: 0,
            grad: RefCell::new(None),
            grad_fn: if requires_grad {
                Some(Rc::new(TensorMatMulTensorFn {
                    lhs: Rc::from(self.clone()),
                    rhs: Rc::from(other.clone()),
                }))
            } else {
                None
            },
            parents: if requires_grad {
                vec![self.clone(), other.clone()]
            } else {
                Vec::new()
            },
            requires_grad,
        }
        .into()
    }
}
