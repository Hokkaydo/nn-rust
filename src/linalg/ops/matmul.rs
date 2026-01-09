use crate::linalg::autograd::grad_fn::matmul::MatMulGradFn;
use crate::linalg::tensor::{InternalTensor, Storage, Tensor};
use std::cell::RefCell;
use std::rc::Rc;

impl Tensor {
    pub fn matmul(&self, other: &Tensor) -> Tensor {
        let a = match self.shape().len() {
            1 => self.unsqueeze(0), // [k] -> [1, k]
            2 => self.clone(),
            _ => panic!("matmul expects 1D or 2D tensor"),
        };

        let b = match other.shape().len() {
            1 => other.unsqueeze(1), // [k] -> [k, 1]
            2 => other.clone(),
            _ => panic!("matmul expects 1D or 2D tensor"),
        };

        assert_eq!(
            a.shape().len(),
            2,
            "Left tensor must be 2D, got shape {:?}",
            a.shape()
        );
        assert_eq!(
            b.shape().len(),
            2,
            "Right tensor must be 2D, got shape {:?}",
            b.shape()
        );
        assert_eq!(
            a.shape()[1],
            b.shape()[0],
            "Inner dimensions must match: got {} vs {}",
            a.shape()[1],
            b.shape()[0]
        );

        let [m, k] = a.shape()[..] else {
            panic!("Expected 2D shape")
        };
        let n = b.shape()[1];

        let mut result_data = vec![0.0; m * n];

        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0;
                for kk in 0..k {
                    let a_idx = a.offset + i * a.strides[0] + kk * a.strides[1];
                    let b_idx = b.offset + kk * b.strides[0] + j * b.strides[1];
                    sum += a.storage.data[a_idx] * b.storage.data[b_idx];
                }
                result_data[i * n + j] = sum;
            }
        }

        let requires_grad = a.requires_grad || b.requires_grad;
        InternalTensor {
            storage: Rc::new(Storage::new(result_data)),
            shape: vec![m, n],
            strides: vec![n, 1],
            offset: 0,
            grad: RefCell::new(None),
            grad_fn: if requires_grad {
                Some(Rc::new(MatMulGradFn {
                    lhs: a.clone(),
                    rhs: b.clone(),
                    lhs_shape: self.shape().to_vec(),
                    rhs_shape: other.shape().to_vec(),
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
