use crate::linalg::autograd::grad_fn::binary::AddGradFn;
use crate::linalg::tensor::{InternalTensor, Storage, Tensor};
use std::cell::RefCell;
use std::rc::Rc;

impl Tensor {
    /// Broadcast addition of a tensor along last dimensions
    /// # Arguments
    /// * `other` - The tensor to add, must be broadcastable to self
    /// # Returns
    /// A new tensor containing the result of the broadcast addition
    pub fn broadcast_add(&self, other: &Tensor) -> Tensor {
        let self_shape = self.shape();
        let self_strides = &self.strides;
        let self_rank = self_shape.len();

        let other_shape = other.shape();
        let other_rank = other_shape.len();

        assert!(self_rank >= other_rank);

        // Pad other shape and strides
        let mut padded_other_shape = vec![1; self_rank - other_rank];
        padded_other_shape.extend_from_slice(other_shape);

        let mut padded_other_strides = vec![0; self_rank - other_rank];
        padded_other_strides.extend_from_slice(&other.strides);

        // Validate broadcast and set stride to 0 for broadcast dimensions
        for d in 0..self_rank {
            if padded_other_shape[d] == 1 {
                padded_other_strides[d] = 0;
            } else {
                assert_eq!(
                    padded_other_shape[d], self_shape[d],
                    "Broadcast mismatch at dim {d}"
                );
            }
        }

        let a = &self.storage.data;
        let b = &other.storage.data;
        let n = a.len();
        let mut out = vec![0.0; n];

        // Pre-compute division factors to avoid repeated division
        let mut div_factors = vec![1; self_rank];
        for d in (0..self_rank - 1).rev() {
            div_factors[d] = div_factors[d + 1] * self_shape[d + 1];
        }

        for linear_idx in 0..n {
            let mut b_offset = 0;
            for d in 0..self_rank {
                let coord = (linear_idx / div_factors[d]) % self_shape[d];
                b_offset += coord * padded_other_strides[d];
            }

            out[linear_idx] = a[linear_idx] + b[b_offset];
        }

        InternalTensor {
            storage: Rc::new(Storage::new(out)),
            shape: self_shape.to_vec(),
            strides: self_strides.clone(),
            offset: 0,
            grad: RefCell::new(None),
            grad_fn: if self.requires_grad || other.requires_grad {
                Some(Rc::new(AddGradFn::new(vec![self.clone(), other.clone()])))
            } else {
                None
            },
            parents: if self.requires_grad || other.requires_grad {
                vec![self.clone(), other.clone()]
            } else {
                Vec::new()
            },
            requires_grad: self.requires_grad || other.requires_grad,
        }
        .into()
    }
}
