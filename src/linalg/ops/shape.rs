use crate::linalg::autograd::grad_fn::shape::TransposeGradFn;
use crate::linalg::tensor::Tensor;
use crate::linalg::tensor::{InternalTensor, Scalar};
use crate::not_implemented_grad_fn;
use std::cell::RefCell;
use std::rc::Rc;

impl Tensor {
    /// Transposes a 2D tensor_old by swapping its rows and columns.
    /// Returns a new tensor_old that is the transposed version of the original tensor_old, with updated shape and strides, without modifying the original tensor_old's data.
    pub fn transpose(&self) -> Tensor {
        assert_eq!(
            self.shape.len(),
            2,
            "Transpose only supported for 2D tensors"
        );
        let new_shape = vec![self.shape[1], self.shape[0]];
        let mut new_strides = vec![0; 2];
        new_strides[0] = self.strides[1];
        new_strides[1] = self.strides[0];

        InternalTensor {
            storage: Rc::clone(&self.storage),
            shape: new_shape,
            strides: new_strides,
            offset: self.offset,
            grad: RefCell::new(None),
            grad_fn: if self.requires_grad {
                Some(Rc::new(TransposeGradFn))
            } else {
                None
            },
            parents: if self.requires_grad {
                vec![self.clone()]
            } else {
                Vec::new()
            },
            requires_grad: self.requires_grad,
        }
        .into()
    }

    /// Reshapes the tensor_old to the specified shape without changing the underlying data.
    /// The total number of elements must remain the same.
    /// * `shape` - A slice representing the new shape of the tensor_old.
    ///
    /// Returns a new tensor_old with the specified shape.
    pub fn reshape(self, shape: &[usize]) -> Self {
        assert_eq!(
            self.shape.iter().product::<usize>(),
            shape.iter().product::<usize>(),
            "Total number of elements must remain the same when reshaping"
        );
        InternalTensor {
            storage: Rc::clone(&self.storage),
            shape: shape.to_vec(),
            strides: Self::compute_strides(shape),
            offset: self.offset,
            grad: RefCell::new(None),
            grad_fn: not_implemented_grad_fn!("reshape"),
            parents: vec![],
            requires_grad: self.requires_grad,
        }
        .into()
    }

    /// Returns the underlying data of the tensor_old as a slice. If the tensor_old is not contiguous or has a non-zero offset, this will panic.
    ///
    /// Returns a slice of the tensor_old's data.
    pub fn as_slice(&self) -> &[Scalar] {
        assert!(
            self.is_contiguous(),
            "Tensor must be contiguous to get as slice"
        );
        assert_eq!(self.offset, 0, "Tensor offset must be zero to get as slice");
        &self.storage.data
    }

    /// Returns the underlying data of the tensor_old as a mutable slice. If the tensor_old is not contiguous or has a non-zero offset, this will panic.
    /// If the storage is shared, it will create a unique copy before returning the mutable slice.
    ///
    /// Returns a mutable slice of the tensor_old's data.
    /// Clones the tensor's storage if it's shared with other tensors (copy-on-write).
    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        // Rc::make_mut clones if refcount > 1
        let inner = Rc::make_mut(&mut self.0);
        let storage = Rc::make_mut(&mut inner.storage);
        &mut storage.data
    }

    pub(crate) fn unsqueeze(&self, dim: usize) -> Tensor {
        let mut shape = self.shape.clone();
        let mut strides = self.strides.clone();

        let stride = if dim < self.strides.len() {
            self.strides[dim] * self.shape[dim]
        } else {
            1
        };

        shape.insert(dim, 1);
        strides.insert(dim, stride);

        InternalTensor {
            storage: self.storage.clone(),
            shape,
            strides,
            offset: self.offset,
            grad: RefCell::new(None),
            grad_fn: not_implemented_grad_fn!("unsqueeze"),
            parents: vec![],
            requires_grad: self.requires_grad,
        }
        .into()
    }
}
