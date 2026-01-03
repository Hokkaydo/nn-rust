use crate::linalg::autograd::grad_fn::shape::TransposeGradFn;
use crate::linalg::tensor_grad::Tensor;
use crate::linalg::tensor_grad::{InternalTensor, Scalar};
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
            grad_fn: self.grad_fn.clone(),
            parents: self.parents.clone(),
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

    /// Creates a tensor_old filled with ones with the specified shape.
    /// * `shape` - A slice representing the shape of the tensor_old.
    ///
    /// Returns a new tensor_old filled with ones.
    pub fn ones(shape: &[usize]) -> Self {
        let data_size = shape.iter().product();
        let data = vec![1.0; data_size];
        Self::new(data, shape)
    }

    /// Creates a tensor_old filled with zeros with the specified shape.
    /// * `shape` - A slice representing the shape of the tensor_old.
    ///
    /// Returns a new tensor_old filled with zeros.
    pub fn zeros(shape: &[usize]) -> Self {
        let data_size = shape.iter().product();
        let data = vec![0.0; data_size];
        Self::new(data, shape)
    }
}
