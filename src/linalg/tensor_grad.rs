use crate::linalg::autograd::grad_fn::GradFn;
use std::cell::RefCell;
use std::fmt::Debug;
use std::rc::Rc;

pub(crate) type Scalar = f32;

/// Internal storage for tensor_old data. Allows multiple tensors to share the same data.
pub(crate) struct Storage {
    pub(crate) data: Vec<Scalar>,
}
impl Storage {
    /// Creates a new Storage with the given data.
    /// * `data` - A vector containing the storage data.
    pub(crate) fn new(data: Vec<Scalar>) -> Self {
        Storage { data }
    }
}

pub struct Tensor {
    pub(crate) storage: Rc<Storage>,
    pub(crate) shape: Vec<usize>,
    pub(crate) strides: Vec<usize>,
    pub(crate) offset: usize,

    pub(crate) grad: RefCell<Option<Box<Tensor>>>,
    pub(crate) grad_fn: Option<Rc<dyn GradFn>>,
    pub(crate) parents: Vec<Rc<Tensor>>,
    pub(crate) requires_grad: bool,
}

impl Tensor {
    /// Creates a new Tensor with the given data and shape.
    ///
    /// The data length must match the product of the shape dimensions.
    ///
    /// * `data` - A vector containing the tensor_old data.
    /// * `shape` - A slice representing the shape of the tensor_old.
    pub fn new(data: Vec<Scalar>, shape: &[usize]) -> Self {
        assert_eq!(
            data.len(),
            shape.iter().product(),
            "Data length does not match shape dimensions"
        );
        let strides = Self::compute_strides(shape);
        let offset = 0;
        let storage = Rc::new(Storage::new(data));
        Tensor {
            storage,
            shape: shape.to_vec(),
            strides,
            offset,
            grad: RefCell::new(None),
            grad_fn: None,
            parents: Vec::new(),
            requires_grad: false,
        }
    }

    /// Computes the strides for a given shape. Strides are used to calculate the memory offset for each dimension.
    /// * `shape` - A slice representing the shape of the tensor_old.
    ///
    /// Returns a vector containing the computed strides.
    pub(crate) fn compute_strides(shape: &[usize]) -> Vec<usize> {
        let mut strides = vec![1; shape.len()];
        for i in (0..shape.len() - 1).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        strides
    }

    /// Checks if the tensor_old is stored in contiguous memory.
    /// Returns true if the tensor_old is contiguous, false otherwise.
    pub(crate) fn is_contiguous(&self) -> bool {
        if self.offset != 0 {
            return false;
        }
        let expected_strides = Self::compute_strides(&self.shape);
        self.strides == expected_strides
    }

    /// Computes the flat index in the storage for the given multidimensional indices.
    /// # Arguments
    /// * `indices` - A slice of indices for each dimension of the tensor_old.
    /// # Returns
    /// The computed flat index.
    pub(crate) fn compute_flat_index(&self, indices: &[usize]) -> usize {
        assert_eq!(
            indices.len(),
            self.shape.len(),
            "Number of indices must match tensor dimensions"
        );
        let mut flat_index = self.offset;
        for (i, &idx) in indices.iter().enumerate() {
            assert!(
                idx < self.shape[i],
                "Index out of bounds for dimension {}",
                i
            );
            flat_index += idx * self.strides[i];
        }
        flat_index
    }

    /// Computes the multidimensional indices for a given flat index in the storage.
    /// # Arguments
    /// * `flat_index` - The flat index in the storage.
    /// # Returns
    /// A vector containing the computed multidimensional indices.
    pub(crate) fn compute_indices(&self, flat_index: usize) -> Vec<usize> {
        let mut indices = vec![0; self.shape.len()];
        let mut remaining = flat_index - self.offset;
        for i in 0..self.shape.len() {
            indices[i] = remaining / self.strides[i];
            remaining %= self.strides[i];
        }
        indices
    }

    /// Returns the shape of the tensor_old as a slice.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Gets the value at the specified multi-dimensional indices.
    /// * `indices` - A slice of indices for each dimension of the tensor_old.
    ///
    /// Returns the value at the specified indices.
    pub fn get(&self, indices: &[usize]) -> Scalar {
        self.storage.data[self.compute_flat_index(indices)]
    }

    /// Sets the value at the specified multi-dimensional indices.
    /// * `indices` - A slice of indices for each dimension of the tensor_old.
    /// * `value` - The value to set at the specified indices.
    pub fn set(&mut self, indices: &[usize], value: Scalar) {
        let index = self.compute_flat_index(indices);
        Rc::get_mut(&mut self.storage).unwrap().data[index] = value;
    }

    /// Ensures that the tensor_old has a unique copy of the storage. If the storage is shared (reference count > 1), it creates a new copy of the data
    pub(crate) fn make_unique(&mut self) {
        if Rc::strong_count(&self.storage) > 1 {
            self.storage = Rc::new(Storage::new(self.storage.data.clone()));
        }
    }

    /// Increments multi-dimensional indices for iterating over a stride tensor.
    /// * `indices` - A mutable slice of indices to increment.
    /// * `shape` - A slice representing the shape of the tensor.
    pub fn increment_indices(indices: &mut [usize], shape: &[usize]) {
        for i in (0..indices.len()).rev() {
            indices[i] += 1;
            if indices[i] < shape[i] {
                break;
            }
            indices[i] = 0;
        }
    }

    /// Checks if the tensor_old is a scalar (i.e., has shape [1]).
    /// # Returns
    /// True if the tensor_old is a scalar, false otherwise.
    pub fn is_scalar(&self) -> bool {
        self.shape.iter().product::<usize>() == 1
    }

    /// Computes the shape without dimensions of size one.
    /// # Arguments
    /// * `shape` - A slice representing the original shape.
    /// # Returns
    /// A vector representing the reduced shape.
    pub fn reduce_shape(shape: &[usize]) -> Vec<usize> {
        let reduced_shape = shape
            .iter()
            .cloned()
            .filter(|&dim| dim != 1)
            .collect::<Vec<usize>>();
        if reduced_shape.is_empty() {
            vec![1]
        } else {
            reduced_shape
        }
    }

    /// Returns the total number of elements in the tensor_old.
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn as_scalar(&self) -> Option<Scalar> {
        if self.is_scalar() {
            return Some(self.storage.data[self.offset]);
        }
        None
    }
}

impl Clone for Tensor {
    fn clone(&self) -> Self {
        Tensor {
            storage: Rc::clone(&self.storage),
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            offset: self.offset,
            grad: RefCell::new(None),
            grad_fn: self.grad_fn.clone(),
            parents: self.parents.clone(),
            requires_grad: self.requires_grad,
        }
    }
}

impl Debug for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tensor")
            .field("shape", &self.shape)
            .field("strides", &self.strides)
            .field("offset", &self.offset)
            .field("requires_grad", &self.requires_grad)
            .field("data", &self.storage.data)
            .finish()
    }
}
