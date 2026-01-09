use crate::linalg::autograd::grad_fn::GradFn;
use std::cell::RefCell;
use std::fmt::Debug;
use std::ops::Deref;
use std::rc::Rc;

pub(crate) type Scalar = f32;

/// Internal storage for tensor_old data. Allows multiple tensors to share the same data.
#[derive(Clone)]
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

/// A multidimensional array (tensor) that supports automatic differentiation.
/// This wrapper struct holds a reference-counted pointer to the internal tensor representation,
/// allowing for efficient sharing and cloning of tensor data.
#[derive(Clone)]
pub struct Tensor(pub(crate) Rc<InternalTensor>);

impl Deref for Tensor {
    type Target = Rc<InternalTensor>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl From<InternalTensor> for Tensor {
    fn from(internal: InternalTensor) -> Self {
        Tensor(Rc::new(internal))
    }
}

impl Tensor {
    /// Creates a new Tensor with the given data and shape.
    pub fn new(data: Vec<Scalar>, shape: &[usize]) -> Self {
        InternalTensor::new(data, shape).into()
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

    pub fn from_scalar(value: Scalar) -> Self {
        InternalTensor {
            storage: Rc::new(Storage::new(vec![value])),
            shape: vec![1],
            strides: vec![1],
            offset: 0,
            grad: RefCell::new(None),
            grad_fn: None,
            parents: Vec::new(),
            requires_grad: false,
        }
        .into()
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

    /// Increments multidimensional indices for iterating over a stride tensor.
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

    /// Ensures this tensor has a unique (non-shared) storage.
    /// Clones the storage if it's shared with other tensors.
    pub fn make_unique(&mut self) {
        // Clone inner if shared
        let inner = Rc::make_mut(&mut self.0);

        // Clone storage if shared
        if Rc::strong_count(&inner.storage) > 1 {
            inner.storage = Rc::new(Storage {
                data: inner.storage.data.clone(),
            });
        }
    }

    /// Sets the value at the specified multidimensional indices.
    /// * `indices` - A slice of indices for each dimension of the tensor_old.
    /// * `value` - The value to set at the specified indices.
    pub fn set(&mut self, indices: &[usize], value: Scalar) {
        let inner = Rc::make_mut(&mut self.0);
        let index = inner.compute_flat_index(indices);
        let storage = Rc::make_mut(&mut inner.storage);
        storage.data[index] = value;
    }
}

pub struct InternalTensor {
    pub(crate) storage: Rc<Storage>,
    pub(crate) shape: Vec<usize>,
    pub(crate) strides: Vec<usize>,
    pub(crate) offset: usize,

    pub(crate) grad: RefCell<Option<Tensor>>,
    pub(crate) grad_fn: Option<Rc<dyn GradFn>>,
    pub(crate) parents: Vec<Tensor>,
    pub(crate) requires_grad: bool,
}

impl InternalTensor {
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
        let strides = Tensor::compute_strides(shape);
        let offset = 0;
        let storage = Rc::new(Storage::new(data));
        InternalTensor {
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

    /// Checks if the tensor_old is stored in contiguous memory.
    /// Returns true if the tensor_old is contiguous, false otherwise.
    pub(crate) fn is_contiguous(&self) -> bool {
        if self.offset != 0 {
            return false;
        }
        let expected_strides = Tensor::compute_strides(&self.shape);
        self.strides == expected_strides
    }

    /// Returns the shape of the tensor_old as a slice.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Gets the value at the specified multidimensional indices.
    /// * `indices` - A slice of indices for each dimension of the tensor_old.
    /// Returns the value at the specified indices.
    pub fn get(&self, indices: &[usize]) -> Scalar {
        self.storage.data[self.compute_flat_index(indices)]
    }

    /// Checks if the tensor_old is a scalar (i.e., has shape [1]).
    /// # Returns
    /// True if the tensor_old is a scalar, false otherwise.
    pub fn is_scalar(&self) -> bool {
        self.shape.iter().product::<usize>() == 1
    }

    /// Returns the total number of elements in the tensor_old.
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn as_scalar(&self) -> Scalar {
        if !self.is_scalar() {
            panic!("Tensor is not a scalar")
        }
        self.storage.data[self.offset]
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
            assert!(idx < self.shape[i], "Index out of bounds for dimension {i}",);
            flat_index += idx * self.strides[i];
        }
        flat_index
    }
}

impl Clone for InternalTensor {
    fn clone(&self) -> Self {
        InternalTensor {
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

impl Tensor {
    fn debug_min(&self) -> Scalar {
        *self
            .storage
            .data
            .iter()
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap()
    }

    fn debug_max(&self) -> Scalar {
        *self
            .storage
            .data
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap()
    }
}

impl Debug for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tensor")
            .field("shape", &self.shape)
            .field("strides", &self.strides)
            .field("offset", &self.offset)
            .field("requires_grad", &self.requires_grad)
            .field(
                "data",
                &format_args!(
                    "mean: {:.4}, sum: {:.4}, min: {:.4}, max: {:.4}",
                    self.mean_scalar().as_scalar(),
                    self.sum().as_scalar(),
                    self.debug_min(),
                    self.debug_max()
                ),
            )
            .field(
                "grad",
                &self
                    .grad
                    .borrow()
                    .as_ref()
                    .map(|grad| format!("Norm {:.4}", grad.norm().as_scalar()))
                    .unwrap_or("None".into()),
            )
            .field("grad_fn", &self.grad_fn.as_deref().map(|f| f.type_name()))
            .field("parents", &self.parents.len())
            .finish()
    }
}
