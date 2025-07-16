use crate::linalg::tensor::Tensor;

#[derive(Clone, Copy, Debug)]
pub struct Complex(f32, f32);
impl Complex {
    fn new(real: f32, imag: f32) -> Self {
        Complex(real, imag)
    }

    fn conj(&self) -> Self {
        Complex(self.0, -self.1)
    }
}

impl std::ops::Add for Complex {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Complex(self.0 + other.0, self.1 + other.1)
    }
}

impl std::ops::Sub for Complex {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Complex(self.0 - other.0, self.1 - other.1)
    }
}

impl std::ops::Mul for Complex {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        Complex(
            self.0 * other.0 - self.1 * other.1,
            self.0 * other.1 + self.1 * other.0,
        )
    }
}

pub fn fft(inputs: &[Complex]) -> Vec<Complex> {
    let n = inputs.len();
    if !n.is_power_of_two() {
        panic!("FFT requires the number of inputs to be a power of two");
    }

    if n == 1 {
        return vec![inputs[0]];
    }

    let even = fft(&inputs.iter().step_by(2).cloned().collect::<Vec<_>>());
    let odd = fft(&inputs
        .iter()
        .skip(1)
        .step_by(2)
        .cloned()
        .collect::<Vec<_>>());

    let mut output = vec![Complex::new(0.0, 0.0); n];

    let angle = -2.0 * std::f32::consts::PI / n as f32;

    for j in 0..(n / 2) {
        let t = odd[j];
        let angle = angle * j as f32;
        let w = Complex::new(angle.cos(), angle.sin());
        output[j] = even[j] + w * t;
        output[j + n / 2] = even[j] - w * t;
    }
    output
}

pub fn inv_fft(inputs: &[Complex]) -> Vec<Complex> {
    let n = inputs.len();
    if !n.is_power_of_two() {
        panic!("IFFT requires the number of inputs to be a power of two");
    }

    let conj_inputs: Vec<Complex> = inputs.iter().map(|x| x.conj()).collect();
    let mut output = fft(&conj_inputs);

    for x in &mut output {
        x.0 /= n as f32;
        x.1 /= n as f32;
    }

    output
}

fn pad_to_power_of_2(data: &[Complex]) -> Vec<Complex> {
    let n = data.len();
    let next_power_of_2 = n.next_power_of_two();
    let mut padded_data = vec![Complex::new(0.0, 0.0); next_power_of_2];
    padded_data[..n].copy_from_slice(data);
    padded_data
}

// Determine the size of the output of convolution
pub enum Mode {
    Full,
    Valid,
}

impl Tensor {
    pub fn fft(&self) -> Vec<Complex> {
        if self.shape.len() != 2 {
            panic!("FFT can only be applied to 2D tensors");
        }
        let data: Vec<Complex> = self.data.iter().map(|&x| Complex::new(x, 0.0)).collect();
        fft(&*pad_to_power_of_2(&data))
    }

    pub fn inv_fft(&self) -> Vec<Complex> {
        if self.shape.len() != 2 {
            panic!("IFFT can only be applied to 2D tensors");
        }
        let data: Vec<Complex> = self.data.iter().map(|&x| Complex::new(x, 0.0)).collect();
        inv_fft(&*pad_to_power_of_2(&data))
    }

    pub fn conv2d(&self, kernel: &Tensor, mode: Mode) -> Tensor {
        assert_eq!(self.shape.len(), 2, "Input tensor must be 2D");
        assert_eq!(kernel.shape.len(), 2, "Kernel tensor must be 2D");

        let output_shape = match mode {
            Mode::Full => (
                self.shape[0] + kernel.shape[0] - 1,
                self.shape[1] + kernel.shape[1] - 1,
            ),
            Mode::Valid => (
                self.shape[0] - kernel.shape[0] + 1,
                self.shape[1] - kernel.shape[1] + 1,
            ),
        };
        let padded_self = self.pad2d_to(output_shape.0, output_shape.1);
        let padded_kernel = kernel.pad2d_to(output_shape.0, output_shape.1);

        let self_fft = padded_self.fft();
        let kernel_fft = padded_kernel.fft();

        println!("Padded Self FFT: {:?}", self_fft);
        println!("Padded Kernel FFT: {:?}", kernel_fft);

        let mut output_fft = self_fft
            .iter()
            .zip(kernel_fft.iter())
            .map(|(&a, &b)| a * b)
            .collect::<Vec<_>>();
        output_fft = inv_fft(&output_fft);

        let output_data: Vec<f32> = output_fft
            .iter()
            .map(|c| c.0)
            .take(output_shape.0 * output_shape.1)
            .collect();
        let output_shape = vec![output_shape.0, output_shape.1];

        Tensor::new(output_data, output_shape)
    }
}
