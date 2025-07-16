use facial_recognition::linalg::convolution::Mode;
use facial_recognition::linalg::tensor::Tensor;

#[cfg(test)]
#[test]
fn simple_convolution() {
    let input = Tensor::new(
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
            17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0,
        ],
        vec![5, 5],
    );

    let kernel = Tensor::new(
        vec![-1.0, -2.0, -1.0, -2.0, -4.0, -2.0, -1.0, -2.0, -1.0],
        vec![3, 3],
    );

    let output = input.conv2d(&kernel, Mode::Full);
    let expected_output = Tensor::new(
        vec![
            -1., -4., -8., -12., -16., -14., -5., -8., -27., -44., -56., -68., -57., -20., -24.,
            -76., -112., -128., -144., -116., -40., -44., -136., -192., -208., -224., -176., -60.,
            -64., -196., -272., -288., -304., -236., -80., -58., -177., -244., -256., -268., -207.,
            -70., -21., -64., -88., -92., -96., -74., -25.,
        ],
        vec![7, 7],
    );

    assert_eq!(output.shape, vec![7, 7]);
    println!("{:?}", output);
    for (i, &value) in output.data.iter().enumerate() {
        assert!(
            (value - expected_output.data[i]).abs() < 1e-3,
            "Mismatch at index {}: expected {}, got {}",
            i,
            expected_output.data[i],
            value
        );
    }
}

#[cfg(test)]
#[test]
fn conv2d_running_time() {
    for i in 1..=10usize {
        let size = 2usize.pow(i as u32);
        let input = Tensor::random(vec![size, size]);
        let kernel = Tensor::random(vec![3, 3]);
        let start = std::time::Instant::now();
        // let _o = input.fft();
        let _output = input.conv2d(&kernel, facial_recognition::linalg::convolution::Mode::Full);
        let duration = start.elapsed();
        println!("[{}, {}],", size * size, duration.as_secs_f64());
    }
}
