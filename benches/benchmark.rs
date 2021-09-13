#![feature(test)]
extern crate test; //rust-analyser complains, but this should work in nightly

use basic_audio_filters::second_order_iir::{IIR2Coefficients, IIR2};
use cranelift_jit_demo::*;
use test::Bencher;

fn rand64(x: f64) -> f64 {
    // Crappy noise
    ((x * 12000000.9898).sin() * 43758.5453).fract()
}

fn filter_benchmark_1(iterations: f64, output_array: &mut [f64]) -> f64 {
    let fs = 48000.0;
    let mut filter1 = IIR2::from(IIR2Coefficients::highpass(100.0, 0.0, 1.0, fs));
    let mut filter2 = IIR2::from(IIR2Coefficients::lowpass(5000.0, 0.0, 1.0, fs));
    let mut filter3 = IIR2::from(IIR2Coefficients::highshelf(2000.0, 6.0, 1.0, fs));
    let mut sum = 0.0;
    let mut i = 0.0;
    while i < iterations {
        let n = (i * 0.01).sin();
        filter1.coeffs = IIR2Coefficients::highpass(n * 100.0 + 200.0, 0.0, 1.0, fs);
        filter2.coeffs = IIR2Coefficients::lowpass(n * 100.0 + 2000.0, 0.0, 1.0, fs);
        filter3.coeffs = IIR2Coefficients::highshelf(n * 100.0 + 1000.0, 6.0, 1.0, fs);
        let mut sample = (rand64(i) * 100000.0).floor() * 0.00001;
        sample = filter1.process(sample);
        sample = filter2.process(sample);
        sample = filter3.process(sample);
        sum += sample;
        output_array[i as usize] = sample;
        i += 1.0;
    }
    //dbg!(IIR2Coefficients::highshelf(2000.0, 6.0, 1.0, fs));
    sum
}

#[bench]
fn test_static_filter_benchmark_1(b: &mut Bencher) {
    let mut result = 0.0;
    let mut output_arr = [0.0f64; 48000];
    b.iter(|| {
        //test::black_box({
            result = filter_benchmark_1(48000.0, &mut output_arr);
        //});
    });
    dbg!((result, output_arr.iter().sum::<f64>()));
}

fn get_eq_jit() -> jit::JIT {
    let code = r#"
    fn rand(x) -> (y) { // Crappy noise
        y = fract(sin(x * 12000000.9898) * 43758.5453)
    }
    fn highpass(cutoff_hz, q_value, sample_rate_hz) -> (a1, a2, a3, m0, m1, m2) {
        cutoff_hz = min(cutoff_hz, sample_rate_hz * 0.5)
        a = 1.0
        g = tan(PI * cutoff_hz / sample_rate_hz)
        k = 1.0 / q_value
        a1 = 1.0 / (1.0 + g * (g + k))
        a2 = g * a1
        a3 = g * a2
        m0 = 1.0
        m1 = 0.0 - k
        m2 = -1.0
    }
    fn lowpass(cutoff_hz, q_value, sample_rate_hz) -> (a1, a2, a3, m0, m1, m2) {
        cutoff_hz = min(cutoff_hz, sample_rate_hz * 0.5)
        a = 1.0
        g = tan(PI * cutoff_hz / sample_rate_hz)
        k = 1.0 / q_value
        a1 = 1.0 / (1.0 + g * (g + k))
        a2 = g * a1
        a3 = g * a2
        m0 = 0.0
        m1 = 0.0
        m2 = 1.0
    }
    fn highshelf(cutoff_hz, gain_db, q_value, sample_rate_hz) -> (a1, a2, a3, m0, m1, m2) {
        cutoff_hz = min(cutoff_hz, sample_rate_hz * 0.5)
        a = pow(10.0, gain_db / 40.0)
        g = tan(PI * cutoff_hz / sample_rate_hz) * sqrt(a)
        k = 1.0 / q_value
        a1 = 1.0 / (1.0 + g * (g + k))
        a2 = g * a1
        a3 = g * a2
        m0 = a * a
        m1 = k * (1.0 - a) * a
        m2 = 1.0 - a * a
    }
    fn process(x, ic1eq, ic2eq, a1, a2, a3, m0, m1, m2) -> (n_x, n_ic1eq, n_ic2eq) {
        v3 = x - ic2eq
        v1 = a1 * ic1eq + a2 * v3
        v2 = ic2eq + a2 * ic1eq + a3 * v3
        n_ic1eq = 2.0 * v1 - ic1eq
        n_ic2eq = 2.0 * v2 - ic2eq
        n_x = m0 * x + m1 * v1 + m2 * v2
    }
    fn main(iterations, &output_arr) -> (sum) {
        fs = 48000.0
        f1_a1, f1_a2, f1_a3, f1_m0, f1_m1, f1_m2 = highpass(100.0, 1.0, fs)
        f2_a1, f2_a2, f2_a3, f2_m0, f2_m1, f2_m2 = lowpass(5000.0, 1.0, fs)
        f3_a1, f3_a2, f3_a3, f3_m0, f3_m1, f3_m2 = highshelf(2000.0, 6.0, 1.0, fs)
        f1_ic1eq, f1_ic2eq = 0.0, 0.0
        f2_ic1eq, f2_ic2eq = 0.0, 0.0
        f3_ic1eq, f3_ic2eq = 0.0, 0.0
        
        sum = 0.0
        i = 0.0
        while i < iterations {
            n = sin(i*0.01)        
            f1_a1, f1_a2, f1_a3, f1_m0, f1_m1, f1_m2 = highpass(n * 100.0 + 200.0, 1.0, fs)
            f2_a1, f2_a2, f2_a3, f2_m0, f2_m1, f2_m2 = lowpass(n * 100.0 + 2000.0, 1.0, fs)
            f3_a1, f3_a2, f3_a3, f3_m0, f3_m1, f3_m2 = highshelf(n * 100.0 + 1000.0, 6.0, 1.0, fs)
            sample = floor(rand(i) * 100000.0) * 0.00001
            sample, f1_ic1eq, f1_ic2eq = process(sample, f1_ic1eq, f1_ic2eq, f1_a1, f1_a2, f1_a3, f1_m0, f1_m1, f1_m2)
            sample, f2_ic1eq, f2_ic2eq = process(sample, f2_ic1eq, f2_ic2eq, f2_a1, f2_a2, f2_a3, f2_m0, f2_m1, f2_m2)
            sample, f3_ic1eq, f3_ic2eq = process(sample, f3_ic1eq, f3_ic2eq, f3_a1, f3_a2, f3_a3, f3_m0, f3_m1, f3_m2)    
            sum += sample
            &output_arr[i] = sample
            i += 1.0
        }
        f3_a1, f3_a2, f3_a3, f3_m0, f3_m1, f3_m2 = highshelf(2000.0, 6.0, 1.0, fs)
    }

"#;

    let mut jit = jit::JIT::default();
    jit.add_math_constants().unwrap();
    let ast = frontend::parser::program(&code).unwrap();
    let ast = validator::validate_program(ast).unwrap();
    jit.translate(ast).unwrap();
    jit
}

#[bench]
fn eq_compile(b: &mut Bencher) {
    let mut sum = 0.0;
    b.iter(|| {
        test::black_box({
            let mut jit = get_eq_jit();
            let mut output_arr = [0.0f64; 2];
            let result: f64 = unsafe { run_fn(&mut jit, "main", (2.0, &mut output_arr)).unwrap() };
            sum += result;
        });
    });
    dbg!(sum);
}

#[bench]
fn eq(b: &mut Bencher) {
    let mut jit = get_eq_jit();
    let mut result = 0.0;
    let mut output_arr = [0.0f64; 48000];
    b.iter(|| {
        test::black_box(unsafe {
            result = run_fn(&mut jit, "main", (48000.0, &mut output_arr)).unwrap();
        });
    });
    dbg!((result, output_arr.iter().sum::<f64>()));
}

#[test]
fn compare_eq() {
    let iterations = 48000.0;
    let mut output_arr = [0.0f64; 48000];
    let mut jit = get_eq_jit();
    let result: f64 = unsafe { run_fn(&mut jit, "main", (iterations, &mut output_arr)).unwrap() };
    //println!("{:?}", &output_arr[0..10]);
    let mut output_arr2 = [0.0f64; 48000];
    let result2 = filter_benchmark_1(iterations, &mut output_arr2);
    println!("{} {}", result, result2);
    //println!("{:?}", output_arr2);
    write_wav(&output_arr, "sc.wav");
    write_wav(&output_arr2, "ru.wav");
}

fn write_wav(samples: &[f64], path: &str) {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 48000,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut writer = hound::WavWriter::create(path, spec).unwrap();
    for sample in samples {
        writer.write_sample(*sample as f32).unwrap();
    }
    writer.finalize().unwrap();
}
