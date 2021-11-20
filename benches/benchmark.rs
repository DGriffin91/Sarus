#![feature(test)]
extern crate test; //rust-analyser complains, but this should work in nightly

use std::mem;

use basic_audio_filters::second_order_iir::{IIR2Coefficients, IIR2};
use sarus::*;
use test::Bencher;

fn rand64(x: f32) -> f32 {
    // Crappy noise
    ((x * 12.9898).sin() * 43758.5453).fract()
}

fn filter_benchmark_1(iterations: usize, output_array: &mut [f32]) -> f32 {
    let fs = 48000.0;
    let mut filter1 = IIR2::from(IIR2Coefficients::highpass(100.0, 0.0, 1.0, fs));
    let mut filter2 = IIR2::from(IIR2Coefficients::lowpass(5000.0, 0.0, 1.0, fs));
    let mut filter3 = IIR2::from(IIR2Coefficients::highshelf(2000.0, 6.0, 1.0, fs));
    let mut sum = 0.0;
    let mut i = 0usize;
    let mut f_i = 0.0f32;
    while i < iterations {
        let n = (f_i * 0.01).sin();
        filter1.coeffs = IIR2Coefficients::highpass(n * 100.0 + 200.0, 0.0, 1.0, fs);
        filter2.coeffs = IIR2Coefficients::lowpass(n * 100.0 + 2000.0, 0.0, 1.0, fs);
        filter3.coeffs = IIR2Coefficients::highshelf(n * 100.0 + 1000.0, 6.0, 1.0, fs);
        let mut sample = (rand64(f_i) * 100000.0).floor() * 0.00001;
        sample = filter1.process(sample);
        sample = filter2.process(sample);
        sample = filter3.process(sample);
        sum += sample;
        output_array[i as usize] = sample;
        i += 1;
        f_i += 1.0;
    }
    sum
}

#[bench]
fn test_static_filter_benchmark_1(b: &mut Bencher) {
    let mut result = 0.0;
    let mut output_arr = [0.0f32; 48000];
    b.iter(|| {
        //test::black_box({
        result = filter_benchmark_1(48000usize, &mut output_arr);
        //});
    });
    dbg!((result, output_arr.iter().sum::<f32>()));
}
const EQ_BENCH_CODE: &str = r#"
inline fn rand(x) -> (y) { // Crappy noise
    y = ((x * 12.9898).sin() * 43758.5453).fract()
}
inline fn highpass(cutoff_hz, q_value, sample_rate_hz) -> (a1, a2, a3, m0, m1, m2) {
    cutoff_hz = cutoff_hz.min(sample_rate_hz * 0.5)
    a = 1.0
    g = (PI * cutoff_hz / sample_rate_hz).tan()
    k = 1.0 / q_value
    a1 = 1.0 / (1.0 + g * (g + k))
    a2 = g * a1
    a3 = g * a2
    m0 = 1.0
    m1 = 0.0 - k
    m2 = -1.0
}
inline fn lowpass(cutoff_hz, q_value, sample_rate_hz) -> (a1, a2, a3, m0, m1, m2) {
    cutoff_hz = cutoff_hz.min(sample_rate_hz * 0.5)
    a = 1.0
    g = (PI * cutoff_hz / sample_rate_hz).tan()
    k = 1.0 / q_value
    a1 = 1.0 / (1.0 + g * (g + k))
    a2 = g * a1
    a3 = g * a2
    m0 = 0.0
    m1 = 0.0
    m2 = 1.0
}
inline fn highshelf(cutoff_hz, gain_db, q_value, sample_rate_hz) -> (a1, a2, a3, m0, m1, m2) {
    cutoff_hz = cutoff_hz.min(sample_rate_hz * 0.5)
    a = (10.0).powf(gain_db / 40.0)
    g = (PI * cutoff_hz / sample_rate_hz).tan() * a.sqrt()
    k = 1.0 / q_value
    a1 = 1.0 / (1.0 + g * (g + k))
    a2 = g * a1
    a3 = g * a2
    m0 = a * a
    m1 = k * (1.0 - a) * a
    m2 = 1.0 - a * a
}
inline fn process(x, ic1eq, ic2eq, a1, a2, a3, m0, m1, m2) -> (n_x, n_ic1eq, n_ic2eq) {
    v3 = x - ic2eq
    v1 = a1 * ic1eq + a2 * v3
    v2 = ic2eq + a2 * ic1eq + a3 * v3
    n_ic1eq = 2.0 * v1 - ic1eq
    n_ic2eq = 2.0 * v2 - ic2eq
    n_x = m0 * x + m1 * v1 + m2 * v2
}

Slice for &[f32]
fn main(iterations: i64, output_arr: &[f32]) -> (sum) {
    output_slice = output_arr.into_slice(iterations)
    fs = 48000.0
    f1_a1, f1_a2, f1_a3, f1_m0, f1_m1, f1_m2 = highpass(100.0, 1.0, fs)
    f2_a1, f2_a2, f2_a3, f2_m0, f2_m1, f2_m2 = lowpass(5000.0, 1.0, fs)
    f3_a1, f3_a2, f3_a3, f3_m0, f3_m1, f3_m2 = highshelf(2000.0, 6.0, 1.0, fs)
    f1_ic1eq, f1_ic2eq = 0.0, 0.0
    f2_ic1eq, f2_ic2eq = 0.0, 0.0
    f3_ic1eq, f3_ic2eq = 0.0, 0.0
    sum = 0.0
    i = 0
    f_i = 0.0
    while i < iterations {
        n = (f_i*0.01).sin()   
        f1_a1, f1_a2, f1_a3, f1_m0, f1_m1, f1_m2 = highpass(n * 100.0 + 200.0, 1.0, fs)
        f2_a1, f2_a2, f2_a3, f2_m0, f2_m1, f2_m2 = lowpass(n * 100.0 + 2000.0, 1.0, fs)
        f3_a1, f3_a2, f3_a3, f3_m0, f3_m1, f3_m2 = highshelf(n * 100.0 + 1000.0, 6.0, 1.0, fs)
        sample = (rand(f_i) * 100000.0).floor() * 0.00001
        sample, f1_ic1eq, f1_ic2eq = process(sample, f1_ic1eq, f1_ic2eq, f1_a1, f1_a2, f1_a3, f1_m0, f1_m1, f1_m2)
        sample, f2_ic1eq, f2_ic2eq = process(sample, f2_ic1eq, f2_ic2eq, f2_a1, f2_a2, f2_a3, f2_m0, f2_m1, f2_m2)
        sample, f3_ic1eq, f3_ic2eq = process(sample, f3_ic1eq, f3_ic2eq, f3_a1, f3_a2, f3_a3, f3_m0, f3_m1, f3_m2)    
        sum += sample
        output_slice.set(i, sample)
        i += 1
        f_i += 1.0
    }
}
"#;

fn get_eq_jit() -> jit::JIT {
    let mut ast = frontend::parse(EQ_BENCH_CODE).unwrap();
    let mut jit_builder = jit::new_jit_builder();
    sarus_std_lib::append_std(&mut ast, &mut jit_builder);
    sarus_std_lib::append_std_math(&mut ast, &mut jit_builder);

    let mut jit = jit::JIT::from(jit_builder, true);
    jit.translate(ast, None).unwrap();
    jit
}

#[bench]
fn eq_compile(b: &mut Bencher) {
    let mut sum = 0.0;
    b.iter(|| {
        test::black_box({
            let mut jit = get_eq_jit();
            let mut output_arr = [0.0f32; 128];

            let func_ptr = jit.get_func("main").unwrap();
            let func =
                unsafe { mem::transmute::<_, extern "C" fn(i64, *mut f32) -> f32>(func_ptr) };

            sum += func(output_arr.len() as i64, output_arr.as_mut_ptr());
        });
    });
    dbg!(sum);
}

#[bench]
fn eq_compile_only(b: &mut Bencher) {
    let ast = frontend::parse(EQ_BENCH_CODE).unwrap();
    let mut sum = 0.0;
    b.iter(|| {
        test::black_box({
            let mut ast = ast.clone();
            let mut jit_builder = jit::new_jit_builder();
            sarus_std_lib::append_std(&mut ast, &mut jit_builder);
            sarus_std_lib::append_std_math(&mut ast, &mut jit_builder);

            let mut jit = jit::JIT::from(jit_builder, true);
            jit.translate(ast, None).unwrap();
            let mut output_arr = [0.0f32; 128];

            let func_ptr = jit.get_func("main").unwrap();
            let func =
                unsafe { mem::transmute::<_, extern "C" fn(i64, *mut f32) -> f32>(func_ptr) };

            sum += func(output_arr.len() as i64, output_arr.as_mut_ptr());
        });
    });
    dbg!(sum);
}

#[bench]
fn eq(b: &mut Bencher) {
    let mut jit = get_eq_jit();
    let mut result = 0.0;
    let mut output_arr = [0.0f32; 48000];
    b.iter(|| {
        test::black_box({
            let func_ptr = jit.get_func("main").unwrap();
            let func =
                unsafe { mem::transmute::<_, extern "C" fn(i64, *mut f32) -> f32>(func_ptr) };
            result = func(output_arr.len() as i64, output_arr.as_mut_ptr())
        });
    });
    dbg!((result, output_arr.iter().sum::<f32>()));
}

#[test]
fn compare_eq() {
    let iterations = 48000;
    let mut output_arr_sarus = [0.0f32; 48000];
    let mut jit = get_eq_jit();
    let func_ptr = jit.get_func("main").unwrap();
    let func = unsafe { mem::transmute::<_, extern "C" fn(i64, *mut f32) -> f32>(func_ptr) };
    let result: f32 = func(output_arr_sarus.len() as i64, output_arr_sarus.as_mut_ptr());
    //println!("{:?}", &output_arr[0..10]);
    let mut output_arr_rust = [0.0f32; 48000];
    let result2 = filter_benchmark_1(iterations as usize, &mut output_arr_rust);

    //for (a, b) in output_arr_sarus.iter().zip(output_arr_rust.iter()) {
    //    if *a != *b {
    //        println!("{}, {}", a, b)
    //    }
    //}

    println!("{} {}", result, result2);
    //println!("{:?}", output_arr2);
    write_wav(&output_arr_sarus, "sc.wav");
    write_wav(&output_arr_rust, "ru.wav");
}

fn write_wav(samples: &[f32], path: &str) {
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

#[bench]
fn compile_bench(b: &mut Bencher) {
    //setup_logging();
    let code = r#"
    fn main() -> (c) {
        if true {
            c = 7.4 * 2.7
        } else {
            c = 7.4 * 2.9
        }
    }
"#;
    b.iter(|| {
        test::black_box({
            let ast = parse(&code).unwrap();
            let jit_builder = jit::new_jit_builder();
            //sarus_std_lib::append_std(&mut ast, &mut jit_builder);
            //sarus_std_lib::append_std_math(&mut ast, &mut jit_builder);

            let mut jit = jit::JIT::from(jit_builder, true);

            jit.translate(ast, None).unwrap();
            let func_ptr = jit.get_func("main").unwrap();
            let func = unsafe { mem::transmute::<_, extern "C" fn()>(func_ptr) };
            func();
        });
    });
}
