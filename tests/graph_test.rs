use std::collections::HashMap;

use sarus::{
    frontend::pretty_indent,
    graph::{Connection, Graph, Node},
    hashmap, run_fn,
};

const STEP_SIZE: usize = 16usize;

#[test]
fn graph_test() -> anyhow::Result<()> {
    let code = r#"
fn add_node(a, b) -> (c) {
    c = a + b
}
fn mul_node(a, b) -> (c) {
    c = a * b
}
fn tanh_node(a) -> (c) {
    c = tanh(a)
}
fn sin_node(a) -> (c) {
    c = sin(a)
}
"#;

    let mut nodes = HashMap::new();
    nodes.insert(
        "INPUT".to_string(),
        Node {
            func_name: "INPUT".to_string(),
            port_defaults: hashmap!("a".to_string() => 0.0),
        },
    );
    nodes.insert(
        "add1".to_string(),
        Node {
            func_name: "add_node".to_string(),
            port_defaults: hashmap!("a".to_string() => 0.0, "b".to_string() => 0.0),
        },
    );
    nodes.insert(
        "OUTPUT".to_string(),
        Node {
            func_name: "OUTPUT".to_string(),
            port_defaults: hashmap!("a".to_string() => 0.0),
        },
    );
    nodes.insert(
        "tanh1".to_string(),
        Node {
            func_name: "tanh_node".to_string(),
            port_defaults: hashmap!("a".to_string() => 0.0),
        },
    );
    nodes.insert(
        "sin1".to_string(),
        Node {
            func_name: "sin_node".to_string(),
            port_defaults: hashmap!("a".to_string() => 0.0),
        },
    );
    nodes.insert(
        "mul1".to_string(),
        Node {
            func_name: "mul_node".to_string(),
            port_defaults: hashmap!("a".to_string() => 0.0, "b".to_string() => 10.0),
        },
    );
    nodes.insert(
        "mul2".to_string(),
        Node {
            func_name: "mul_node".to_string(),
            port_defaults: hashmap!("a".to_string() => 0.0, "b".to_string() => 0.2),
        },
    );

    let connections = vec![
        Connection {
            src_node: "INPUT".to_string(),
            dst_node: "mul1".to_string(),
            src_port: "src".to_string(),
            dst_port: "a".to_string(),
        },
        Connection {
            src_node: "mul1".to_string(),
            dst_node: "tanh1".to_string(),
            src_port: "c".to_string(),
            dst_port: "a".to_string(),
        },
        Connection {
            src_node: "mul1".to_string(),
            dst_node: "sin1".to_string(),
            src_port: "c".to_string(),
            dst_port: "a".to_string(),
        },
        Connection {
            src_node: "tanh1".to_string(),
            dst_node: "add1".to_string(),
            src_port: "c".to_string(),
            dst_port: "a".to_string(),
        },
        Connection {
            src_node: "sin1".to_string(),
            dst_node: "add1".to_string(),
            src_port: "c".to_string(),
            dst_port: "b".to_string(),
        },
        Connection {
            src_node: "add1".to_string(),
            dst_node: "mul2".to_string(),
            src_port: "c".to_string(),
            dst_port: "a".to_string(),
        },
        Connection {
            src_node: "mul2".to_string(),
            dst_node: "OUTPUT".to_string(),
            src_port: "c".to_string(),
            dst_port: "DST".to_string(),
        },
    ];

    //initialize graph, will arrange graph, generate graph code, and compile
    let mut graph = Graph::new(code.to_string(), nodes, connections, STEP_SIZE)?;

    //print out the resulting code for fun
    for d in graph.ast {
        println!("{}", pretty_indent(&format!("{}", d)));
    }

    // Calls to graph will be broken into STEP_SIZE chunks.
    // A pointer to the current chunk is sent to the graph.
    // The graph overwrites it with the new values
    const STEPS: usize = 48000 / STEP_SIZE;
    let mut output_arr = [[0.0f64; STEP_SIZE]; STEPS];
    let mut n = 0;
    for i in 0..STEPS {
        let mut audio_buffer = [0.0f64; STEP_SIZE];
        for j in 0..STEP_SIZE {
            audio_buffer[j] = ((n as f64).powi(2) * 0.000001).sin(); //sound source is sine sweep
            n += 1;
        }
        unsafe { run_fn(&mut graph.jit, "graph", &mut audio_buffer)? };

        //Collect output audio
        output_arr[i] = audio_buffer;
    }

    //Flatten output audio chunks for saving as wav
    let flat = output_arr
        .iter()
        .flatten()
        .map(|x| *x)
        .collect::<Vec<f64>>();
    write_wav(&flat, "graph_test.wav");
    dbg!(flat.iter().sum::<f64>());

    Ok(())
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
