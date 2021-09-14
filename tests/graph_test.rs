use sarus::{
    graph::{Connection, Graph, Node},
    run_fn,
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

    let nodes = vec![
        Node {
            func_name: "INPUT".to_string(),
            id: "INPUT".to_string(),
            port_defaults: vec![0.0],
            position: (0.0, 0.0),
        },
        Node {
            func_name: "add_node".to_string(),
            id: "add1".to_string(),
            port_defaults: vec![0.0, 0.0],
            position: (0.0, 0.0),
        },
        Node {
            func_name: "mul_node".to_string(),
            id: "mul3".to_string(),
            port_defaults: vec![0.0, 0.0],
            position: (0.0, 0.0),
        },
        Node {
            func_name: "OUTPUT".to_string(),
            id: "OUTPUT".to_string(),
            port_defaults: vec![0.0],
            position: (0.0, 0.0),
        },
        Node {
            func_name: "tanh_node".to_string(),
            id: "tanh1".to_string(),
            port_defaults: vec![0.0],
            position: (0.0, 0.0),
        },
        Node {
            func_name: "sin_node".to_string(),
            id: "sin1".to_string(),
            port_defaults: vec![0.0],
            position: (0.0, 0.0),
        },
        Node {
            func_name: "mul_node".to_string(),
            id: "mul1".to_string(),
            port_defaults: vec![0.0, 0.1],
            position: (0.0, 0.0),
        },
        Node {
            func_name: "mul_node".to_string(),
            id: "mul2".to_string(),
            port_defaults: vec![0.0, 10.0],
            position: (0.0, 0.0),
        },
    ];

    let connections = vec![
        Connection {
            src_node: 7,
            dst_node: 4,
            src_port: 0,
            dst_port: 0,
        },
        Connection {
            src_node: 7,
            dst_node: 5,
            src_port: 0,
            dst_port: 0,
        },
        Connection {
            src_node: 4,
            dst_node: 1,
            src_port: 0,
            dst_port: 0,
        },
        Connection {
            src_node: 5,
            dst_node: 1,
            src_port: 0,
            dst_port: 1,
        },
        Connection {
            src_node: 7,
            dst_node: 2,
            src_port: 0,
            dst_port: 0,
        },
        Connection {
            src_node: 5,
            dst_node: 2,
            src_port: 0,
            dst_port: 1,
        },
        Connection {
            src_node: 2,
            dst_node: 6,
            src_port: 0,
            dst_port: 0,
        },
        Connection {
            src_node: 6,
            dst_node: 3,
            src_port: 0,
            dst_port: 0,
        },
        Connection {
            src_node: 0,
            dst_node: 7,
            src_port: 0,
            dst_port: 0,
        },
    ];

    //initialize graph, will arrange graph, generate graph code, and compile
    let mut graph = Graph::new(code.to_string(), nodes, connections, STEP_SIZE).unwrap();

    //print out the resulting code for fun
    for d in graph.ast {
        println!("{}", d);
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
