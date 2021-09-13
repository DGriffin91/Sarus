#![feature(test)]
extern crate test; //rust-analyser complains, but this should work in nightly

use std::collections::HashMap;

use basic_audio_filters::second_order_iir::{IIR2Coefficients, IIR2};
use test::Bencher;

const STEP_SIZE: usize = 128usize;

#[test]
fn compare_eq() {
    const STEPS: usize = 48000 / STEP_SIZE;
    let mut output_arr = [[0.0f64; STEP_SIZE]; STEPS];

    let (mut nodes, mut connections) = build_graph();
    for i in 0..STEPS {
        output_arr[i] = process_graph_step(&mut nodes, &mut connections);
    }
    let flat = output_arr
        .iter()
        .flatten()
        .map(|x| *x)
        .collect::<Vec<f64>>();
    write_wav(&flat, "ng.wav");
    dbg!(flat.iter().sum::<f64>());
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
//7,871,635
#[bench]
fn benchmark(b: &mut Bencher) {
    const STEPS: usize = 48000 / STEP_SIZE;
    let mut output_arr = [[0.0f64; STEP_SIZE]; STEPS];

    //test::black_box({
        //move this into iter for exactly correct output_arr result
        let (mut nodes, mut connections) = build_graph();
        b.iter(|| {
            for i in 0..STEPS {
                output_arr[i] = process_graph_step(&mut nodes, &mut connections);
            }
        });
    //});
    let flat = output_arr
        .iter()
        .flatten()
        .map(|x| *x)
        .collect::<Vec<f64>>();
    dbg!(flat.iter().sum::<f64>());
}

///////////////////////////////
///////////////////////////////
///////////////////////////////
pub trait Node {
    fn get_output(&self, name: &str) -> [f64; STEP_SIZE];
    fn set_input(&mut self, name: &str, val: [f64; STEP_SIZE]);
    fn process(&mut self);
    fn mark_process(&mut self);
}

struct Connection {
    src_node: String,
    src_port: String,
    dst_node: String,
    dst_port: String,
}

fn build_graph() -> (HashMap<String, Box<dyn Node>>, Vec<Connection>) {
    let mut nodes: HashMap<String, Box<dyn Node>> = HashMap::new();
    nodes.insert(
        "step".to_string(),
        Box::new(Step {
            step: [-1.0f64; STEP_SIZE], //so we start on 0
            need_processes: true,
        }),
    );
    nodes.insert(
        "lfo".to_string(),
        Box::new(Sin {
            in_val: [0.0f64; STEP_SIZE],
            out_val: [0.0f64; STEP_SIZE],
            need_processes: true,
        }),
    );
    nodes.insert(
        "noise".to_string(),
        Box::new(Noise {
            in_val: [0.0f64; STEP_SIZE],
            out_val: [0.0f64; STEP_SIZE],
            need_processes: true,
        }),
    );
    nodes.insert(
        "highpass".to_string(),
        Box::new(Highpass {
            filter: IIR2::from(IIR2Coefficients::highpass(1000.0, 0.0, 1.0, 48000.0)),
            in_val: [0.0f64; STEP_SIZE],
            out_val: [0.0f64; STEP_SIZE],
            cutoff_val: [0.0f64; STEP_SIZE],
            need_processes: true,
        }),
    );
    nodes.insert(
        "lowpass".to_string(),
        Box::new(Lowpass {
            filter: IIR2::from(IIR2Coefficients::lowpass(5000.0, 0.0, 1.0, 48000.0)),
            in_val: [0.0f64; STEP_SIZE],
            out_val: [0.0f64; STEP_SIZE],
            cutoff_val: [0.0f64; STEP_SIZE],
            need_processes: true,
        }),
    );
    nodes.insert(
        "highshelf".to_string(),
        Box::new(Highshelf {
            filter: IIR2::from(IIR2Coefficients::highshelf(2000.0, 6.0, 1.0, 48000.0)),
            in_val: [0.0f64; STEP_SIZE],
            out_val: [0.0f64; STEP_SIZE],
            cutoff_val: [0.0f64; STEP_SIZE],
            need_processes: true,
        }),
    );
    nodes.insert(
        "output".to_string(),
        Box::new(Output {
            out_val: [0.0f64; STEP_SIZE],
        }),
    );
    let connections = vec![
        Connection {
            src_node: "STEPEVENT".to_string(),
            src_port: "audio".to_string(),
            dst_node: "step".to_string(),
            dst_port: "audio".to_string(),
        },
        Connection {
            src_node: "step".to_string(),
            src_port: "audio".to_string(),
            dst_node: "lfo".to_string(),
            dst_port: "audio".to_string(),
        },
        Connection {
            src_node: "step".to_string(),
            src_port: "audio".to_string(),
            dst_node: "noise".to_string(),
            dst_port: "audio".to_string(),
        },
        Connection {
            src_node: "lfo".to_string(),
            src_port: "audio".to_string(),
            dst_node: "highpass".to_string(),
            dst_port: "cutoff".to_string(),
        },
        Connection {
            src_node: "lfo".to_string(),
            src_port: "audio".to_string(),
            dst_node: "lowpass".to_string(),
            dst_port: "cutoff".to_string(),
        },
        Connection {
            src_node: "lfo".to_string(),
            src_port: "audio".to_string(),
            dst_node: "highshelf".to_string(),
            dst_port: "cutoff".to_string(),
        },
        Connection {
            src_node: "noise".to_string(),
            src_port: "audio".to_string(),
            dst_node: "highpass".to_string(),
            dst_port: "audio".to_string(),
        },
        Connection {
            src_node: "highpass".to_string(),
            src_port: "audio".to_string(),
            dst_node: "lowpass".to_string(),
            dst_port: "audio".to_string(),
        },
        Connection {
            src_node: "lowpass".to_string(),
            src_port: "audio".to_string(),
            dst_node: "highshelf".to_string(),
            dst_port: "audio".to_string(),
        },
        Connection {
            src_node: "highshelf".to_string(),
            src_port: "audio".to_string(),
            dst_node: "output".to_string(),
            dst_port: "audio".to_string(),
        },
    ];
    (nodes, connections)
}

fn process_graph_step(
    nodes: &mut HashMap<String, Box<dyn Node>>,
    connections: &mut Vec<Connection>,
) -> [f64; STEP_SIZE] {
    for conn in connections {
        let output = {
            if &conn.src_node != "STEPEVENT" {
                let src_node = nodes.get_mut(&conn.src_node).unwrap();
                src_node.process();
                src_node.get_output(&conn.src_port)
            } else {
                //nodes.get_mut(&conn.dst_node).unwrap().process();
                [0.0f64; STEP_SIZE]
            }
        };
        nodes
            .get_mut(&conn.dst_node)
            .unwrap()
            .set_input(&conn.dst_port, output);
    }
    for (_, node) in nodes.into_iter() {
        node.mark_process();
    }
    nodes["output"].get_output("audio")
}

fn rand64(x: f64) -> f64 {
    // Crappy noise
    ((x * 12000000.9898).sin() * 43758.5453).fract()
}

pub struct Noise {
    in_val: [f64; STEP_SIZE],
    out_val: [f64; STEP_SIZE],
    need_processes: bool,
}

impl Node for Noise {
    fn mark_process(&mut self) {
        self.need_processes = true;
    }
    fn process(&mut self) {
        if self.need_processes {
            self.need_processes = false;
            for (out, inp) in self.out_val.iter_mut().zip(self.in_val) {
                *out = (rand64(inp) * 100000.0).floor() * 0.00001;
            }
            //dbg!("PROCESS Noise", self.out_val);
        }
    }
    fn get_output(&self, _name: &str) -> [f64; STEP_SIZE] {
        self.out_val
    }
    fn set_input(&mut self, _name: &str, val: [f64; STEP_SIZE]) {
        self.in_val = val;
    }
}

pub struct Sin {
    in_val: [f64; STEP_SIZE],
    out_val: [f64; STEP_SIZE],
    need_processes: bool,
}

impl Node for Sin {
    fn mark_process(&mut self) {
        self.need_processes = true;
    }
    fn process(&mut self) {
        if self.need_processes {
            self.need_processes = false;
            for (out, inp) in self.out_val.iter_mut().zip(self.in_val) {
                *out = (inp * 0.01).sin();
            }
            //dbg!("PROCESS Sin", self.out_val);
        }
    }
    fn get_output(&self, _name: &str) -> [f64; STEP_SIZE] {
        self.out_val
    }
    fn set_input(&mut self, _name: &str, val: [f64; STEP_SIZE]) {
        self.in_val = val;
    }
}

pub struct Output {
    out_val: [f64; STEP_SIZE],
}

impl Node for Output {
    fn mark_process(&mut self) {}
    fn process(&mut self) {}
    fn get_output(&self, _name: &str) -> [f64; STEP_SIZE] {
        self.out_val
    }
    fn set_input(&mut self, _name: &str, val: [f64; STEP_SIZE]) {
        self.out_val = val;
    }
}

pub struct Highpass {
    filter: IIR2,
    out_val: [f64; STEP_SIZE],
    in_val: [f64; STEP_SIZE],
    cutoff_val: [f64; STEP_SIZE],
    need_processes: bool,
}

impl Node for Highpass {
    fn mark_process(&mut self) {
        self.need_processes = true;
    }
    fn process(&mut self) {
        if self.need_processes {
            self.need_processes = false;
            for ((out, inp), cutoff) in self
                .out_val
                .iter_mut()
                .zip(self.in_val)
                .zip(self.cutoff_val)
            {
                self.filter.update(IIR2Coefficients::highpass(
                    cutoff * 100.0 + 200.0,
                    0.0,
                    1.0,
                    48000.0,
                ));
                *out = self.filter.process(inp);
            }
            //dbg!("PROCESS Lowpass", self.out_val);
        }
    }
    fn get_output(&self, _name: &str) -> [f64; STEP_SIZE] {
        self.out_val
    }
    fn set_input(&mut self, name: &str, val: [f64; STEP_SIZE]) {
        if name == "audio" {
            self.in_val = val;
        } else if name == "cutoff" {
            self.cutoff_val = val;
        }
    }
}

pub struct Lowpass {
    filter: IIR2,
    out_val: [f64; STEP_SIZE],
    in_val: [f64; STEP_SIZE],
    cutoff_val: [f64; STEP_SIZE],
    need_processes: bool,
}

impl Node for Lowpass {
    fn mark_process(&mut self) {
        self.need_processes = true;
    }
    fn process(&mut self) {
        if self.need_processes {
            self.need_processes = false;
            for ((out, inp), cutoff) in self
                .out_val
                .iter_mut()
                .zip(self.in_val)
                .zip(self.cutoff_val)
            {
                self.filter.update(IIR2Coefficients::lowpass(
                    cutoff * 100.0 + 2000.0,
                    0.0,
                    1.0,
                    48000.0,
                ));
                *out = self.filter.process(inp);
            }
            //dbg!("PROCESS Lowpass", self.out_val);
        }
    }
    fn get_output(&self, _name: &str) -> [f64; STEP_SIZE] {
        self.out_val
    }
    fn set_input(&mut self, name: &str, val: [f64; STEP_SIZE]) {
        if name == "audio" {
            self.in_val = val;
        } else if name == "cutoff" {
            self.cutoff_val = val;
        }
    }
}

pub struct Highshelf {
    filter: IIR2,
    out_val: [f64; STEP_SIZE],
    in_val: [f64; STEP_SIZE],
    cutoff_val: [f64; STEP_SIZE],
    need_processes: bool,
}

impl Node for Highshelf {
    fn mark_process(&mut self) {
        self.need_processes = true;
    }
    fn process(&mut self) {
        if self.need_processes {
            self.need_processes = false;
            for ((out, inp), cutoff) in self
                .out_val
                .iter_mut()
                .zip(self.in_val)
                .zip(self.cutoff_val)
            {
                self.filter.update(IIR2Coefficients::highshelf(
                    cutoff * 100.0 + 1000.0,
                    6.0,
                    1.0,
                    48000.0,
                ));
                *out = self.filter.process(inp);
            }
            //dbg!("PROCESS Lowpass", self.out_val);
        }
    }
    fn get_output(&self, _name: &str) -> [f64; STEP_SIZE] {
        self.out_val
    }
    fn set_input(&mut self, name: &str, val: [f64; STEP_SIZE]) {
        if name == "audio" {
            self.in_val = val;
        } else if name == "cutoff" {
            self.cutoff_val = val;
        }
    }
}

pub struct Step {
    step: [f64; STEP_SIZE],
    need_processes: bool,
}

impl Node for Step {
    fn mark_process(&mut self) {
        self.need_processes = true;
    }
    fn process(&mut self) {
        if self.need_processes {
            self.need_processes = false;
            let mut current_step = *self.step.last().unwrap();
            for step in self.step.iter_mut() {
                current_step += 1.0;
                *step = current_step;
            }
            //dbg!("PROCESS Step", self.step);
        }
    }
    fn get_output(&self, _name: &str) -> [f64; STEP_SIZE] {
        self.step
    }
    fn set_input(&mut self, _name: &str, _val: [f64; STEP_SIZE]) {}
}
