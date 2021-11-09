use std::{collections::HashMap, fmt::Debug};

use toposort_scc::IndexGraph;

use crate::{
    frontend::{
        assign_op_to_assign, parse, Arg, Binop, Cmp, CodeRef, Declaration, Expr, Function,
        InlineKind,
    },
    jit, sarus_std_lib,
    validator::{ArraySizedExpr, ExprType},
};

#[derive(Debug, Clone)]
pub struct Node {
    pub func_name: String,
    pub port_defaults: HashMap<String, f32>,
}

#[derive(Debug, Clone)]
pub struct Connection {
    pub src_node: String,
    pub dst_node: String,
    pub src_port: String,
    pub dst_port: String,
}

pub struct Graph {
    pub code: String,
    pub nodes: HashMap<String, Node>,
    pub connections: Vec<Connection>,
    pub jit: jit::JIT,
    pub ast: Vec<Declaration>,
}

impl Graph {
    pub fn new(
        code: String,
        nodes: HashMap<String, Node>,
        connections: Vec<Connection>,
        block_size: usize,
    ) -> anyhow::Result<Graph> {
        let mut ast = parse(&code)?;
        let mut jit_builder = jit::new_jit_builder();
        sarus_std_lib::append_std(&mut ast, &mut jit_builder);
        sarus_std_lib::append_std_math(&mut ast, &mut jit_builder);

        let mut jit = jit::JIT::from(jit_builder);

        let node_execution_order = order_connections(&connections, &nodes);

        println!("Execution order:");
        for node_id in &node_execution_order {
            println!("{}", node_id);
        }
        println!();

        let graph_func_ast =
            build_graph_func(&connections, &nodes, &ast, block_size, node_execution_order)?;

        // Append graph function to ast
        ast.push(graph_func_ast);

        // Pass the AST to the JIT to compile
        jit.translate(ast.clone(), None)?;

        Ok(Graph {
            code,
            nodes,
            connections,
            jit,
            ast,
        })
    }
}

fn order_connections(connections: &[Connection], nodes: &HashMap<String, Node>) -> Vec<String> {
    // TODO probably implement our own toposort
    let node_indices = nodes
        .iter()
        .map(|(k, _v)| k.to_string())
        .collect::<Vec<String>>();
    let mut node_map = HashMap::new();
    for (i, (k, _v)) in nodes.iter().enumerate() {
        node_map.insert(k, i);
    }
    let mut g = IndexGraph::with_vertices(nodes.len());
    for connection in connections {
        g.add_edge(
            node_map[&connection.src_node],
            node_map[&connection.dst_node],
        );
    }

    let node_execution_order = g.toposort_or_scc().unwrap();

    let mut order = Vec::new();
    for idx in &node_execution_order {
        order.push(node_indices[*idx].clone());
    }
    order
}

fn build_graph_func(
    connections: &[Connection],
    nodes: &HashMap<String, Node>,
    ast: &[Declaration],
    block_size: usize,
    node_execution_order: Vec<String>,
) -> anyhow::Result<Declaration> {
    let mut funcs = HashMap::new();
    for decl in ast.iter() {
        if let Declaration::Function(func) = decl {
            funcs.insert(func.name.clone(), func.clone());
        }
    }

    let mut main_body = Vec::new();
    let mut body = Vec::new();

    main_body.push(Expr::assign(
        &vec![Expr::identifier("i")],
        &vec![Expr::literal_int(0)],
    ));

    body.push(Expr::assign(
        &vec![Expr::identifier("vINPUT_src")],
        &vec![Expr::array_access(
            &Expr::identifier("audio"),
            &Expr::identifier("i"),
        )],
    ));

    for node_id in &node_execution_order {
        let node = &nodes[node_id];

        if &node.func_name == "INPUT" || &node.func_name == "OUTPUT" {
            continue;
        }
        let node_src_ast = &funcs[&node.func_name];

        let mut return_var_names = Vec::new();
        for ret in node_src_ast.returns.iter() {
            return_var_names.push(Expr::identifier(&format!("v{}_{}", node_id, ret)))
        }

        let mut param_names = Vec::new();

        for param in node_src_ast.params.iter() {
            // find the connection that has this node and port as a dst
            let connection = connections
                .iter()
                .filter(|c| c.dst_node == *node_id && c.dst_port == *param.name)
                .collect::<Vec<&Connection>>();

            if !connection.is_empty() {
                // If a connection if found use the appropriate var name
                let connection = connection.first().unwrap();
                param_names.push(Expr::identifier(&format!(
                    "v{}_{}",
                    &connection.src_node, connection.src_port
                )))
            } else {
                println!("{}", format!("{}", node.port_defaults[&param.name]));
                // If there is no connection use the default val
                param_names.push(Expr::literal_float(node.port_defaults[&param.name]))
                //TODO arbitrary precision while always printing decimal?
            }
        }

        body.push(Expr::assign(
            &return_var_names,
            &vec![Expr::Call(
                CodeRef::z(),
                node.func_name.clone(),
                param_names,
                false,
            )],
        ))
    }

    let last_node_id = node_execution_order.last().unwrap();

    let last_connection = connections
        .iter()
        .filter(|c| c.dst_node == *last_node_id)
        .collect::<Vec<&Connection>>();

    let last_connection = last_connection.first().unwrap();

    body.push(Expr::assign(
        //assign last node to output
        &vec![Expr::identifier(&format!("v{}_dst", last_node_id))],
        &vec![Expr::identifier(&format!(
            "v{}_{}",
            &last_connection.src_node, last_connection.src_port
        ))],
    ));

    body.push(Expr::assign(
        &vec![Expr::array_access(
            &Expr::identifier("audio"),
            &Expr::identifier("i"),
        )],
        &vec![Expr::identifier("vOUTPUT_dst")],
    ));

    body.push(assign_op_to_assign(
        Binop::Add,
        Expr::identifier("i"),
        Expr::literal_int(1),
    ));

    main_body.push(Expr::WhileLoop(
        CodeRef::z(),
        Box::new(Expr::Compare(
            CodeRef::z(),
            Cmp::Le,
            Box::new(Expr::identifier("i")),
            Box::new(Expr::literal_int((block_size - 1) as i64)),
        )),
        body,
    ));

    Ok(Declaration::Function(Function {
        name: "graph".to_string(),
        params: vec![Arg {
            name: "audio".into(),
            expr_type: ExprType::Array(
                CodeRef::z(),
                Box::new(ExprType::F32(CodeRef::z())),
                ArraySizedExpr::Unsized,
            ),
            default_to_float: false,
            closure_arg: None,
        }],
        returns: vec![],
        body: main_body,
        extern_func: false,
        inline: InlineKind::Default,
    }))
}
