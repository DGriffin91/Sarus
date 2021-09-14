use std::{collections::HashMap, fmt::Debug};

use toposort_scc::IndexGraph;

use crate::{
    frontend::{make_nonempty, parser, Binop, Cmp, Declaration, Expr},
    jit,
    validator::validate_program,
};

#[derive(Debug, Clone)]
pub struct Node {
    pub func_name: String,
    pub id: String,
    pub port_defaults: Vec<f64>,
    pub position: (f64, f64),
}

#[derive(Debug, Clone, Copy)]
pub struct Connection {
    pub src_node: usize,
    pub dst_node: usize,
    pub src_port: usize,
    pub dst_port: usize,
}

pub struct Graph {
    pub code: String,
    pub nodes: Vec<Node>,
    pub connections: Vec<Connection>,
    pub jit: jit::JIT,
    pub ast: Vec<Declaration>,
}

impl Graph {
    pub fn new(
        code: String,
        nodes: Vec<Node>,
        connections: Vec<Connection>,
        block_size: usize,
    ) -> anyhow::Result<Graph> {
        // Create the JIT instance, which manages all generated functions and data.
        let mut jit = jit::JIT::default();
        jit.add_math_constants()?;

        // Generate AST from string
        let ast = parser::program(&code)?;

        // Validate type useage
        let mut ast = validate_program(ast)?;

        let node_execution_order = order_connections(&connections, nodes.len());

        println!("Execution order:");
        for n in &node_execution_order {
            println!("{}", nodes[*n].id);
        }
        println!("");

        let graph_func_ast =
            build_graph_func(&connections, &nodes, &ast, block_size, node_execution_order)?;

        // Append graph function to ast
        ast.push(graph_func_ast);

        // Test ast again with graph function added
        let ast = validate_program(ast)?;

        // Pass the AST to the JIT to compile
        jit.translate(ast.clone())?;

        Ok(Graph {
            code,
            nodes,
            connections,
            jit,
            ast,
        })
    }
}

fn order_connections(connections: &Vec<Connection>, nodes_qty: usize) -> Vec<usize> {
    let mut g = IndexGraph::with_vertices(nodes_qty);
    for connection in connections {
        g.add_edge(connection.src_node, connection.dst_node);
    }

    let node_execution_order = g.toposort_or_scc().unwrap();
    node_execution_order
}

fn build_graph_func(
    connections: &Vec<Connection>,
    nodes: &Vec<Node>,
    ast: &Vec<Declaration>,
    block_size: usize,
    node_execution_order: Vec<usize>,
) -> anyhow::Result<Declaration> {
    let mut node_lookup = HashMap::new();
    for (i, decl) in ast.iter().enumerate() {
        node_lookup.insert(decl.name.clone(), i);
    }

    let mut main_body = Vec::new();
    let mut body = Vec::new();

    main_body.push(Expr::Assign(
        //i = 0.0
        make_nonempty(vec!["i".to_string()]).unwrap(),
        make_nonempty(vec![Expr::Literal("0.0".to_string())]).unwrap(),
    ));

    body.push(Expr::Assign(
        //vINPUT_0 = &audio[i]
        make_nonempty(vec!["vINPUT_0".to_string()]).unwrap(),
        make_nonempty(vec![Expr::ArrayGet(
            "&audio".to_string(),
            Box::new(Expr::Identifier("i".to_string())),
        )])
        .unwrap(),
    ));

    for node_idx in &node_execution_order {
        let node = &nodes[*node_idx];

        if &node.func_name == "INPUT" || &node.func_name == "OUTPUT" {
            continue;
        }
        let node_src_ast = &ast[node_lookup[&node.func_name]];

        let mut return_var_names = Vec::new();
        for (i, _ret) in node_src_ast.returns.iter().enumerate() {
            return_var_names.push(format!("v{}_{}", node.id, i))
        }

        let mut param_names = Vec::new();

        for (port, _param_name) in node_src_ast.params.iter().enumerate() {
            // find the connection that has this node and port as a dst
            let connection = connections
                .into_iter()
                .filter(|c| c.dst_node == *node_idx && c.dst_port == port)
                .collect::<Vec<&Connection>>();

            if connection.len() > 0 {
                // If a connection if found use the appropriate var name
                let connection = connection.first().unwrap();
                param_names.push(Expr::Identifier(format!(
                    "v{}_{}",
                    nodes[connection.src_node].id, connection.src_port
                )))
            } else {
                println!("{}", format!("{}", node.port_defaults[port]));
                // If there is no connection use the default val
                param_names.push(Expr::Literal(format!("{:.10}", node.port_defaults[port])))
                //TODO arbitrary precision while always printing decimal?
            }
        }

        body.push(Expr::Assign(
            make_nonempty(return_var_names).unwrap(),
            make_nonempty(vec![Expr::Call(node.func_name.clone(), param_names)]).unwrap(),
        ))
    }

    let last_node_idx = *node_execution_order.last().unwrap();
    let last_node = &nodes[last_node_idx];

    let last_connection = connections
        .into_iter()
        .filter(|c| c.dst_node == last_node_idx && c.dst_port == 0)
        .collect::<Vec<&Connection>>();

    let last_connection = last_connection.first().unwrap();

    let node_that_feeds_last_node = &nodes[last_connection.src_node];

    body.push(Expr::Assign(
        //assign last node to output
        make_nonempty(vec![format!("v{}_0", last_node.id)]).unwrap(),
        make_nonempty(vec![Expr::Identifier(format!(
            "v{}_{}",
            node_that_feeds_last_node.id, last_connection.src_port
        ))])
        .unwrap(),
    ));

    body.push(Expr::ArraySet(
        "&audio".to_string(),
        Box::new(Expr::Identifier("i".to_string())),
        Box::new(Expr::Identifier("vOUTPUT_0".to_string())),
    ));

    body.push(Expr::AssignOp(
        //i += 1.0
        Binop::Add,
        Box::new("i".to_string()),
        Box::new(Expr::Literal("1.0".to_string())),
    ));

    main_body.push(Expr::WhileLoop(
        Box::new(Expr::Compare(
            Cmp::Le,
            Box::new(Expr::Identifier("i".to_string())),
            Box::new(Expr::Literal(format!("{:.1}", (block_size - 1) as f64))),
        )),
        body,
    ));

    Ok(Declaration {
        name: "graph".to_string(),
        params: vec!["&audio".to_string()],
        returns: vec![],
        body: main_body,
    })
}
