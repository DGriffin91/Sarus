use std::{collections::HashMap, fmt::Debug};

use toposort_scc::IndexGraph;

use crate::{
    frontend::{
        assign_op_to_assign, make_nonempty, parser, Arg, Binop, Cmp, CodeRef, Declaration, Expr,
        Function, InlineKind,
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
        let mut ast = parser::program(&code)?;
        let mut jit_builder = jit::new_jit_builder();
        sarus_std_lib::append_std(&mut ast, &mut jit_builder);
        sarus_std_lib::append_std_math(&mut ast, &mut jit_builder);

        let mut jit = jit::JIT::from(jit_builder);

        let node_execution_order = order_connections(&connections, &nodes);

        println!("Execution order:");
        for node_id in &node_execution_order {
            println!("{}", node_id);
        }
        println!("");

        let graph_func_ast =
            build_graph_func(&connections, &nodes, &ast, block_size, node_execution_order)?;

        // Append graph function to ast
        ast.push(graph_func_ast);

        // Pass the AST to the JIT to compile
        jit.translate(ast.clone(), code.to_string())?;

        Ok(Graph {
            code,
            nodes,
            connections,
            jit,
            ast,
        })
    }
}

fn order_connections(connections: &Vec<Connection>, nodes: &HashMap<String, Node>) -> Vec<String> {
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
    connections: &Vec<Connection>,
    nodes: &HashMap<String, Node>,
    ast: &Vec<Declaration>,
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

    main_body.push(Expr::Assign(
        //i = 0
        CodeRef::z(),
        make_nonempty(vec![Expr::Identifier(CodeRef::z(), "i".to_string())]).unwrap(),
        make_nonempty(vec![Expr::LiteralInt(CodeRef::z(), "0".to_string())]).unwrap(),
    ));

    body.push(Expr::Assign(
        //vINPUT_0 = audio[i]
        CodeRef::z(),
        make_nonempty(vec![Expr::Identifier(
            CodeRef::z(),
            "vINPUT_src".to_string(),
        )])
        .unwrap(),
        make_nonempty(vec![Expr::ArrayAccess(
            CodeRef::z(),
            "audio".to_string(),
            Box::new(Expr::Identifier(CodeRef::z(), "i".to_string())),
        )])
        .unwrap(),
    ));

    for node_id in &node_execution_order {
        let node = &nodes[node_id];

        if &node.func_name == "INPUT" || &node.func_name == "OUTPUT" {
            continue;
        }
        let node_src_ast = &funcs[&node.func_name];

        let mut return_var_names = Vec::new();
        for ret in node_src_ast.returns.iter() {
            return_var_names.push(Expr::Identifier(
                CodeRef::z(),
                format!("v{}_{}", node_id, ret),
            ))
        }

        let mut param_names = Vec::new();

        for param in node_src_ast.params.iter() {
            // find the connection that has this node and port as a dst
            let connection = connections
                .into_iter()
                .filter(|c| c.dst_node == *node_id && c.dst_port == *param.name)
                .collect::<Vec<&Connection>>();

            if connection.len() > 0 {
                // If a connection if found use the appropriate var name
                let connection = connection.first().unwrap();
                param_names.push(Expr::Identifier(
                    CodeRef::z(),
                    format!("v{}_{}", &connection.src_node, connection.src_port),
                ))
            } else {
                println!("{}", format!("{}", node.port_defaults[&param.name]));
                // If there is no connection use the default val
                param_names.push(Expr::LiteralFloat(
                    CodeRef::z(),
                    format!("{:.10}", node.port_defaults[&param.name]),
                ))
                //TODO arbitrary precision while always printing decimal?
            }
        }

        body.push(Expr::Assign(
            CodeRef::z(),
            make_nonempty(return_var_names).unwrap(),
            make_nonempty(vec![Expr::Call(
                CodeRef::z(),
                node.func_name.clone(),
                param_names,
                false,
            )])
            .unwrap(),
        ))
    }

    let last_node_id = node_execution_order.last().unwrap();

    let last_connection = connections
        .into_iter()
        .filter(|c| c.dst_node == *last_node_id)
        .collect::<Vec<&Connection>>();

    let last_connection = last_connection.first().unwrap();

    body.push(Expr::Assign(
        CodeRef::z(),
        //assign last node to output
        make_nonempty(vec![Expr::Identifier(
            CodeRef::z(),
            format!("v{}_dst", last_node_id),
        )])
        .unwrap(),
        make_nonempty(vec![Expr::Identifier(
            CodeRef::z(),
            format!(
                "v{}_{}",
                &last_connection.src_node, last_connection.src_port
            ),
        )])
        .unwrap(),
    ));

    body.push(Expr::Assign(
        CodeRef::z(),
        make_nonempty(vec![Expr::ArrayAccess(
            CodeRef::z(),
            "audio".to_string(),
            Box::new(Expr::Identifier(CodeRef::z(), "i".to_string())),
        )])
        .unwrap(),
        make_nonempty(vec![Expr::Identifier(
            CodeRef::z(),
            "vOUTPUT_dst".to_string(),
        )])
        .unwrap(),
    ));

    body.push(assign_op_to_assign(
        Binop::Add,
        Expr::Identifier(CodeRef::z(), "i".to_string()),
        Expr::LiteralInt(CodeRef::z(), "1".to_string()),
    ));

    main_body.push(Expr::WhileLoop(
        CodeRef::z(),
        Box::new(Expr::Compare(
            CodeRef::z(),
            Cmp::Le,
            Box::new(Expr::Identifier(CodeRef::z(), "i".to_string())),
            Box::new(Expr::LiteralInt(
                CodeRef::z(),
                format!("{}", (block_size - 1) as f32),
            )),
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
        }],
        returns: vec![],
        body: main_body,
        extern_func: false,
        inline: InlineKind::Default,
    }))
}
