use crate::frontend::*;
use crate::validator::ArraySizedExpr;
use crate::validator::ExprType;
pub use crate::variables::*;
use cranelift::prelude::*;
pub use cranelift_jit::{JITBuilder, JITModule};
use std::collections::HashMap;
use tracing::trace;

#[derive(Debug, Clone)]
pub struct StructDef {
    pub size: usize,
    pub name: String,
    pub fields: HashMap<String, StructField>,
}

#[derive(Debug, Clone)]
pub struct StructField {
    pub offset: usize,
    pub size: usize,
    pub name: String,
    pub expr_type: ExprType,
}

pub fn create_struct_map(
    prog: &Vec<Declaration>,
    ptr_type: types::Type,
) -> anyhow::Result<HashMap<String, StructDef>> {
    let mut in_structs = HashMap::new();
    for decl in prog {
        if let Declaration::Struct(s) = decl {
            in_structs.insert(s.name.to_string(), s);
        }
    }
    let structs_order = order_structs(&in_structs)?;

    let mut structs: HashMap<String, StructDef> = HashMap::new();

    for struct_name in structs_order {
        let mut largest_field_in_struct = 0;
        let fields_def = &in_structs[&struct_name].fields;
        let mut fields = HashMap::new();
        let mut fields_v = Vec::new();
        let mut struct_size = 0usize;
        trace!("determine size of struct {}", struct_name);
        for (i, field) in fields_def.iter().enumerate() {
            let (field_size, is_struct) =
                get_field_size(&field.expr_type, &structs, ptr_type, false)?;
            let new_field = StructField {
                offset: struct_size,
                size: field_size,
                name: field.name.to_string(),
                expr_type: field.expr_type.clone(),
            };
            fields.insert(field.name.to_string(), new_field.clone());
            fields_v.push(new_field);

            struct_size += field_size;
            trace!(
                "struct {} field {} with size {} \t",
                struct_name,
                field.name,
                field_size
            );

            if i < fields_def.len() - 1 {
                //repr(C) alignment see memoffset crate

                // pad based on size of next non struct/array field,
                // or largest field if next item is struct/array of structs
                let mut field_size: usize;
                let (next_field_size, _is_struct) =
                    get_field_size(&fields_def[i + 1].expr_type, &structs, ptr_type, true)?;
                field_size = next_field_size;
                if is_struct {
                    let (this_field_size, _is_struct) =
                        get_field_size(&fields_def[i].expr_type, &structs, ptr_type, true)?;
                    field_size = field_size.max(this_field_size)
                }
                largest_field_in_struct = largest_field_in_struct.max(field_size);
                let m = struct_size % field_size;
                let padding = if m > 0 { field_size - m } else { m };
                struct_size += padding;
                if padding > 0 {
                    trace!("padding added for next field: {}", padding);
                }
            }
        }

        //Padding at end of struct
        if largest_field_in_struct > 0 {
            let m = struct_size % largest_field_in_struct;
            let padding = if m > 0 {
                largest_field_in_struct - m
            } else {
                m
            };
            struct_size += padding;
            if padding > 0 {
                trace!("{} padding added at end of struct", padding);
            }
        }

        trace!("struct {} final size {}", struct_name, struct_size);
        structs.insert(
            struct_name.to_string(),
            StructDef {
                size: struct_size,
                name: struct_name.to_string(),
                fields,
            },
        );
    }

    Ok(structs)
}

fn get_field_size(
    expr_type: &ExprType,
    struct_map: &HashMap<String, StructDef>,
    ptr_type: types::Type,
    max_base_field: bool,
) -> anyhow::Result<(usize, bool)> {
    Ok(match expr_type {
        ExprType::Struct(_code_ref, name) => {
            if max_base_field {
                (
                    get_largest_field_size(0, &expr_type, struct_map, ptr_type)?,
                    true,
                )
            } else {
                (struct_map[&name.to_string()].size, true)
            }
        }
        ExprType::Array(_code_ref, expr_type, size_type) => match size_type {
            ArraySizedExpr::Unsized => (ptr_type.bytes() as usize, false),
            ArraySizedExpr::Sized => todo!(),
            ArraySizedExpr::Fixed(len) => {
                if max_base_field {
                    get_field_size(expr_type, struct_map, ptr_type, max_base_field)?
                } else {
                    let (size, is_struct) =
                        get_field_size(expr_type, struct_map, ptr_type, max_base_field)?;
                    (size * len, is_struct)
                }
            }
        },
        _ => (
            (expr_type
                .width(ptr_type, struct_map)
                .unwrap_or(ptr_type.bytes() as usize) as usize),
            false,
        ),
    })
}

fn get_largest_field_size(
    largest: usize,
    expr_type: &ExprType,
    struct_map: &HashMap<String, StructDef>,
    ptr_type: types::Type,
) -> anyhow::Result<usize> {
    let mut largest = largest;
    match expr_type {
        ExprType::Struct(_code_ref, name) => {
            for (_name, field) in &struct_map[&name.to_string()].fields {
                let size = get_largest_field_size(largest, &field.expr_type, struct_map, ptr_type)?;
                if size > largest {
                    largest = size;
                }
            }
        }
        _ => {
            let size = expr_type
                .width(ptr_type, struct_map)
                .unwrap_or(ptr_type.bytes() as usize);
            if size > largest {
                largest = size;
            }
        }
    };
    Ok(largest)
}

fn can_insert_into_map(
    struct_name: &str,
    field_name: &str,
    expr_type: &ExprType,
    in_structs: &HashMap<String, &Struct>,
    structs_order: &Vec<String>,
    can_insert: bool,
) -> anyhow::Result<bool> {
    // if this expr's dependencies (if it has any) are already in the
    // structs_order, then we can safely add this one
    Ok(match expr_type {
        ExprType::Void(_code_ref)
        | ExprType::Bool(_code_ref)
        | ExprType::F32(_code_ref)
        | ExprType::I64(_code_ref)
        | ExprType::Address(_code_ref)
        | ExprType::Tuple(_code_ref, _) => can_insert,
        ExprType::Struct(code_ref, field_struct_name) => {
            if !in_structs.contains_key(&field_struct_name.to_string()) {
                anyhow::bail!(
                    "{} Can't find Struct {} referenced in Struct {} field {}",
                    code_ref,
                    field_struct_name,
                    struct_name,
                    field_name
                )
            }
            if structs_order.contains(&field_struct_name.to_string()) {
                can_insert
            } else {
                false
            }
        }
        ExprType::Array(_code_ref, expr_type, _size_type) => can_insert_into_map(
            struct_name,
            field_name,
            expr_type,
            in_structs,
            structs_order,
            can_insert,
        )?,
    })
}

fn order_structs(in_structs: &HashMap<String, &Struct>) -> anyhow::Result<Vec<String>> {
    // find order of structs based on dependency hierarchy
    let mut structs_order = Vec::new();
    let mut last_structs_len = 0usize;
    while structs_order.len() < in_structs.len() {
        for (name, struc) in in_structs {
            let mut can_insert = true;
            for field in &struc.fields {
                can_insert = can_insert_into_map(
                    &struc.name,
                    &field.name,
                    &field.expr_type,
                    in_structs,
                    &structs_order,
                    can_insert,
                )?
            }
            if can_insert && !structs_order.contains(&name.to_string()) {
                structs_order.push(name.to_string());
            }
        }
        if structs_order.len() > last_structs_len {
            last_structs_len = structs_order.len()
        } else {
            anyhow::bail!("Structs references resulting in loop unsupported")
        }
    }

    Ok(structs_order)
}
