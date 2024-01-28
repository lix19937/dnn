
import onnx
from loguru import logger  

# find first node by name_str match input
def find_with_input_node(model, name):
    for node in model.graph.node:
        if len(node.input) > 0 and name in node.input:
            return node

# find nodes by name_str match input
def find_all_with_input_node(model, name):
    all = []
    for node in model.graph.node:
        if len(node.input) > 0 and name in node.input:
            all.append(node)
    return all

# find node by name_str match output
def find_with_output_node(model, name):
    for node in model.graph.node:
        if len(node.output) > 0 and name in node.output:
            return node

# 
def find_with_no_change_parent_node(model, node):
    parent = find_with_output_node(model, node.input[0])
    if parent is not None:
      # ref https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#qdq-limitations
        if parent.op_type in ["Concat", "MaxPool"]: # note: "AveragePool"] not commute 
            return find_with_no_change_parent_node(model, parent)
    return parent

# 
def find_quantizelinear_conv(model, qnode):
    dq   = find_with_input_node(model, qnode.output[0])
    conv = find_with_input_node(model, dq.output[0])

    if conv.op_type == "Conv": return conv, 1
    if conv.op_type == "AveragePool": return conv, 0
    if conv.op_type == "Pad":
      pad = conv
      avg = find_with_input_node(model, pad.output[0])
      if avg.op_type == "AveragePool":
        return avg, 0
      else:
        logger.exception("not support, {} ".format(avg));exit(0)
    logger.exception("not support, {} ".format(conv));exit(0)


def find_quantize_conv_name(model, conv, idx, lonlp_map):
    if conv.op_type != "Conv": 
      lonlp_list = lonlp_map[0]
      if len(lonlp_list) > 0:
        lonlp_idx = lonlp_map[1]
        nstr = lonlp_list[lonlp_idx]
        lonlp_map[1] = lonlp_map[1] + 1
        return nstr
      else:
        logger.error("some error case")
        return None 

    weight_qname = conv.input[idx]
    dq = find_with_output_node(model, weight_qname)
    if len(dq.input) > 0:
      q  = find_with_output_node(model, dq.input[0])
      nstr = ".".join(q.input[0].split(".")[:-1]) # resBlock1.conv5.weight --> resBlock1.conv5
      return nstr
  
    logger.exception("dq has no input\n{}\nweight_qname:{}".format(dq, weight_qname)) 
    return None


def find_quantizer_pairs(onnx_file, lonlp: list = []) -> list:
    # logger.info("load model:{}".format(onnx_file))
    model = onnx.load(onnx_file)
    match_pairs = []
    lonlp_map = [lonlp, 0]
    for node in model.graph.node:   
        if node.op_type == "Concat":
            # concat -> q  maybe concat has some subnodes
            qnodes = find_all_with_input_node(model, node.output[0])
            major = None
            for qnode in qnodes:
                if qnode.op_type != "QuantizeLinear":
                    continue
                
                # q-dq-conv or q-dq-pad-avgpool
                conv, idx = find_quantizelinear_conv(model, qnode)
                if major is None:
                    major = find_quantize_conv_name(model, conv, idx, lonlp_map)
                else:
                    ext = find_quantize_conv_name(model, conv, idx, lonlp_map)
                    if ext is not None:
                        match_pairs.append([major, ext])
						  
                for subnode in model.graph.node:
                    if len(subnode.input) > 0 and subnode.op_type == "QuantizeLinear" and subnode.input[0] in node.input:
                        subconv, idx = find_quantizelinear_conv(model, subnode)
                        ext = find_quantize_conv_name(model, subconv, idx, lonlp_map)
                        if ext is not None:
                            match_pairs.append([major, ext])
						
        elif node.op_type == "MaxPool": 
            qnode = find_with_input_node(model, node.output[0])
            if not (qnode and qnode.op_type == "QuantizeLinear"):
                continue

            major, idx = find_quantizelinear_conv(model, qnode)
            major = find_quantize_conv_name(model, major, idx, lonlp_map)
            same_input_nodes = find_all_with_input_node(model, node.input[0])

            for same_input_node in same_input_nodes:
                if same_input_node.op_type == "QuantizeLinear":
                    subconv, idx = find_quantizelinear_conv(model, same_input_node)
                    ext = find_quantize_conv_name(model, subconv, idx, lonlp_map)
                    if major is not None and ext is not None:
                        match_pairs.append([major, ext])
    
    return match_pairs
