from collections import defaultdict
from .proto.attr_value_pb2 import AttrValue
from .proto.graph_pb2 import GraphDef
from .proto.node_def_pb2 import NodeDef
from .proto.step_stats_pb2 import RunMetadata, StepStats, DeviceStepStats, NodeExecStats, AllocatorMemoryUsed
from .proto.tensor_shape_pb2 import TensorShapeProto
from .proto.versions_pb2 import VersionDef

try:
    import chainer
    from chainer.computational_graph import build_computational_graph
    chainer_installed = True
except ImportError as e:
    chainer_installed = False


def chainer_graph(model, input_to_model):
    if not chainer_installed:
        raise RuntimeError("Chainer is not installed")

    if isinstance(model, chainer.Link):
        if isinstance(input_to_model, chainer.Variable):
            variable = model(input_to_model)
        elif isinstance(input_to_model, list):
            variable = model(*input_to_model)
        elif isinstance(input_to_model, dict):
            variable = model(**input_to_model)
        else:
            raise ValueError

    else:
        variable = model

    if isinstance(variable, chainer.Variable):
        variables = [variable]
    else:
        variables = list(variable)

    if not all([isinstance(variable, chainer.Variable) for variable in variables]):
        raise ValueError("variable must be chainer.Variable")

    nodes = build_computational_graph(variables).nodes

    # process graph

    # nodes = [
    #     NodeDef(name='input/input', op='Input', input=[]),
    #     NodeDef(name='a/b/c/d/hidden/hidden1', op='Hidden', input=['input/input']),
    #     NodeDef(name='d/hidden/hidden2', op='Hidden2', input=['hidden/hidden1']),
    #     NodeDef(name='output', op='Output', input=['hidden/hidden2']),
    # ]

    label_count = defaultdict(int)
    id2name = {}
    for node in nodes:
        id2name[id(node)] = f'{node.label}_{label_count[node.label]}'
        label_count[node.label] += 1

    def convert(node):
        i = id(node)
        op = node.label
        name = id2name[i]
        if isinstance(node, chainer.function_node.FunctionNode):
            inputs = [id2name[id(n)] for n in node.inputs]
        elif isinstance(node, chainer.variable.VariableNode):
            if node.creator_node is None:
                inputs = []
            else:
                inputs = [id2name[id(node.creator_node)]]
        else:
            raise ValueError

        return {'name': name, 'input': inputs, 'op': op}

    node_def = [NodeDef(**convert(node)) for node in nodes]

    return GraphDef(node=node_def, versions=VersionDef(producer=22)), RunMetadata()
