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


def chainer_graph(variables):
    if not chainer_installed:
        raise RuntimeError("Chainer is not installed")

    if not all([isinstance(variable, chainer.Variable) for variable in variables]):
        raise ValueError("variable must be chainer.Variable")

    graph = build_computational_graph(variables)

    # process graph

    # nodes = [
    #     NodeDef(name='input/input', op='Input', input=[]),
    #     NodeDef(name='a/b/c/d/hidden/hidden1', op='Hidden', input=['input/input']),
    #     NodeDef(name='d/hidden/hidden2', op='Hidden2', input=['hidden/hidden1']),
    #     NodeDef(name='output', op='Output', input=['hidden/hidden2']),
    # ]

    id2idx = {id(node): i for i, node in enumerate(graph.nodes)}

    def convert(node):
        i = id(node)
        name = f'node{id2idx[i]}'
        op = node.label
        if isinstance(node, chainer.function_node.FunctionNode):
            inputs = [f'node{id2idx[id(n)]}' for n in node.inputs]
        elif isinstance(node, chainer.variable.VariableNode):
            if node.creator_node is None:
                inputs = []
            else:
                inputs = [f'node{id2idx[id(node.creator_node)]}']
        else:
            raise Exception

        return {'name': name, 'input': inputs, 'op': op}

    nodes = [NodeDef(**convert(node)) for node in graph.nodes]

    return GraphDef(node=nodes, versions=VersionDef(producer=22)), RunMetadata()
