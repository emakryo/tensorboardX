from .proto.attr_value_pb2 import AttrValue
from .proto.graph_pb2 import GraphDef
from .proto.node_def_pb2 import NodeDef
from .proto.step_stats_pb2 import RunMetadata
from .proto.tensor_shape_pb2 import TensorShapeProto
from .proto.versions_pb2 import VersionDef

try:
    import chainer
    from chainer.variable import VariableNode
    from chainer.function_node import FunctionNode
    from chainer.computational_graph import build_computational_graph
    chainer_installed = True
    current_scope = []
    top_link_name = 'main'

    class FunctionNameHook(chainer.FunctionHook):
        def forward_preprocess(self, function, in_data):
            global current_scope
            function.name = '/'.join(current_scope + [function.label])

    class SetScopeHook(chainer.LinkHook):
        def __init__(self):
            self.scope = current_scope

        def forward_preprocess(self, args):
            self.scope.append(args.link.name or top_link_name)

        def forward_postprocess(self, args):
            self.scope.pop()

    class Node:
        def __init__(self, chainer_node):
            self.node = chainer_node
            if isinstance(chainer_node, VariableNode):
                self.shape = chainer_node.shape
            else:
                self.shape = None
            self.name = str(chainer_node.name)
            self.op = str(chainer_node.label)
            self.inputs = set()
            self.outputs = set()

        def to_tensorboard(self):
            if self.shape:
                shape = AttrValue(list=AttrValue.ListValue(
                    shape=[TensorShapeProto(
                        dim=[TensorShapeProto.Dim(size=d) for d in self.shape])]))
                return NodeDef(
                    name=self.name, op=self.op, attr={'_output_shapes': shape},
                    input=[i.name for i in self.inputs],
                )
            else:
                return NodeDef(
                    name=self.name, op=self.op,
                    input=[i.name for i in self.inputs],
                )

    class Graph:
        def __init__(self, output_vars):
            graph = build_computational_graph(output_vars)
            id2nodes = {id(node): Node(node) for node in graph.nodes}
            for i, o in graph.edges:
                i, o = id2nodes[id(i)], id2nodes[id(o)]
                o.inputs.add(i)
                i.outputs.add(o)

            self.nodes = list(id2nodes.values())
            self.ouputs = list(id2nodes[id(o.node)] for o in output_vars)

        @property
        def variable_nodes(self):
            return [node for node in self.nodes
                    if isinstance(node.node, VariableNode)]

        def remove_intermediate_vars(self):
            for v_node in self.variable_nodes:
                for f_in in v_node.inputs:
                    # preserve shape
                    f_in.shape = v_node.shape
                    for f_out in v_node.outputs:
                        f_in.outputs.add(f_out)
                        f_in.outputs.discard(v_node)
                        f_out.inputs.add(f_in)
                        f_out.inputs.discard(v_node)

                if len(v_node.inputs) > 0 and len(v_node.outputs) > 0:
                    del self.nodes[self.nodes.index(v_node)]

        def to_tensorboard(self):
            return GraphDef(
                node=[node.to_tensorboard() for node in self.nodes],
                versions=VersionDef(producer=22)
            )

except ImportError as e:
    chainer_installed = False


def chainer_graph(model, *input_args, remove_intermediate_vars=True,
                  **input_kwargs):
    if not chainer_installed:
        raise RuntimeError("Chainer is not installed")

    assert isinstance(model, chainer.Link)

    # change names of input variables
    default_input_name = {}
    for i, v in enumerate(input_args):
        if not isinstance(v, chainer.Variable):
            continue
        default_input_name[i] = v.name
        v.name = 'input[%d]' % i

    for k, v in input_kwargs.items():
        if not isinstance(v, chainer.Variable):
            continue
        default_input_name[k] = v.name
        v.name = 'input[%s]' % k

    with FunctionNameHook(), SetScopeHook():
        output = model(*input_args, **input_kwargs)

    # change parameter name
    for param_name, param in model.namedparams():
        param._old_name = param.name
        param.name = top_link_name + param_name

    if isinstance(output, chainer.Variable):
        output_list = [output]
        output.name = 'output'
    elif isinstance(output, list) or isinstance(output, list):
        output_list = list(output)
        for i, o in enumerate(output):
            o.name = 'output[%d]' % i
    elif isinstance(output, dict):
        output_list = list(output.values())
        for k, o in enumerate(output):
            o.name = 'output[%s]' % k
    else:
        raise ValueError('Output of model must be Variable, dict, list, '
                         'or tuple of Variable')

    if not all([isinstance(variable, chainer.Variable)
                for variable in output_list]):
        raise ValueError('Output of model must be Variable, dict, list, '
                         'or tuple of Variable')

    graph = Graph(output_list)

    # set functions output name
    for node in graph.nodes:
        if isinstance(node.node, VariableNode) \
                and len(node.inputs) > 0 and len(node.outputs) > 0:
            node.name = list(node.inputs)[0].name + '_out'

    if remove_intermediate_vars:
        graph.remove_intermediate_vars()

    # clean up
    for param in model.params():
        param.name = param._old_name
        del param._old_name

    return graph.to_tensorboard(), RunMetadata()
