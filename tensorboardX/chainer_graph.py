import warnings
from .proto.attr_value_pb2 import AttrValue
from .proto.graph_pb2 import GraphDef
from .proto.node_def_pb2 import NodeDef
from .proto.step_stats_pb2 import RunMetadata
from .proto.tensor_shape_pb2 import TensorShapeProto
from .proto.versions_pb2 import VersionDef

try:
    import chainer
    from chainer.computational_graph import build_computational_graph
    chainer_installed = True
    current_scope = []
    top_link_name = 'top'


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


except ImportError as e:
    chainer_installed = False


def chainer_graph(model, *input_args, **input_kwargs):
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
    elif isinstance(output, list) or isinstance(output, list):
        output_list = list(output)
    elif isinstance(output, dict):
        output_list = list(output.values())
    else:
        raise ValueError('Output of model must be Variable, dict, list, '
                         'or tuple of Variable')

    if not all([isinstance(variable, chainer.Variable)
                for variable in output_list]):
        raise ValueError('Output of model must be Variable, dict, list, '
                         'or tuple of Variable')

    nodes = build_computational_graph(output_list).nodes

    # set functions output name
    for node in nodes:
        if isinstance(node, chainer.function_node.FunctionNode):
            for i, o in enumerate(node.outputs):
                o().name = node.name + '_out[%d]' % i

    if isinstance(output, chainer.Variable):
        output.name = 'output'
    elif isinstance(output, list) or isinstance(output, tuple):
        for i, o in enumerate(output):
            o.name = 'output[%d]' % i
    elif isinstance(output, dict):
        for k, o in enumerate(output):
            o.name = 'output[%s]' % k

    def convert(node):
        if isinstance(node, chainer.function_node.FunctionNode):
            inputs = [str(n.name) for n in node.inputs]
            return {'name': str(node.name), 'input': inputs,
                    'op': str(node.label)}
        elif isinstance(node, chainer.variable.VariableNode):
            if node.creator_node is None:
                inputs = []
            else:
                inputs = [node.creator_node.name]

            shape = AttrValue(list=AttrValue.ListValue(
                shape=[TensorShapeProto(
                    dim=[TensorShapeProto.Dim(size=d) for d in node.shape])]))
            return {'name': str(node.name), 'input': inputs,
                    'op': str(node.label), 'attr': {'_output_shapes': shape}}
        else:
            raise ValueError

    node_def = [NodeDef(**convert(node)) for node in nodes]

    # clean up
    for i, v in enumerate(input_args):
        if not isinstance(v, chainer.Variable):
            continue
        v.name = default_input_name[i]

    for k, v in input_kwargs.items():
        if not isinstance(v, chainer.Variable):
            continue
        v.name = default_input_name[k]

    for param in model.params():
        param.name = param._old_name
        del param._old_name

    return (GraphDef(node=node_def, versions=VersionDef(producer=22)),
            RunMetadata())
