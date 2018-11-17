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


def chainer_graph(model, input_to_model):
    if not chainer_installed:
        raise RuntimeError("Chainer is not installed")

    assert isinstance(model, chainer.Link)

    with FunctionNameHook(), SetScopeHook():
        if isinstance(input_to_model, chainer.Variable):
            input_name = input_to_model.name
            input_to_model.name = 'input'
            variable = model(input_to_model)
        elif isinstance(input_to_model, list):
            input_name = [x.name for x in input_to_model]
            for i, x in enumerate(input_to_model):
                x.name = 'input[%d]' % i

            variable = model(*input_to_model)
        elif isinstance(input_to_model, dict):
            input_name = {k: v.name for k, v in input_to_model.items()}
            for k, v in input_to_model.items():
                v.name = 'input[%s]' % k

            variable = model(**input_to_model)
        else:
            raise ValueError

    # change parameter name
    for param_name, param in model.namedparams():
        param._old_name = param.name
        param.name = top_link_name + param_name

    if isinstance(variable, chainer.Variable):
        variables = [variable]
    elif isinstance(variable, list):
        variables = variable
    elif isinstance(variable, dict):
        variables = list(variable.values())
    else:
        raise ValueError('Output of model must be Variable, dict of Variable, or list of Variable')

    if not all([isinstance(variable, chainer.Variable) for variable in variables]):
        raise ValueError('Output of model must be Variable, dict of Variable, or list of Variable')

    nodes = build_computational_graph(variables).nodes

    # set functions output name
    for node in nodes:
        if isinstance(node, chainer.function_node.FunctionNode):
            for i, output in enumerate(node.outputs):
                output().name = node.name + '_out[%d]' % i

    for i, v in enumerate(variables):
        v.name = 'output[%d]' % i

    def convert(node):
        if isinstance(node, chainer.function_node.FunctionNode):
            op = node.name
            inputs = [n.name for n in node.inputs]
        elif isinstance(node, chainer.variable.VariableNode):
            op = node.label
            if node.creator_node is None:
                inputs = []
            else:
                inputs = [node.creator_node.name]
        else:
            raise ValueError

        return {'name': node.name, 'input': inputs, 'op': str(op)}

    node_def = [NodeDef(**convert(node)) for node in nodes]

    # clean up
    if isinstance(model, chainer.Link):
        if isinstance(input_to_model, chainer.Variable):
            input_to_model.name = input_name
        elif isinstance(input_to_model, list):
            for i, v in enumerate(input_to_model):
                v.name = input_name[i]
        elif isinstance(input_to_model, dict):
            for k, v in input_to_model.items():
                v.name = input_name[k]

    for param in model.params():
        param.name = param._old_name
        del param._old_name

    return GraphDef(node=node_def, versions=VersionDef(producer=22)), RunMetadata()
