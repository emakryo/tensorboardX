from .proto.attr_value_pb2 import AttrValue
from .proto.graph_pb2 import GraphDef
from .proto.node_def_pb2 import NodeDef
from .proto.step_stats_pb2 import RunMetadata, StepStats, DeviceStepStats, NodeExecStats, AllocatorMemoryUsed
from .proto.tensor_shape_pb2 import TensorShapeProto
from .proto.versions_pb2 import VersionDef

try:
    import chainer
    chainer_installed = True
except ImportError as e:
    chainer_installed = False


def chainer_graph(variable):
    if not chainer_installed:
        raise RuntimeError("Chainer is not installed")

    if not isinstance(variable, chainer.Variable):
        raise ValueError("variable must be chainer.Variable")

    return GraphDef(), RunMetadata()
