# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

from controllers.acados_interface import mpc_pb2 as rpc_dot_mpc__pb2


class ModelPredictiveControllerStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.initialize_solver = channel.unary_unary(
                '/ModelPredictiveController/initialize_solver',
                request_serializer=rpc_dot_mpc__pb2.Settings.SerializeToString,
                response_deserializer=rpc_dot_mpc__pb2.Response.FromString,
                )
        self.solve = channel.unary_unary(
                '/ModelPredictiveController/solve',
                request_serializer=rpc_dot_mpc__pb2.Problem.SerializeToString,
                response_deserializer=rpc_dot_mpc__pb2.Solution.FromString,
                )
        self.learn_from_data = channel.unary_unary(
                '/ModelPredictiveController/learn_from_data',
                request_serializer=rpc_dot_mpc__pb2.LearningData.SerializeToString,
                response_deserializer=rpc_dot_mpc__pb2.Response.FromString,
                )
        self.done = channel.unary_unary(
                '/ModelPredictiveController/done',
                request_serializer=rpc_dot_mpc__pb2.Empty.SerializeToString,
                response_deserializer=rpc_dot_mpc__pb2.Empty.FromString,
                )


class ModelPredictiveControllerServicer(object):
    """Missing associated documentation comment in .proto file."""

    def initialize_solver(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def solve(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def learn_from_data(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def done(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_ModelPredictiveControllerServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'initialize_solver': grpc.unary_unary_rpc_method_handler(
                    servicer.initialize_solver,
                    request_deserializer=rpc_dot_mpc__pb2.Settings.FromString,
                    response_serializer=rpc_dot_mpc__pb2.Response.SerializeToString,
            ),
            'solve': grpc.unary_unary_rpc_method_handler(
                    servicer.solve,
                    request_deserializer=rpc_dot_mpc__pb2.Problem.FromString,
                    response_serializer=rpc_dot_mpc__pb2.Solution.SerializeToString,
            ),
            'learn_from_data': grpc.unary_unary_rpc_method_handler(
                    servicer.learn_from_data,
                    request_deserializer=rpc_dot_mpc__pb2.LearningData.FromString,
                    response_serializer=rpc_dot_mpc__pb2.Response.SerializeToString,
            ),
            'done': grpc.unary_unary_rpc_method_handler(
                    servicer.done,
                    request_deserializer=rpc_dot_mpc__pb2.Empty.FromString,
                    response_serializer=rpc_dot_mpc__pb2.Empty.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'ModelPredictiveController', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class ModelPredictiveController(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def initialize_solver(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/ModelPredictiveController/initialize_solver',
            rpc_dot_mpc__pb2.Settings.SerializeToString,
            rpc_dot_mpc__pb2.Response.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def solve(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/ModelPredictiveController/solve',
            rpc_dot_mpc__pb2.Problem.SerializeToString,
            rpc_dot_mpc__pb2.Solution.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def learn_from_data(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/ModelPredictiveController/learn_from_data',
            rpc_dot_mpc__pb2.LearningData.SerializeToString,
            rpc_dot_mpc__pb2.Response.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def done(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/ModelPredictiveController/done',
            rpc_dot_mpc__pb2.Empty.SerializeToString,
            rpc_dot_mpc__pb2.Empty.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
