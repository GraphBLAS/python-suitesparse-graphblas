class GraphBLASException(Exception):
    pass


class NoValue(GraphBLASException):
    pass


class UninitializedObject(GraphBLASException):
    pass


class InvalidObject(GraphBLASException):
    pass


class NullPointer(GraphBLASException):
    pass


class InvalidValue(GraphBLASException):
    pass


class InvalidIndex(GraphBLASException):
    pass


class DomainMismatch(GraphBLASException):
    pass


class DimensionMismatch(GraphBLASException):
    pass


class OutputNotEmpty(GraphBLASException):
    pass


class OutOfMemory(GraphBLASException):
    pass


class InsufficientSpace(GraphBLASException):
    pass


class IndexOutOfBound(GraphBLASException):
    pass


class Panic(GraphBLASException):
    pass
