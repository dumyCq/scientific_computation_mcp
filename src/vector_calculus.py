import numpy as np
from sympy.vector import CoordSys3D


def register_tools(mcp, matrix_store):
    # Basic vector operations

    @mcp.tool()
    def vector_project(name: str, new_vector: list[float]) -> np.ndarray:
        if name not in matrix_store:
            raise ValueError("The tensor name is not found in the store.")

        try:
            new_vector = np.asarray(new_vector)
            result = np.dot(matrix_store[name], new_vector) / np.linalg.norm(new_vector) * new_vector
        except ValueError as e:
            raise ValueError(f"Error computing projection:{e}")

        return result

    @mcp.tool()
    def vector_dot_product(name_a: str, name_b: str) -> np.ndarray:
        if name_a not in matrix_store or name_b not in matrix_store:
            raise ValueError("One or both tensor names not found in the store.")

        try:
            result = np.dot(matrix_store[name_a], matrix_store[name_b])
        except ValueError as e:
            raise ValueError(f"Error computing dot product:{e}")

        return result

    @mcp.tool()
    def vector_cross_product(name_a: str, name_b: str) -> np.ndarray:
        if name_a not in matrix_store or name_b not in matrix_store:
            raise ValueError("One or both tensor names not found in the store.")

        try:
            result = np.cross(matrix_store[name_a], matrix_store[name_b])
        except ValueError as e:
            raise ValueError(f"Error computing cross product:{e}")

        return result
