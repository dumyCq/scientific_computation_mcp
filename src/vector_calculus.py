import numpy as np
import sympy as sp
from sympy.parsing.sympy_parser import standard_transformations, implicit_multiplication_application
from sympy import parse_expr
from sympy.vector import Del, CoordSys3D


def register_tools(mcp, matrix_store):
    # Basic vector operations

    @mcp.tool()
    def vector_project(name: str, new_vector: list[float]) -> np.ndarray:
        """
        Projects a stored vector onto another vector.

        Args:
            name (str): Name of the stored vector to project.
            new_vector (list[float]): The vector to project onto.

        Returns:
            np.ndarray: The projection result vector.

        Raises:
            ValueError: If the vector name is not found or projection fails.
        """
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
        """
        Computes the dot product between two stored vectors.

        Args:
            name_a (str): Name of the first vector in the tensor store.
            name_b (str): Name of the second vector in the tensor store.

        Returns:
            np.ndarray: Scalar result of the dot product.

        Raises:
            ValueError: If either vector is not found or if the dot product computation fails.
        """
        if name_a not in matrix_store or name_b not in matrix_store:
            raise ValueError("One or both tensor names not found in the store.")

        try:
            result = np.dot(matrix_store[name_a], matrix_store[name_b])
        except ValueError as e:
            raise ValueError(f"Error computing dot product:{e}")

        return result

    @mcp.tool()
    def vector_cross_product(name_a: str, name_b: str) -> np.ndarray:
        """
        Computes the cross product of two stored vectors.

        Args:
            name_a (str): Name of the first vector in the tensor store.
            name_b (str): Name of the second vector in the tensor store.

        Returns:
            np.ndarray: Vector result of the cross product.

        Raises:
            ValueError: If either vector is not found or if the cross product computation fails.
        """
        if name_a not in matrix_store or name_b not in matrix_store:
            raise ValueError("One or both tensor names not found in the store.")

        try:
            result = np.cross(matrix_store[name_a], matrix_store[name_b])
        except ValueError as e:
            raise ValueError(f"Error computing cross product:{e}")

        return result

    @mcp.tool()
    def gradient(f_str: str) -> str:
        f_sym = sp.sympify(f_str)
        variable = sorted(list(f_sym.free_symbols), key=lambda s: s.name)
        grad = sp.Matrix([f_sym]).jacobian(variable)
        return str(grad)

    @mcp.tool()
    def curl(f_str: str, point: list[float] = None) -> dict:
        # 1. Trim "[...]" and split
        raw = f_str.strip().strip("[]")
        comps_str = [c.strip() for c in raw.split(",")]

        # 2. Set up parser
        transformations = standard_transformations + (implicit_multiplication_application,)
        C = CoordSys3D('C')
        local_ns = {"x": C.x, "y": C.y, "z": C.z, "sin": sp.sin, "cos": sp.cos}

        # 3. Parse each component
        comp_syms = [
            parse_expr(expr, local_dict=local_ns, transformations=transformations)
            for expr in comps_str
        ]
        F = comp_syms[0] * C.i + comp_syms[1] * C.j + comp_syms[2] * C.k

        # 4. Compute symbolic curl
        curl_sym = Del().cross(F).doit()

        result = {"curl_sym": str(curl_sym)}

        if point:
            vars = [C.x, C.y, C.z]
            comps = [curl_sym.dot(dir_) for dir_ in (C.i, C.j, C.k)]
            lamb = sp.lambdify(vars, sp.Matrix(comps), 'numpy')
            result['curl_val'] = [float(v) for v in lamb(*point)]
        return result

    @mcp.tool()
    def divergence(f_str: str, point: list[float] = None) -> dict:
        # 1. Trim "[...]" and split
        raw = f_str.strip().strip("[]")
        comps_str = [c.strip() for c in raw.split(",")]

        # 2. Set up parser
        transformations = standard_transformations + (implicit_multiplication_application,)
        C = CoordSys3D('C')
        local_ns = {"x": C.x, "y": C.y, "z": C.z, "sin": sp.sin, "cos": sp.cos}

        # 3. Parse each component
        comp_syms = [
            parse_expr(expr, local_dict=local_ns, transformations=transformations)
            for expr in comps_str
        ]
        F = comp_syms[0] * C.i + comp_syms[1] * C.j + comp_syms[2] * C.k
        div_sym = Del().dot(F, doit=True)
        result = {'divergence_sym': str(div_sym)}

        if point:
            variables = [C.x, C.y, C.z]
            lamb = sp.lambdify(variables, div_sym, 'numpy')
            result['divergence_val'] = float(lamb(*point))
        return result
