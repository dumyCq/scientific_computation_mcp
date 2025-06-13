from io import BytesIO
from mcp.server.fastmcp import Image
from sympy import symbols
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from sympy.parsing.sympy_parser import standard_transformations, implicit_multiplication_application, parse_expr

x, y, z = symbols("x y z")


def register_tools(mcp):
    @mcp.tool()
    def plot_vector_field(f_str: str, bounds=(-1, 1, -1, 1, -1, 1), n: int = 10) -> Image:
        """
        Plots a 3D vector field from a string "[u(x,y,z), v(x,y,z), w(x,y,z)]"

        Args:
            f_str: string representation of 3D field, e.g. "[z, -y, x]".
            bounds: (xmin, xmax, ymin, ymax, zmin, zmax)
            n: grid resolution per axis

        Returns: Displayed Matplotlib 3D quiver plot (no image return needed)
        """
        # 1. Extract component strings
        raw = f_str.strip().lstrip("[").rstrip("]")
        u_s, v_s, w_s = [s.strip() for s in raw.split(",")]

        # 2. Parse each component with bare symbols
        transforms = standard_transformations + (implicit_multiplication_application,)
        local_ns = {"x": x, "y": y, "z": z, "sin": sp.sin, "cos": sp.cos}
        u_expr = parse_expr(u_s, local_dict=local_ns, transformations=transforms)
        v_expr = parse_expr(v_s, local_dict=local_ns, transformations=transforms)
        w_expr = parse_expr(w_s, local_dict=local_ns, transformations=transforms)

        # 3. Convert to numpy functions
        u_fn = sp.lambdify((x, y, z), u_expr, "numpy")
        v_fn = sp.lambdify((x, y, z), v_expr, "numpy")
        w_fn = sp.lambdify((x, y, z), w_expr, "numpy")

        # 4. Prepare grid
        xmin, xmax, ymin, ymax, zmin, zmax = bounds
        X, Y, Z = np.meshgrid(
            np.linspace(xmin, xmax, n),
            np.linspace(ymin, ymax, n),
            np.linspace(zmin, zmax, n),
            indexing="ij"
        )
        U = u_fn(X, Y, Z)
        V = v_fn(X, Y, Z)
        W = w_fn(X, Y, Z)

        # 5. Plot quiver
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(projection="3d")
        ax.quiver(X, Y, Z, U, V, W, length=0.1, normalize=True, color="blue")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(f"3D Vector Field: {f_str}")
        plt.tight_layout()
        # Save to buffer and return
        buf = BytesIO()
        fig.savefig(buf, format="png")
        plt.close(fig)
        img_bytes = buf.getvalue()

        return Image(data=img_bytes, format="png")
