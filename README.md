# Scientific Computation MCP

[![smithery badge](https://smithery.ai/badge/@Aman-Amith-Shastry/scientific_computation_mcp)](https://smithery.ai/server/@Aman-Amith-Shastry/scientific_computation_mcp)

## Installation Guide

### Claude Desktop

Open Claude Desktop's configuration file (claude_desktop_config.json) and add the following:

- Mac/Linux: 
```json
{
  "mcpServers": {
    "numpy_mcp": {
      "command": "npx",
      "args": [
        "-y",
        "@smithery/cli@latest",
        "run",
        "@Aman-Amith-Shastry/scientific_computation_mcp",
        "--key",
        "<YOUR_SMITHERY_API_KEY>"
      ]
    }
  }
}
```

- Windows:
```json
{
  "mcpServers": {
    "numpy_mcp": {
      "command": "cmd",
      "args": [
        "/c",
        "npx",
        "-y",
        "@smithery/cli@latest",
        "run",
        "@Aman-Amith-Shastry/scientific_computation_mcp",
        "--key",
        "<YOUR_SMITHERY_API_KEY>"
      ]
    }
  }
}
```

Or alternatively, run the following command:
```commandline
npx -y @smithery/cli@latest install @Aman-Amith-Shastry/scientific_computation_mcp --client claude --key <YOUR_SMITHERY_API_KEY>
```

Restart Claude to load the server properly

### Cursor

If you prefer to access the server through Cursor instead, then run the following command:
```commandline
npx -y @smithery/cli@latest install @Aman-Amith-Shastry/scientific_computation_mcp --client cursor --key <YOUR_SMITHERY_API_KEY>
```

## Components of the Server

### Tools

#### Tensor storage
- ```create_tensor(shape: list[int], values: list[float], name: str)```: Creates a new tensor based on a given name, shape, and values, and adds it to the tensor store. For the purposes of this server, tensors are vectors and matrices.
- ```view_tensor(name: str)```: Display the contents of a tensor from the store .
- ```delete_tensor(name: str)```: Deletes a tensor based on its name in the tensor store.

#### Linear Algebra
- ```add_matrices(name_a: str, name_b: str)```: Adds two matrices with the provided names, if compatible.
- ```subtract_matrices(name_a: str, name_b: str)```: Subtracts two matrices with the provided names, if compatible.
- ```multiply_matrices(name_a: str, name_b: str)```: Multiplies two matrices with the provided names, if compatible.
- ```scale_matrix(name: str, scale_factor: float, in_place: bool = True)```: Scales a matrix of the provided name by a certain factor, in-place by default.
- ```matrix_inverse(name: str)```: Computes the inverse of the matrix with the provided name.
- ```transpose(name: str)```: Computes the transpose of the inverse of the matrix of the provided name.
- ```determinant(name: str)```: Computes the determinant of the matrix of the provided name.
- ```rank(name: str)```: Computes the rank (number of pivots) of the matrix of the provided name.
- ```compute_eigen(name: str)```: Calculates the eigenvectors and eigenvalues of the matrix of the provided name.
- ```qr_decompose(name: str)```: Computes the QR factorization of the matrix of the provided name. The columns of Q are an orthonormal basis for the image of the matrix, and R is upper triangular.
- ```svd_decompose(name: str)```: Computes the Singular Value Decomposition of the matrix of the provided name.
- ```find_orthonormal_basis(name: str)```: Finds an orthonormal basis for the matrix of the provided name. The vectors returned are all pair-wise orthogonal and are of unit length.
- ```change_basis(name: str, new_basis: list[list[float]])```: Computes the matrix of the provided name in the new basis.

#### Vector Calculus
- ```vector_project(name: str, new_vector: list[float])```: Projects a vector in the tensor store to the specified vector in the same vector space
- ```vector_dot_product(name_a: str, name_b: str)```: Computes the dot product of two vectors in the tensor stores based on their provided names.
- ```vector_cross_product(name_a: str, name_b: str)```: Computes the cross product of two vectors in the tensor stores based on their provided names.
- ```gradient(f_str: str)```: Computes the gradient of a multivariable function based on the input function. Example call: ```gradient("x^2 + 2xyz + zy^3")```. Do NOT include the function name (like f(x, y, z) = ...`).
- ```curl(f_str: str)```: Computes the curl of a vector field based on the input vector field. The input string must be formatted as a python list. Example call: ```curl("[3xy, 2z^4, 2y]"")```.
- ```divergence(f_str: str)```Computes the curl of a vector field based on the input vector field. The input string must be formatted as a python list. Example call: ```curl("[3xy, 2z^4, 2y]"")```.