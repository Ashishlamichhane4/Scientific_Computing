# Introduction to Matrices

## 1. What Is a Matrix?

A **matrix** is a rectangular arrangement of numbers organized into rows and columns.

For example, consider the matrix \(A\):

$$
A =
\begin{bmatrix}
2 & 4 & 6 \\
1 & 3 & 5
\end{bmatrix}
$$

This matrix has:

- **2 rows** — horizontal lines of numbers.
- **3 columns** — vertical lines of numbers.
- **Shape:** \(2 \times 3\), read as "2 by 3".

The shape tells us the number of rows first and the number of columns second.

## 2. Rows and Columns

Consider the matrix:

$$
A =
\begin{bmatrix}
2 & 4 & 6 \\
1 & 3 & 5
\end{bmatrix}
$$

### Rows

The first row is:

$$
[2,\ 4,\ 6]
$$

The second row is:

$$
[1,\ 3,\ 5]
$$

### Columns

The first column is:

$$
\begin{bmatrix}
2 \\
1
\end{bmatrix}
$$

The second column is:

$$
\begin{bmatrix}
4 \\
3
\end{bmatrix}
$$

The third column is:

$$
\begin{bmatrix}
6 \\
5
\end{bmatrix}
$$

**Remember:** Rows go horizontally; columns go vertically.

## 3. Why Do We Use Matrices?

Imagine collecting information about three students:

| Student | Hours studied | Exam score |
|---|---:|---:|
| A | 1 | 50 |
| B | 2 | 65 |
| C | 3 | 80 |

We can represent the numerical information as a matrix \(X\):

$$
X =
\begin{bmatrix}
1 & 50 \\
2 & 65 \\
3 & 80
\end{bmatrix}
$$

Here:

- Each row represents one student.
- The first column represents hours studied.
- The second column represents exam scores.
- The matrix has shape \(3 \times 2\).

Matrices allow us to store and calculate with many values together. They are widely used in data science, machine learning, computer graphics, and scientific computing.

## 4. Scalar, Vector, and Matrix

### Scalar

A **scalar** is a single number.

Example:

$$
a = 5
$$

### Vector

A **vector** is an ordered list of numbers.

A column vector looks like this:

$$
v =
\begin{bmatrix}
2 \\
4 \\
6
\end{bmatrix}
$$

This vector has three elements.

### Matrix

A matrix contains numbers arranged in rows and columns.

For example:

$$
B =
\begin{bmatrix}
2 & 4 \\
1 & 3 \\
5 & 6
\end{bmatrix}
$$

This matrix has 3 rows and 2 columns, so its shape is \(3 \times 2\).

## 5. Creating a Matrix in Python Using NumPy

In data science, we often create matrices using the NumPy library.

```python
import numpy as np

A = np.array([
    [2, 4, 6],
    [1, 3, 5]
])

print(A)
print("Shape:", A.shape)
```

Output:

```text
[[2 4 6]
 [1 3 5]]
Shape: (2, 3)
```

### Understanding the Code

- `import numpy as np` imports NumPy.
- `np.array()` creates an array containing our numbers.
- Each inner list represents one row.
- `A.shape` tells us the number of rows and columns.

In this example, `A.shape` returns `(2, 3)`, meaning 2 rows and 3 columns.

## 6. Practice Question

Consider the following matrix:

$$
C =
\begin{bmatrix}
1 & 2 \\
3 & 4 \\
5 & 6
\end{bmatrix}
$$

Answer these questions:

1. How many rows does the matrix have?
2. How many columns does it have?
3. What is its shape?
4. What is the second row?
5. What is the first column?

### Answers

1. Number of rows: 3
2. Number of columns: 2
3. Shape: \(3 \times 2\)
4. Second row: \([3,\ 4]\)
5. First column: \(\begin{bmatrix}1 \\ 3 \\ 5\end{bmatrix}\)

## Key Takeaway

A matrix is a rectangular arrangement of numbers in rows and columns. Its shape is written as `(number of rows, number of columns)` in NumPy.

---

# Matrices for Data Science and Computer Science

These notes introduce matrix operations used in Python, NumPy, PyTorch, linear algebra, and machine learning.

## 1. Matrix Multiplication

Matrix multiplication is used in linear regression, neural networks, computer graphics, and scientific computing.

Consider two matrices:

$$
A = \begin{bmatrix}1 & 2 \\ 3 & 4\end{bmatrix},
\qquad
B = \begin{bmatrix}5 & 6 \\ 7 & 8\end{bmatrix}
$$

To calculate \(AB\), multiply each row of \(A\) by each column of \(B\), then add the products:

$$
AB =
\begin{bmatrix}
1(5)+2(7) & 1(6)+2(8) \\
3(5)+4(7) & 3(6)+4(8)
\end{bmatrix}
=
\begin{bmatrix}19 & 22 \\ 43 & 50\end{bmatrix}
$$

For example, the top-left element is \(1(5)+2(7)=19\).

### The Shape Rule

For matrix multiplication, the number of columns in the first matrix must equal the number of rows in the second matrix.

If \(A\) has shape \(m \times n\) and \(B\) has shape \(n \times p\), then \(AB\) has shape \(m \times p\).

For example:

$$
(2 \times 3)(3 \times 2) \longrightarrow (2 \times 2)
$$

The inner dimensions must match. The outer dimensions give the result's shape.

### NumPy Example

```python
import numpy as np

A = np.array([
    [1, 2],
    [3, 4]
])

B = np.array([
    [5, 6],
    [7, 8]
])

C = A @ B

print(C)
print("Shape:", C.shape)
```

Output:

```text
[[19 22]
 [43 50]]
Shape: (2, 2)
```

You can also use `np.matmul(A, B)`.

### `*` versus `@`

In NumPy, `*` multiplies corresponding elements, while `@` performs matrix multiplication.

```python
print(A * B)  # Element-by-element multiplication
print(A @ B)  # Matrix multiplication
```

Element-by-element result:

$$
A * B = \begin{bmatrix}5 & 12 \\ 21 & 32\end{bmatrix}
$$

Matrix multiplication result:

$$
A @ B = \begin{bmatrix}19 & 22 \\ 43 & 50\end{bmatrix}
$$

## 2. Matrix Multiplication Is Not Commutative

For ordinary numbers, \(2 \times 3 = 3 \times 2\). But matrix multiplication generally depends on the order:

$$
AB \ne BA
$$

For the matrices above:

$$
AB = \begin{bmatrix}19 & 22 \\ 43 & 50\end{bmatrix}
$$

but

$$
BA = \begin{bmatrix}23 & 34 \\ 31 & 46\end{bmatrix}
$$

Both products are valid, but their values differ. In machine learning, the order of operations determines how data and model parameters are combined.

## 3. Transpose of a Matrix

The transpose switches rows and columns. It is written as \(A^T\).

For example:

$$
A = \begin{bmatrix}1 & 2 & 3 \\ 4 & 5 & 6\end{bmatrix}
$$

Then:

$$
A^T = \begin{bmatrix}1 & 4 \\ 2 & 5 \\ 3 & 6\end{bmatrix}
$$

The original shape is \(2 \times 3\) and the transposed shape is \(3 \times 2\).

```python
import numpy as np

A = np.array([
    [1, 2, 3],
    [4, 5, 6]
])

print(A.T)
print(A.T.shape)
```

Output:

```text
[[1 4]
 [2 5]
 [3 6]]
(3, 2)
```

Transposes appear in linear regression, gradient calculations, and neural networks. For example, you may see `X.T @ error`.

## 4. Vectors and the Dot Product

A vector is an ordered list of numbers. The dot product multiplies corresponding elements and adds the results.

If:

$$
a = [1, 2, 3], \qquad b = [4, 5, 6]
$$

then:

$$
a \cdot b = 1(4)+2(5)+3(6)=32
$$

In NumPy:

```python
import numpy as np

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

print(a @ b)
```

Output:

```text
32
```

The dot product is essential for predictions in linear regression and neural networks. A simple prediction may be:

$$
\hat{y} = w_1x_1 + w_2x_2 + b
$$

The weighted sum is a dot product, followed by adding the bias \(b\).

## 5. Identity Matrix

An identity matrix acts like the number 1 in matrix multiplication: multiplying a compatible matrix by it leaves the matrix unchanged.

For example:

$$
I = \begin{bmatrix}1 & 0 \\ 0 & 1\end{bmatrix}
$$

For a compatible matrix \(A\):

$$
AI = IA = A
$$

Create a \(3 \times 3\) identity matrix with NumPy:

```python
import numpy as np

I = np.eye(3)
print(I)
```

Output:

```text
[[1. 0. 0.]
 [0. 1. 0.]
 [0. 0. 1.]]
```

Identity matrices appear in linear algebra, solving systems of equations, and machine-learning algorithms.

## 6. Inverse of a Matrix

The inverse of a square matrix \(A\) is written as \(A^{-1}\). When the inverse exists:

$$
AA^{-1} = A^{-1}A = I
$$

For example:

$$
A = \begin{bmatrix}2 & 0 \\ 0 & 4\end{bmatrix}
\quad \Rightarrow \quad
A^{-1} = \begin{bmatrix}1/2 & 0 \\ 0 & 1/4\end{bmatrix}
$$

In NumPy:

```python
import numpy as np

A = np.array([
    [2.0, 0.0],
    [0.0, 4.0]
])

A_inv = np.linalg.inv(A)

print(A_inv)
print(A @ A_inv)
```

Output:

```text
[[0.5  0.  ]
 [0.   0.25]]
[[1. 0.]
 [0. 1.]]
```

The last result should be approximately the identity matrix because computer calculations can involve small rounding errors.

Not every square matrix has an inverse. A matrix must be nonsingular (have full rank) for an ordinary inverse to exist. In numerical machine learning, solving a linear system directly is often preferable to explicitly calculating an inverse.

## 7. Determinant

The determinant is a number associated with a square matrix. For a \(2 \times 2\) matrix:

$$
A = \begin{bmatrix}a & b \\ c & d\end{bmatrix}
$$

the determinant is:

$$
\det(A) = ad - bc
$$

For example:

$$
\det\begin{bmatrix}2 & 1 \\ 3 & 4\end{bmatrix}
= 2(4)-1(3)=5
$$

If the determinant of a square matrix is zero, it has no ordinary inverse.

```python
import numpy as np

A = np.array([[2, 1], [3, 4]])
print(np.linalg.det(A))  # approximately 5.0 (may show a tiny rounding error)
```

## 8. Rank

The rank tells you how many linearly independent rows or columns a matrix has. It helps identify redundant information and whether a system of equations has enough independent constraints.

```python
import numpy as np

A = np.array([
    [1, 2],
    [2, 4]
])

print(np.linalg.matrix_rank(A))
```

Output:

```text
1
```

The second row is twice the first row, so the rows are not independent. The matrix has rank 1.

## 9. Eigenvalues and Eigenvectors

For a square matrix \(A\), an eigenvector \(v\) and its eigenvalue \(\lambda\) satisfy:

$$
Av = \lambda v
$$

Multiplying the eigenvector by the matrix scales it by the eigenvalue, without changing its direction (except that a negative eigenvalue reverses the direction).

Eigenvalues and eigenvectors are used in principal component analysis (PCA), dimensionality reduction, and scientific computing.

```python
import numpy as np

A = np.array([
    [2, 0],
    [0, 3]
])

eigenvalues, eigenvectors = np.linalg.eig(A)

print("Eigenvalues:", eigenvalues)
print("Eigenvectors:\n", eigenvectors)
```

Output:

```text
Eigenvalues: [2. 3.]
Eigenvectors:
 [[1. 0.]
 [0. 1.]]
```

For this matrix, the eigenvalues are 2 and 3. You can learn the details after becoming comfortable with multiplication, transpose, and shapes.

## 10. Broadcasting in NumPy

Broadcasting allows NumPy to perform operations between compatible arrays of different shapes.

For example, adding a number to every element of a matrix:

```python
import numpy as np

A = np.array([
    [1, 2],
    [3, 4]
])

print(A + 10)
```

Output:

```text
[[11 12]
 [13 14]]
```

NumPy also allows adding a vector of length 2 to each row of a \(2 \times 2\) matrix:

```python
A = np.array([
    [1, 2],
    [3, 4]
])

v = np.array([10, 20])

print(A + v)
```

Output:

```text
[[11 22]
 [13 24]]
```

Broadcasting is useful for adding biases in neural networks, scaling features, and normalizing data.

## 11. Recommended Learning Order

For data science and deep learning, study these topics in this order:

1. **Shapes and indexing:** Understand rows, columns, `.shape`, and how to select elements.
2. **Matrix multiplication and dot products:** Learn the shape rule and how weighted sums produce predictions.
3. **Transpose and broadcasting:** These appear frequently in NumPy, PyTorch, and machine-learning formulas.
4. **Inverse, rank, and determinants:** Understand when systems can be solved and when information is redundant.
5. **Eigenvalues and eigenvectors:** Study these when you begin PCA and more advanced linear algebra.

## Practice Exercises

Try these without looking at the answers first.

### Exercise 1: Shape

A matrix has 3 rows and 4 columns. What is its shape?

### Exercise 2: Multiplication Compatibility

Can a matrix with shape \(3 \times 4\) be multiplied by a matrix with shape \(4 \times 2\)? If yes, what is the output shape?

### Exercise 3: Transpose

What is the shape of the transpose of a \(2 \times 5\) matrix?

### Exercise 4: Dot Product

Calculate:

$$
[2, 3] \cdot [4, 5]
$$

### Exercise 5: Python

Use NumPy to create the matrix below and print its shape:

$$
A = \begin{bmatrix}1 & 2 & 3 \\ 4 & 5 & 6\end{bmatrix}
$$

### Answers

1. \((3, 4)\)
2. Yes. The inner dimensions match, and the output shape is \((3, 2)\).
3. \((5, 2)\)
4. \(2(4)+3(5)=23\)
5. The shape is `(2, 3)`:

   ```python
   import numpy as np

   A = np.array([
       [1, 2, 3],
       [4, 5, 6]
   ])

   print(A.shape)  # (2, 3)
   ```

## Key Takeaway

Matrices are more than tables of numbers. Their shapes determine which operations are possible, and operations such as multiplication, transpose, and dot products are central to machine learning. Practise small examples by hand and then verify them in NumPy before moving on to PyTorch.



# Indexing, Slicing, Reshaping and Axis Operations in NumPy

These notes cover the array-handling skills you will use every day in NumPy, pandas and PyTorch. Most bugs in data science code are shape or indexing mistakes, so it is worth learning this well.

## 1. Indexing Elements

NumPy uses zero-based indexing: the first row is row `0`, the first column is column `0`. For a matrix, use `A[row, column]`.

$$
A =
\begin{bmatrix}
1 & 2 & 3 \\
4 & 5 & 6 \\
7 & 8 & 9
\end{bmatrix}
$$

```python
import numpy as np

A = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

print(A[0, 1])    # row 0, column 1
print(A[2, 2])    # row 2, column 2
print(A[-1, -1])  # negative indices count from the end
```

Output:

```text
2
9
9
```

**Remember:** `A[1][2]` also works, but `A[1, 2]` is faster and is the standard style.

## 2. Slicing Rows and Columns

A slice has the form `start:stop:step`. The `start` is included and the `stop` is **excluded**. A bare `:` means "everything along this dimension".

```python
print(A[0])        # first row
print(A[0, :])     # first row (same thing, more explicit)
print(A[:, 1])     # second column
print(A[:2, :2])   # top-left 2x2 block
print(A[1:, 1:])   # bottom-right 2x2 block
print(A[::2])      # every second row
```

Output:

```text
[1 2 3]
[1 2 3]
[2 5 8]
[[1 2]
 [4 5]]
[[5 6]
 [8 9]]
[[1 2 3]
 [7 8 9]]
```

Notice that `A[:, 1]` returns a 1D array with shape `(3,)`, not a column with shape `(3, 1)`. To keep the column shape, use a slice or a list:

```python
print(A[:, 1:2])    # shape (3, 1)
print(A[:, [1]])    # shape (3, 1)
```

Output (for both):

```text
[[2]
 [5]
 [8]]
```

## 3. Boolean Indexing (Masks)

A comparison creates a Boolean array of the same shape. Using it as an index selects the elements where the condition is `True`. The result is always a 1D array.

```python
print(A > 5)
print(A[A > 5])
print(A[A % 2 == 0])
```

Output:

```text
[[False False False]
 [False False  True]
 [ True  True  True]]
[6 7 8 9]
[2 4 6 8]
```

You can also assign through a mask:

```python
B = A.copy()
B[B > 8] = 0
print(B)
```

Output:

```text
[[1 2 3]
 [4 5 6]
 [7 8 0]]
```

In machine learning, masks are used to select samples, for example `X[y == 1]` gives all rows belonging to class 1.

## 4. Fancy Indexing

Passing lists of indices selects specific rows, columns or elements.

```python
print(A[[0, 2]])         # rows 0 and 2
print(A[:, [0, 2]])      # columns 0 and 2
print(A[[0, 2], [1, 2]]) # elements (0, 1) and (2, 2)
```

Output:

```text
[[1 2 3]
 [7 8 9]]
[[1 3]
 [4 6]
 [7 9]]
[2 9]
```

The last example pairs the indices: it returns `A[0, 1]` and `A[2, 2]`, not a 2x2 block.

## 5. Views versus Copies

Basic slices return a **view**: a window onto the same data. Changing the view changes the original.

```python
A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

row = A[0]
row[0] = 100
print(A[0])
```

Output:

```text
[100   2   3]
```

To get an independent array, use `.copy()`:

```python
row = A[0].copy()
row[0] = -1
print(A[0])   # unchanged
```

Output:

```text
[100   2   3]
```

Boolean and fancy indexing, by contrast, return copies.

## 6. Reshaping

`reshape` changes the shape without changing the data. The total number of elements must stay the same.

```python
a = np.arange(6)
print(a)
print(a.reshape(2, 3))
print(a.reshape(3, 2))
print(a.reshape(2, -1).shape)
```

Output:

```text
[0 1 2 3 4 5]
[[0 1 2]
 [3 4 5]]
[[0 1]
 [2 3]
 [4 5]]
(2, 3)
```

The `-1` means "work this dimension out for me". Here NumPy computes `6 / 2 = 3`.

NumPy fills the new shape row by row. To flatten a matrix back to 1D, use `flatten()` (always a copy) or `ravel()` (a view when possible).

```python
M = np.array([[1, 2, 3], [4, 5, 6]])
print(M.flatten())
```

Output:

```text
[1 2 3 4 5 6]
```

## 7. Row Vectors, Column Vectors and Adding Axes

A 1D array with shape `(3,)` is neither a row nor a column. When a library expects a 2D shape, you must be explicit:

```python
v = np.array([1, 2, 3])

print(v.shape)                    # (3,)
print(v.reshape(3, 1).shape)      # column vector
print(v[:, np.newaxis].shape)     # column vector
print(v[np.newaxis, :].shape)     # row vector
```

Output:

```text
(3,)
(3, 1)
(3, 1)
(1, 3)
```

This matters for matrix multiplication and broadcasting. For example, \((3, 1) @ (1, 3)\) gives a \(3 \times 3\) matrix (an outer product), while \((3,) @ (3,)\) gives a single number (a dot product).

## 8. Operations Along an Axis

Many functions (`sum`, `mean`, `max`, `min`, `std`, `argmax`) accept an `axis` argument.

$$
M =
\begin{bmatrix}
1 & 2 & 3 \\
4 & 5 & 6
\end{bmatrix}
$$

```python
M = np.array([[1, 2, 3], [4, 5, 6]])

print(M.sum())
print(M.sum(axis=0))
print(M.sum(axis=1))
print(M.mean(axis=0))
print(M.mean(axis=1))
print(M.max(axis=1))
print(M.argmax(axis=1))
```

Output:

```text
21
[5 7 9]
[ 6 15]
[2.5 3.5 4.5]
[2. 5.]
[3 6]
[2 2]
```

### The axis rule

**The axis you specify is the one that disappears.**

| Call | What it does | Result shape for a `(2, 3)` matrix |
|---|---|---|
| `M.sum(axis=0)` | collapses the rows, giving one value per column | `(3,)` |
| `M.sum(axis=1)` | collapses the columns, giving one value per row | `(2,)` |

### `keepdims=True`

Keeping the collapsed axis (with length 1) makes broadcasting work cleanly:

```python
row_sums = M.sum(axis=1, keepdims=True)
print(row_sums)
print(M / row_sums)   # each row now sums to 1
```

Output:

```text
[[ 6]
 [15]]
[[0.16666667 0.33333333 0.5       ]
 [0.26666667 0.33333333 0.4       ]]
```

## 9. Stacking and Joining Arrays

```python
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6]])
c = np.array([[10], [20]])

print(np.vstack([a, b]))   # stack vertically (add rows)
print(np.hstack([a, c]))   # stack horizontally (add columns)
```

Output:

```text
[[1 2]
 [3 4]
 [5 6]]
[[ 1  2 10]
 [ 3  4 20]]
```

`np.concatenate([a, b], axis=0)` does the same as `vstack`, and `axis=1` does the same as `hstack`. The arrays must match in every dimension except the one you join along.

## 10. How This Looks in Data Science

By convention, a dataset `X` has shape `(n_samples, n_features)`: each row is one example and each column is one feature.

```python
X = np.array([
    [1.0, 2.0],
    [3.0, 4.0],
    [5.0, 6.0]
])
y = np.array([0, 1, 1])

print(X[:, 0])            # first feature for all samples
print(X[y == 1])          # all samples with label 1
print(X[:2], X[2:])       # simple train/test split by position
```

Output:

```text
[1. 3. 5.]
[[3. 4.]
 [5. 6.]]
[[1. 2.]
 [3. 4.]] [[5. 6.]]
```

Two very common patterns:

```python
# Add a bias column of ones (used in linear regression)
X_bias = np.hstack([np.ones((X.shape[0], 1)), X])

# Center each feature (subtract the column mean)
X_centered = X - X.mean(axis=0)
```

## 11. Common Mistakes

1. **Forgetting that the stop index is excluded.** `A[0:2]` returns rows 0 and 1, not row 2.
2. **Expecting a column but getting a 1D array.** `A[:, 1]` has shape `(3,)`. Use `A[:, 1:2]` if you need `(3, 1)`.
3. **Changing the original through a view.** Use `.copy()` when you want to modify a slice independently.
4. **Using the wrong axis.** Check the result shape. If you wanted one value per column, the result should have as many entries as there are columns.
5. **Reshaping to an incompatible size.** `np.arange(6).reshape(4, 2)` fails because 6 elements cannot fill 8 slots.

**Debugging habit:** print `.shape` of every array whenever something looks wrong.

## Practice Exercises

Try these without looking at the answers first.

### Exercise 1: Indexing and slicing

Let `A = np.arange(16).reshape(4, 4)`. Write the result of:

1. `A[1, 2]`
2. `A[:, 0]`
3. `A[2:, 2:]`
4. `A[A > 12]`

### Exercise 2: Reshape

What is the shape of `np.arange(12).reshape(3, -1)`?

### Exercise 3: Axis

A matrix `M` has shape `(4, 5)`. What are the shapes of:

1. `M.sum(axis=0)`
2. `M.sum(axis=1)`
3. `M.sum(axis=1, keepdims=True)`

### Exercise 4: Adding an axis

`x = np.array([1, 2, 3])`. What are the shapes of `x` and `x[:, np.newaxis]`?

### Exercise 5: Centering data

Given `X = np.array([[1, 2], [3, 4], [5, 6]])`, write code that subtracts each column's mean from that column, and give the result.

### Answers

1. 
   1. `6`
   2. `[ 0  4  8 12]`
   3. A 2x2 matrix with rows `[10 11]` and `[14 15]`
   4. `[13 14 15]`
2. `(3, 4)`
3. 
   1. `(5,)`
   2. `(4,)`
   3. `(4, 1)`
4. `(3,)` and `(3, 1)`
5. Use `X - X.mean(axis=0)`. The column means are `[3, 4]`, so the result is:

   ```text
   [[-2. -2.]
    [ 0.  0.]
    [ 2.  2.]]
   ```

## Key Takeaway

Shape is the most important property of an array. Know the shape of every array you work with, remember that `axis` names the dimension that disappears, and use `.copy()` when you do not want to modify the original. Once indexing, reshaping and axis operations feel natural, matrix multiplication, broadcasting and PyTorch tensors become much easier.



