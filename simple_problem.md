## Matrix Polynomial Problem

Find all positive integers $n$ for which there exists an $n \times n$ matrix $A$ with **real entries** that satisfies the equation:

$$A^2 + A + 6I = \mathbf{0}$$

Where:
* $I$ is the $n \times n$ identity matrix.
* $\mathbf{0}$ is the $n \times n$ zero matrix.

---

### Problem Description

This is a problem in **Linear Algebra** that explores the properties of matrices satisfying a given polynomial equation. 💡 You are not asked to find the specific matrices, but rather to determine for which **dimensions** ($n$) such a real matrix can exist.

The solution involves connecting the properties of the matrix $A$ to the roots of the corresponding scalar polynomial $p(x) = x^2 + x + 6$. A key strategy is to consider the **eigenvalues** of the matrix $A$. If $\lambda$ is an eigenvalue of $A$, what equation must $\lambda$ itself satisfy? Analyzing the nature of these eigenvalues provides a powerful constraint on the matrix $A$ and its dimension $n$.