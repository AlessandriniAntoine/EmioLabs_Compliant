::::: collapse Linear Model Identification

## Linear Model Identification

**Linear Model Identification.**
From the previous step, we have reduced the dimensionality of the state vector, making it suitable for control and model identification. We now aim to identify a discrete-time linear state-space model of the system.

A linear state-space model is defined by the state matrix $A$, the input matrix $B$, and the output matrix $C$, following the equations::

$$\begin{array}{rcl} x(k+1) &=& Ax(k) + Bu(k) \\ y(k) &=& Cx(k) \end{array}.$$

For simplicity, we will use $x$ to denote $x(k)$ and $x^{+}$ for $x(k+1)$.

By stacking all collected measurements into matrices, the system dynamics can be compactly written as:

$$ \begin{bmatrix} x_1^+ & \ldots & x_L^+ \end{bmatrix} =  \begin{bmatrix} A & B \end{bmatrix} \begin{bmatrix} x_1 & \ldots & x_L \\ u_1 & \ldots & u_L \end{bmatrix}$$

The matrices $A$ and $B$ can therefore be identified by solving a least-squares problem, using the Moore–Penrose pseudo-inverse. Specifically:

$$\begin{bmatrix} A & B \end{bmatrix} = \begin{bmatrix} x_{1}^{+} & \ldots & x_{L}^{+}\end{bmatrix} \begin{bmatrix} x_{1} & \ldots & x_{L} \\ u_{1} & \ldots & u_{L} \end{bmatrix}^{\dagger} $$

where $*^\dagger$ is the pseudo inverse of matrix.

Similarly, the output matrix $C$ can be identified. Since the output corresponds to marker positions, which depend only on the position components $p$ of the state vector (and not the velocities), we restrict the output identification to the position subspace.

Let $C_p$ be the matrix mapping from the reduced positions to the measured outputs. It can be obtained by solving the following least-squares problem:

$$C_p = \begin{bmatrix} y_{1} & \ldots & y_{L}\end{bmatrix} \begin{bmatrix} p_{1} & \ldots & p_{L}\end{bmatrix}^{\dagger}$$

where $y_i$ is the measured output (e.g., marker position) and $p_i$ is the corresponding position extracted from the full state vector.

Since the full reduced state vector is composed of both velocities and positions (i.e., $x^\top=\begin{pmatrix} v^\top & p^\top\end{pmatrix}$), the final output matrix $C$, which maps the full reduced state to the output, is given by:

$$C = \begin{bmatrix} 0 & C_p \end{bmatrix}$$

This structure reflects the fact that the output depends solely on the position part of the state.

:::: exercise

**Exercise 3:**
Check the following file to do the tasks listed below.:
#open-button("assets/labs/EmioLabs_Compliant/scripts/identification.py")

Given the reduction matrix that you generated on the previous step, you will:
1. Reduce the data.
2. Compute the state matrix and input matrix.
3. Compute the output matrix.
4. Simulate the model given the same input as the system and compare the results.
#runsofa-button("assets/labs/EmioLabs_Compliant/scripts/identification.py" "--order" "order")
::::
:::::
