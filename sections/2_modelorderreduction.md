:::::: collapse Model Order Reduction
## Model Order Reduction

**Model Order Reduction.**
From the previous step, we are able to generate trajectories of the system. We run simulations starting from various initial conditions, applying random inputs with added noise to excite the dynamics.

However, since we are using a finite element model (FEM), the number of degrees of freedom is quite high. For example, if we consider only the positions and velocities of all the mesh nodes in the Y-Z plane, we end up with around 3000 states. This is too large for efficient model identification and control. To address this, we apply a model order reduction technique to reduce the system's dimensionality to a more manageable size.

The idea is to collect all system measurements into a single data matrix. In our case, we retain only the positions of the nodes, where $p_i$ denotes the position of the FEM node $i$, and $L$ is the total number of measurements. The matrix $M$ is defined as:

$$M=\begin{bmatrix} p_1 & \ldots & p_L \end{bmatrix}$$

We then apply *Singular Value Decomposition* (SVD) to this matrix. SVD allows us to decompose $M$ into three matrices:

$$M = U \Sigma V^\top$$

where $U$ contains the left singular vectors (spatial modes), $\Sigma$ is a diagonal matrix of singular values (related to the importance of each mode), and $V^\top$ contains the right singular vectors (temporal information).

The columns of $U$ represent spatial modes sorted by importance. These are linear combinations of node positions that best capture the variance in the data. We then construct a reduction matrix $T$ using the first $r$ dominant modes:

$$T=\begin{bmatrix} u_1 & \ldots & u_r \end{bmatrix}$$

where $u_i$ are the first $r$ columns of $U$.  The reduced position is obtained by projecting the full position vector pp onto this subspace:

$$p_{red} = T^{\top} p$$

The quality of the approximation can be measured using the retained energy (information), computed as the ratio of the sum of the first $r$ singular values to the total sum:

$$\text{error} = 1 - \displaystyle\frac{\sum_{i=1}^{r} \sigma_i}{\sum_{i=1}^{L} \sigma_i}$$

where $\sigma_i$ are the eigenvalues from $\Sigma$.

To preserve the relationship between positions and velocities in the state vector, we apply the same transformation to the velocities. The final reduction matrix and the mapping from the full state $x$ (positions and velocities) to the reduced state $x_{red}$ is:

$$R = \begin{bmatrix} T & 0 \\ 0 & T \end{bmatrix} \Rightarrow x_{red} = R^{\top} x$$

::::: exercise

**Exercise 2:**

Check the following file to do the tasks listed below.:
#open-button("assets/labs/EmioLabs_Compliant/scripts/reduction.py")

Given the data generated in the previous step, you will:
1. Compute the SVD of the matrix $M$.
2. Compute the reduction error for different orders.

#runsofa-button("assets/labs/EmioLabs_Compliant/scripts/reduction.py" "--mode" "0")

3. Select the order of the reduction $r$ to compute the transformation matrix $T$.:
:::: select order
::: option 1
::: option 2
::: option 3
::: option 4
::: option 5
::: option 6
::: option 7
::: option 8
::: option 9
::: option 10
::::

#runsofa-button("assets/labs/EmioLabs_Compliant/scripts/reduction.py" "--mode" "1" "--order" "order")
:::::
::::::
