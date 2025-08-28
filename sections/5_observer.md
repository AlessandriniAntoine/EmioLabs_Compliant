::::: collapse Observer Design

## Observer Design

**Observer Design.**
The control law presented in the previous section relies on access to the full system state.

In simulation, it is possible to measure the full state, reduce it, and use it directly for control. However, in practice—on a physical robot—this is not feasible. Only a limited set of outputs can be measured. Therefore, we need a method to estimate the full state from partial measurements.

To do this, we implement a Luenberger observer based on measurements of the marker positions in the Y-Z plane. The observer dynamics are given by:
$$\hat{x}^+ = A\hat{x} + Bu + L(y - \hat{y})$$
where $\hat{x}$ is the estimated state, $y$ is the measured output, and $\hat{y}=C\hat{x}$ is the predicted output from the estimated state.

The block diagram of the observer is shown below:
|  ![](assets/data/images/labCompliant-observer-structure.png)   |
|:------------------------------------------------:|
| **Observer structure** |

To analyze the convergence of the observer, we define the estimation error $e=x-\hat{x}$. The error dynamics become:

$$\begin{array}{rcl} e^+ &=& x^+ - \hat{x}^+ \\&=& Ax + Bu - A\hat{x} - Bu - L(y - \hat{y}) \\ &=& A(x - \hat{x}) - LC(x - \hat{x}) \\ &=& (A - LC)(x - \hat{x}) \\ e^+ &=& (A - LC)e \end{array}$$

Thus, the error dynamics are governed by the matrix $A-LC$, and the observer is stable if this matrix is asymptotically stable (i.e., all eigenvalues inside the unit circle).

The observer gain $L$ can be computed using the same techniques as the feedback gain $K$:
1. *Pole placement* (if system is controlable).
2. *Linear quadratic regulation* (LQR).
3. *Linear matrix inequalities* (LMI).

::: highlight
#icon("warning") **Warning:** As seen in the error dynamics, there is a duality between the closed-loop control system and the observer. In fact, the observer design is the transpose problem of state feedback design. This means that, when using functions originally intended for control synthesis, the following substitution holds:
$$(A, B, K) \rightarrow (A^\top, C^\top, L^\top)$$.
So you can compute $L$ using the same methods and tools as for $K$, by transposing the appropriate matrices.
:::

:::: exercise

**Exercise 5:**

Check the following file to do the tasks listed below.:
#open-button("assets/labs/EmioLabs_Compliant/scripts/observer.py")

1. Implement one the method to compute the observer gain $L$
#runsofa-button("assets/labs/EmioLabs_Compliant/scripts/observer.py" "--order" "order")
2. Test the observer on Sofa to see how it can estimate the state of the system.
::::
:::::
