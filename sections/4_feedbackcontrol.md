::::: collapse State Feedback Control

## State Feedback Control

**State Feedback Control.**
Based on the identified linear model, we can implement a state feedback controller with feedforward compensation, defined as
$$ u = Gr - Kx $$
where $r$ is the reference signal that the controlled output should track.

The block diagram of the state feedback control system is shown below:
|  ![](assets/data/images/labCompliant-feedback-structure.png)   |
|:------------------------------------------------:|
| **State feedback control structure** |

This results in the following closed-loop dynamics:
$$ x^+ = (A-BK)x + BGr$$

The gain matrix $K$ can be computed using several techniques:
1. *Pole placement* (if system is controlable).
2. *Linear quadratic regulation* (LQR).
3. *Linear matrix inequalities* (LMI).

The first two methods can be implemented using the [control library](https://python-control.readthedocs.io/en/0.10.2/) in Python. The LMI approach, on the other hand, provides guarantees on performance and robustness and can be solved using convex optimization tools such as [cvxpy](https://www.cvxpy.org/index.html).

Once the matrix $K$ is known, the feedforward gain $G$ must be chosen to ensure that the controlled output $y_c$ converges to the reference $r$ in steady state.
::: highlight
#icon("warning") **Warning:** Here you need to be careful between two notions:
1. *The ouput of the system* $y$: what you can really measure. In our case it is the positions of all markers.
2. *The controlled output* $y_c$: this is the specific output you want to regulate or track. It must have the same dimension as the input $u$. Therefore, we define a new output matric for control $C_{c}$, which maps the state to the controlled output.
:::

At steady state, $x^+=x$. Plugging this into the closed-loop dynamics yields:

$$(I - A + BK)x = BGr \Leftrightarrow x = (I - A + BK)^{-1}BGr$$

We now substitute this into the controlled output equation:

$$y_{c} = C_{c}x = C_{c}(I-A+BK)^{-1}BGr$$

To ensure perfect tracking (i.e., $y_c=r$), we choose the feedforward matrix $G$ as:

$$G = (C_{ctr} (I-A+BK)^{-1}B)^{-1}$$

:::: exercise

**Exercise 4:**
Check the following file to do the tasks listed below.:
#open-button("assets/labs/EmioLabs_Compliant/scripts/controller.py")

Given the linear model that you generated on the previous step, you will:
1. Use one of the methods given earlier to compute the state feedback matrix $K$.
2. Compute the feedforward matrix $G$
#runsofa-button("assets/labs/EmioLabs_Compliant/scripts/controller.py" "--order" "order")
3. Test on Sofa to see how you can control the robot using this control law.
#runsofa-button("assets/labs/EmioLabs_Compliant/lab_compliant.py" "--controller" "closedloop" "--motorCutoffFreq" "cutoffFreq" "--motorInit" "motorInit" "--motorMin" "motorMin" "--motorMax" "motorMax" "--order" "order" "--useObserver" "0")
::::
:::::
