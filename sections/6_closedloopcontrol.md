::::: collapse Closed Loop Control

## Closed Loop Control

**Closed Loop Control.** In simulation, we have access to the full (reduced) state of the system. However, to better reflect realistic conditions, we simulate the control using an observer rather than the true state.

The control law becomes:

$$u = Gr - K\hat{x}$$

where $\hat{x}$ is the estimated state from the observer, and $r$ is the reference for the controlled output.

The overall structure of the closed-loop control system is shown below:
|  ![](assets/data/images/labCompliant-feedback-observer-structure.png)   |
|:------------------------------------------------:|
| **State feedback with observer structure** |

The simulation loop proceeds as follows:

- Initialize the estimated state $\hat{x}$.
- At each time step:
  1. Measure the output y and estimate output $\hat{y} = C\hat{x}$.
  2. Compute the control input $u = Gr - K\hat{x}$.
  3. Update the observer using $\hat{x}^+ = A\hat{x} + Bu + L(y−\hat{y})$.


This setup will first be implemented in SOFA, which provides the system dynamics and sensor measurements. It enables evaluation of the full closed-loop system without requiring physical hardware, while still respecting realistic sensing and estimation constraints.

#runsofa-button("assets/labs/EmioLabs_Compliant/lab_compliant.py" "--controller" "closedloop" "--motorCutoffFreq" "cutoffFreq" "--motorInit" "motorInit" "--motorMin" "motorMin" "--motorMax" "motorMax" "--order" "order" "--useObserver" "1")

In a second step, the same control law can be deployed on the real robot. The observer will estimate the state based on measured outputs (e.g., marker positions), and the computed control input $u$ will be applied to the motor.

```bash
$ python scripts/hardware.py --motorCutoffFreq cutoffFreq --motorInit motorInit --motorMin motorMin --motorMax motorMax --order order
```

::: highlight
#icon("warning") Insight:
This architecture enables real-time control using only partial measurements, and bridges the gap between model-based design and physical implementation. The exact same controller and observer gains designed in simulation can be reused on the real system, as long as the model accurately captures the dynamics.
:::
:::::
