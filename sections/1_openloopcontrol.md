::::::: collapse Open Loop Control
## Open Loop Control

**Open Loop Control.**
The first step consists in using open-loop control within the simulation environment to generate data required for the identification of a linear model of the system. It is crucial that the simulation accurately captures the system’s dynamics, and that an appropriate time step has been chosen beforehand.

The motor used in the simulation is subject to physical constraints, including a maximum speed of 75 revolutions per minute (rpm), which must be strictly enforced. Additionally, the initial position of the motor must be specified, along with the corresponding minimum and maximum angles that can be reached from this starting point.

To avoid high-frequency oscillations in motor position, a low-pass filter is applied, defined by a given cutoff frequency $f_c$. The transfer function of the filter is given by:
$$H(s) = \frac{1}{\tau p + 1}$$
where $\tau = \frac{1}{2\pi f_c}$ is the time constant of the filter. This filter smooths the input commands to the motor, preventing abrupt changes that could lead to unrealistic behavior in the simulation. This filter can be implemented in discrete time using Euler's implicit method, resulting in the following difference equation:
$$ s(k+1) = \frac{dt}{\tau + dt} s(k) + (1 - \frac{dt}{\tau + dt}) u(k) $$
where $s(k)$ is the filtered signal at time step $k$, $u(k)$ is the signal at time step $k$, and $dt$ is the simulation time step.

To excite the system in a way that is informative for model identification, it is recommended to apply inputs drawn from a normal distribution, resulting in a pseudo-random motion of the motor. To further enrich the data, random noise is also added to the input commands.

The final open loop control structure is as follows:
|  ![](assets/data/images/labCompliant-openloop-structure.png)   |
|:------------------------------------------------:|
| **Open loop control structure** |


:::::: exercise
**Exercise 1:**

Check the following file to do the tasks listed below.:
#open-button("assets/labs/EmioLabs_Compliant/scripts/baseController.py")

 1. Implement the function `filter` to apply the low-pass filter to a signal.

2. Set up the scene with the following parameters and try to simulate the robot with the open loop.

::::: group-grid {style="grid-template-rows:repeat(5, 0fr);"}
**Motor**
Init, Min, Max (rad)
#input("motorInit")

#input("motorMin")

#input("motorMax")

* * *
**Cutoff frequency (Hz)**
#input("cutoffFreq")

:::::

#runsofa-button("assets/labs/EmioLabs_Compliant/lab_compliant.py" "--controller" "openloop" "--motorCutoffFreq" "cutoffFreq" "--motorInit" "motorInit" "--motorMin" "motorMin" "--motorMax" "motorMax")
::::::

:::::::
