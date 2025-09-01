# Lab Compliant

::: highlight
##### Overview

This lab is dedicated to linear control. Its goals are to make you understand:

1. How to **obtain data** from a SOFA scene that corresponds to the real robot.
2. How to **reduce** the model order to a manageable size.
3. **Identify** a linear model of the dynamic of the reduced state.
4. The use of this model to **estimate** the state.
5. The use ot this model to **control** the robot.

This lab relies on linear algebra and control theory. It is recommended to check the [numpy linear algebra](https://numpy.org/doc/stable/reference/routines.linalg.html) and [control](https://python-control.readthedocs.io/en/0.10.2/) libraries documentation for more details on the implementation.
:::

::: collapse {open} Set up Emio  for the Lab
## Set up Emio

In this lab session, we will use only the following configuration: Emio with one <span style="color:rgba(200, 200, 0, 1);">*yellow leg*</span>,
the <span style="color:grey">*grey mass*</span>, and two <span style="color:green">*green markers*</span>. We also use the **FEM** modeling of the leg to simulate the dynamics of the system.

![](assets/data/images/labCompliant-setup-emio.png){width=75% .center}
:::

#include(assets/labs/EmioLabs_Compliant/sections/6_closedloopcontrol.md)

<!--#include(assets/labs/EmioLabs_Compliant/sections/1_openloopcontrol.md)
#include(assets/labs/EmioLabs_Compliant/sections/2_modelorderreduction.md)
#include(assets/labs/EmioLabs_Compliant/sections/3_linearidentification.md)
#include(assets/labs/EmioLabs_Compliant/sections/4_feedbackcontrol.md)
#include(assets/labs/EmioLabs_Compliant/sections/5_observer.md)-->
<!--#include(assets/labs/EmioLabs_Compliant/sections/7_compliantcontrol.md)-->
