::::::: collapse {open} Closed Loop Control

## Closed Loop Control

**Closed Loop Control.**

:::::: highlight
1. Generate Open Loop data
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

:::::: highlight
2. Compute the reduction error for different orders.
#runsofa-button("assets/labs/EmioLabs_Compliant/scripts/reduction.py" "--mode" "0")

3. Select the order of the reduction $r$.
::::: select order
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
:::::
#runsofa-button("assets/labs/EmioLabs_Compliant/scripts/reduction.py" "--mode" "1" "--order" "order")
::::::

:::::: highlight
4. Identify the model
#runsofa-button("assets/labs/EmioLabs_Compliant/scripts/identification.py" "--order" "order")
::::::

:::::: highlight
5. Select the controller type:
#open-button("assets/labs/EmioLabs_Compliant/scripts/controller.py")
::::: group-grid {style="grid-template-rows:repeat(2, 0fr);"}
**Controller type**
:::: select controller_type
::: option state_feedback
::: option state_feedback_integral
::::
:::::
#runsofa-button("assets/labs/EmioLabs_Compliant/scripts/controller.py" "--order" "order" "--controller_type" "controller_type")
::::::

:::::: highlight
6. Select the observer type:
#open-button("assets/labs/EmioLabs_Compliant/scripts/observer.py")
::::: group-grid {style="grid-template-rows:repeat(2, 0fr);"}
**Observer type**
:::: select observer_type
::: option default
::: option perturbation
::: option force
::: option perturbation_force
::::
:::::
#runsofa-button("assets/labs/EmioLabs_Compliant/scripts/observer.py" "--order" "order" "--observer_type" "observer_type")
::::::

:::::: highlight
7. Try on Sofa
#runsofa-button("assets/labs/EmioLabs_Compliant/lab_compliant.py" "--controller" "closedloop" "--motorCutoffFreq" "cutoffFreq" "--motorInit" "motorInit" "--motorMin" "motorMin" "--motorMax" "motorMax" "--order" "order" "--controller_type" "controller_type" "--observer_type" "observer_type")
::::::

:::::: highlight
8. Try on compliant control
::::: group-grid {style="grid-template-rows:repeat(2, 0fr);"}
**Mass**
#input("mass")

**Damping**
#input("damping")

**Stiffness**
#input("stiffness")
:::::
#runsofa-button("assets/labs/EmioLabs_Compliant/lab_compliant.py" "--controller" "compliant" "--motorCutoffFreq" "cutoffFreq" "--motorInit" "motorInit" "--motorMin" "motorMin" "--motorMax" "motorMax" "--order" "order" "--controller_type" "controller_type" "--observer_type" "observer_type" "--mass" "mass" "--damping" "damping" "--stiffness" "stiffness")
::::::

:::::::
