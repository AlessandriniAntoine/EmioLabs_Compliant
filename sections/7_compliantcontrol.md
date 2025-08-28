::::::: collapse Compliant Control

## Compliant Control

:::::: exercise
Select the parameters for the reference system.
::::: group-grid {style="grid-template-rows:repeat(2, 0fr);"}
**Mass**
#input("mass")

**Damping**
#input("damping")

**Stiffness**
#input("stiffness")
:::::

#runsofa-button("assets/labs/EmioLabs_Compliant/lab_compliant.py" "--controller" "compliant" "--motorCutoffFreq" "cutoffFreq" "--motorInit" "motorInit" "--motorMin" "motorMin" "--motorMax" "motorMax" "--mass" "mass" "--damping" "damping" "--stiffness" "stiffness" "--order" "order" "--useObserver" "1")
::::::

:::::::
