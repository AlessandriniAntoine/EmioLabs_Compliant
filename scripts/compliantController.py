from .closedLoopController import *
from .utils import Observer


ControlMode = {"Open Loop": False, "State Feedback": True}


class CompliantController(ClosedLoopController):
    def __init__(self, leg, motor, markers, load, motorInit, motorMin, motorMax, cutoffFreq, order, controller_type, observer_type, mass, damping, stiffness):
        super().__init__(leg, motor, markers, load, motorInit, motorMin, motorMax, cutoffFreq, order, controller_type, "default")

        self.setup_compliant_variables(mass, damping, stiffness, order)
        self.setup_compliant_gui()

    def setup_compliant_variables(self, mass, damping, stiffness, order):
        self.mass = mass
        self.damping = damping
        self.stiffness = stiffness
        self.mass_exponant=int(np.floor(np.log10(abs(self.mass))))-1
        self.damping_exponant=int(np.floor(np.log10(abs(self.damping))))-1
        self.stiffness_exponant=int(np.floor(np.log10(abs(self.stiffness))))-1
        self.compute_state_matrix()

        self.desired_pos = np.zeros((self.observer.B.shape[1], 1))
        self.reference_state = np.zeros((2, 1))

        self.observer_force = Observer(os.path.join(data_path, f"observer_order{order}_force.npz"))
        self.filterForce = np.zeros((self.observer.B.shape[1], 1))


    def setup_compliant_gui(self):
        self.guiNode.addData(name="desired_pos", type="float", value=0.)
        self.guiNode.addData(name="mass", type="float", value=self.mass/(10**self.mass_exponant))
        self.guiNode.addData(name="damping", type="float", value=self.damping/(10**self.damping_exponant))
        self.guiNode.addData(name="stiffness", type="float", value=self.stiffness/(10**self.stiffness_exponant))
        self.guiNode.addData(name="update", type="int", value=0)

        MyGui.MyRobotWindow.addSettingInGroup("Desired position (mm)", self.guiNode.desired_pos, -50, 50, "Control Law")
        MyGui.MyRobotWindow.addSettingInGroup(f"Mass (10^{self.mass_exponant})", self.guiNode.mass, 0, 100, "Reference System")
        MyGui.MyRobotWindow.addSettingInGroup(f"Damping (10^{self.damping_exponant})", self.guiNode.damping, 0, 100, "Reference System")
        MyGui.MyRobotWindow.addSettingInGroup(f"Stiffness (10^{self.stiffness_exponant})", self.guiNode.stiffness, 0, 100, "Reference System")
        MyGui.MyRobotWindow.addSettingInGroup("Update", self.guiNode.update, 0, 1, "Reference System")

        MyGui.PlottingWindow.addData("Desired pos (mm)", self.guiNode.desired_pos)

        self.guiNode.addData(name="observerForce", type="float", value=0.)
        self.guiNode.addData(name="cutoffForceFreq", type="float", value=600.)
        MyGui.MyRobotWindow.addSettingInGroup("Force (0.1Hz)", self.guiNode.cutoffForceFreq, 0, 1000, "Cutoff Frequency")
        MyGui.PlottingWindow.addData("Force (g)", self.guiNode.force)
        MyGui.PlottingWindow.addData("Force obs (g)", self.guiNode.observerForce)


    def compute_state_matrix(self):
        dt = self.root.dt.value
        m = np.array([[self.mass]])
        d = np.array([[self.damping]])
        s = np.array([[self.stiffness]])
        alpha = np.linalg.inv(m + dt * d + dt**2 * s)
        self.A_ref = np.block([[alpha @ m, -dt * alpha @ s], [dt * alpha @ m, np.eye(1) - (dt**2) * alpha * s]])
        self.E_ref = np.block([[dt * alpha], [(dt**2) * alpha]])
        self.B_ref = np.block([[dt*alpha@s], [(dt**2)*alpha@s]])
        self.C_ref = np.array([[0, 1]])


    def execute_control_at_camera_frame(self):
        # observer
        cmd = self.currentMotorPos.reshape(-1, 1)
        self.update_observer(cmd)
        self.observer_force.update(cmd, self.filterMeasure)
        self.filterForce = self.filter(self.observer_force.state[self.nb_state:], self.filterForce, cutoffFreq=self.guiNode.cutoffForceFreq.value)
        self.measurePrev = self.markersPos.copy()

        if self.guiNode.active.value:
            self.reference_pos = self.C_ref @ self.reference_state
            markersPos = self.markers.position.value.flatten()
            self.refMo.position.value = np.array([[self.initRefMo[0], markersPos[1], self.initRefMo[2] + self.reference_pos[0, 0]]])

        # control
        if self.guiNode.controlMode.value == ControlMode["State Feedback"]:
            desiredMotorPos = self.compute_command()
            self.motor.position.value = desiredMotorPos*1e2
        else:
            desiredMotorPos = self.currentMotorPos.copy()
            if self.guiNode.active.value:
                desiredMotorPos[0] = self.motor.position.value*1e-2

        self.command = self.filter(
            desiredMotorPos, self.command,
            cutoffFreq=self.guiNode.cutoffMotorFreq.value, samplingFreq=self.samplingFreq)

        self.desired_pos = np.array([[self.guiNode.desired_pos.value]])
        self.reference_state = self.A_ref @ self.reference_state + self.B_ref @ self.desired_pos + self.E_ref @ self.filterForce

    def execute_control_at_simu_frame(self):
        super().execute_control_at_simu_frame()
        self.guiNode.reference_pos.value = self.reference_pos[0, 0]
        self.guiNode.observerForce.value = self.filterForce[0, 0] / 9.81
        if self.guiNode.update.value:
            self.mass = self.guiNode.mass.value*(10**self.mass_exponant)
            self.damping = self.guiNode.damping.value*(10**self.damping_exponant)
            self.stiffness = self.guiNode.stiffness.value*(10**self.stiffness_exponant)
            try:
                self.compute_state_matrix()
            except Exception as e:
                print(f"Error computing state matrix: {e}")
            self.guiNode.update.value = False
