from .baseController import *


ControlMode = {"Open Loop": False, "State Feedback": True}


class CompliantController(BaseController):
    def __init__(self, leg, motor, markers, load, motorInit, motorMin, motorMax, cutoffFreq, mass, damping, stiffness, order, use_observer=1):
        super().__init__(leg, motor, markers, load, motorInit, motorMin, motorMax, cutoffFreq)

        # add mechanical object for reference
        self.refMo = self.guiNode.addObject("MechanicalObject",
            name="refMo",
            template="Vec3d",
            position=[[0, 0, 0]],
            showObject=True,
            showObjectScale=3,
            drawMode=1,
            showColor=[0, 0, 1, 1]
        )

        self.setup_additional_variables(use_observer, order, mass, damping, stiffness)
        self.setup_additional_gui(use_observer)


    def setup_additional_variables(self, use_observer, order, mass, damping, stiffness):

        self.mass = mass
        self.damping = damping
        self.stiffness = stiffness
        self.mass_exponant=int(np.floor(np.log10(abs(self.mass))))-1
        self.damping_exponant=int(np.floor(np.log10(abs(self.damping))))-1
        self.stiffness_exponant=int(np.floor(np.log10(abs(self.stiffness))))-1
        self.compute_state_matrix()

        # Control and Observer data
        model = np.load(os.path.join(data_path, f"model_order{order}.npz"))
        self.A, self.B, self.E, self.C = model["stateMatrix"], model["inputMatrix"], model["forceMatrix"], model["outputMatrix"]
        control = np.load(os.path.join(data_path, f"controller_order{order}.npz"))
        self.K, self.G = control["feedbackGain"], control["feedForwardGain"]
        if use_observer:
            observer = np.load(os.path.join(data_path, f"observer_order{order}.npz"))
            self.L_state = observer["stateGain"]
            self.L_perturbation = observer["perturbationGain"]
            self.L_force = observer["forceGain"]
        else:
            reduction = np.load(os.path.join(data_path, f"reduction_order{order}.npz"))
            self.R = reduction["reductionMatrix"]

        # additional states for closed-loop control
        self.desired_pos = np.zeros((self.B.shape[1], 1))
        self.reference_pos = np.zeros((2*self.B.shape[1], 1))
        self.observerState = np.zeros((self.A.shape[0], 1))
        self.observerPerturbation = np.zeros((self.B.shape[1], 1))
        self.observerForce = np.zeros((self.E.shape[1], 1))
        self.observerOutput = np.zeros((self.C.shape[0], 1))
        self.measurePrev = np.zeros((self.C.shape[0], 1))
        self.measure_filter = np.zeros((self.C.shape[0], 1))
        self.force_filter = np.zeros((self.E.shape[1], 1))
        self.perturbation_filter = np.zeros((self.B.shape[1], 1))

        # additional constant
        self.use_observer = use_observer

        # additional data storage
        self.observerStateList = []
        self.observerOutputList = []
        self.commandModeList = []


    def execute_control_at_camera_frame(self):
        # observer
        if self.use_observer:
            cmd = self.currentMotorPos.reshape(-1, 1)
            self.observerOutput = self.C @ self.observerState
            measureNoised = self.measurePrev + self.guiNode.noise.value*(2*np.random.rand(self.measurePrev.shape[0], 1)-1)
            # print(f"Measurements: {self.measurePrev.flatten()}")
            # print(f"M with noise: {measureNoised.flatten()}")
            self.measure_filter = self.filter(measureNoised, self.measure_filter, cutoffFreq=1.)

            self.observerState = self.A @ self.observerState + self.B @ cmd + self.E @ self.observerForce + self.L_state @ (self.measure_filter - self.observerOutput)
            self.observerPerturbation = self.observerPerturbation + self.L_perturbation @ (self.measure_filter - self.observerOutput)
            self.observerForce = self.observerForce + self.L_force @ (self.measure_filter - self.observerOutput)

            self.measurePrev = self.markersPos.copy()
            state4Control = self.observerState.copy()

            self.perturbation_filter = self.filter(self.observerPerturbation, self.perturbation_filter, cutoffFreq=100.)
            self.force_filter = self.filter(self.observerForce, self.force_filter, cutoffFreq=100.)

            print(f"force: {self.current_force}, observer: {self.force_filter}")
            # print(f"perturbation: {self.perturbation_filter}")


        if self.guiNode.active.value:
            self.desired_pos = np.array([[self.guiNode.desired_pos.value]])
            markersPos = self.markers.position.value.flatten()
            self.refMo.position.value = np.array([[self.initRefMo[0], markersPos[1], self.initRefMo[2] + self.desired_pos[0, 0]]])

        # control
        if self.guiNode.controlMode.value == ControlMode["State Feedback"]:
            if not self.use_observer:
                fullState = np.vstack([self.legVel, self.legPos])
                state4Control = self.R.T @ fullState
            desiredMotorPos = ( self.G @ self.reference_pos[[1]] - self.K @ state4Control).flatten()
            self.motor.position.value = desiredMotorPos[0]*1e2
            self.reference_pos = self.A_ref @ self.reference_pos + self.B_ref @ self.desired_pos + self.E_ref @ self.force_filter
        else:
            desiredMotorPos = self.currentMotorPos.copy()
            if self.guiNode.active.value:
                desiredMotorPos[0] = self.motor.position.value*1e-2

        self.command = self.filter(
            desiredMotorPos, self.command,
            cutoffFreq=self.cutoffFreq, samplingFreq=self.samplingFreq)


    def execute_control_at_simu_frame(self):
        super().execute_control_at_simu_frame()
        self.guiNode.output.value = self.markersPos[1, 0]
        self.guiNode.obsForce.value = self.observerForce[0, 0]*1e-2
        self.guiNode.reference_pos.value = self.reference_pos[1, 0]

        if self.guiNode.update.value:
            self.mass = self.guiNode.mass.value*(10**self.mass_exponant)
            self.damping = self.guiNode.damping.value*(10**self.damping_exponant)
            self.stiffness = self.guiNode.stiffness.value*(10**self.stiffness_exponant)
            try:
                self.compute_state_matrix()
            except Exception as e:
                print(f"Error computing state matrix: {e}")
            self.guiNode.update.value = False


    def setup_additional_gui(self, use_observer):
        # Specific gui setup
        if use_observer:
            self.guiNode.addData(name="noise", type="float", value=0.)
        self.guiNode.addData(name="desired_pos", type="float", value=0.)
        self.guiNode.addData(name="reference_pos", type="float", value=0.)
        self.guiNode.addData(name="output", type="float", value=0.)
        self.guiNode.addData(name="obsForce", type="float", value=0.)
        self.guiNode.addData(name="controlMode", type="bool", value=ControlMode["Open Loop"])
        self.guiNode.addData(name="mass", type="float", value=self.mass/(10**self.mass_exponant))
        self.guiNode.addData(name="damping", type="float", value=self.damping/(10**self.damping_exponant))
        self.guiNode.addData(name="stiffness", type="float", value=self.stiffness/(10**self.stiffness_exponant))
        self.guiNode.addData(name="update", type="int", value=0)
        MyGui.MyRobotWindow.addSettingInGroup("Reference (mm)", self.guiNode.desired_pos, -50, 50, "Control Law")
        if use_observer:
            MyGui.MyRobotWindow.addSettingInGroup("Observer Noise (mm)", self.guiNode.noise, 0, 10, "Control Law")
        MyGui.MyRobotWindow.addSettingInGroup("Control Mode", self.guiNode.controlMode, 0, 1, "Buttons")
        MyGui.MyRobotWindow.addSettingInGroup(f"Mass (10^{self.mass_exponant})", self.guiNode.mass, 0, 100, "Reference System")
        MyGui.MyRobotWindow.addSettingInGroup(f"Damping (10^{self.damping_exponant})", self.guiNode.damping, 0, 100, "Reference System")
        MyGui.MyRobotWindow.addSettingInGroup(f"Stiffness (10^{self.stiffness_exponant})", self.guiNode.stiffness, 0, 100, "Reference System")
        MyGui.MyRobotWindow.addSettingInGroup("Update", self.guiNode.update, 0, 1, "Reference System")


        # Plotting data
        MyGui.PlottingWindow.addData("Desired pos", self.guiNode.desired_pos)
        MyGui.PlottingWindow.addData("Reference pos", self.guiNode.reference_pos)
        MyGui.PlottingWindow.addData("Output", self.guiNode.output)
        MyGui.PlottingWindow.addData("Force", self.guiNode.force)
        MyGui.PlottingWindow.addData("Obs force", self.guiNode.obsForce)


    def record_data(self):
        super().record_data()
        self.observerStateList.append(self.observerState.copy())
        self.observerOutputList.append(self.observerOutput.copy())
        self.commandModeList.append(self.guiNode.controlMode.value)


    def initialize_simulation(self):
        super().initialize_simulation()
        markerPos = self.markers.position.value.flatten()
        self.refMo.position.value = np.array([[markerPos[0]-10, markerPos[1], markerPos[2]]])
        self.initRefMo = self.refMo.position.value.flatten()


    def save(self):
        print("Saving data...")
        np.savez(
            os.path.join(data_path, "sofa_closedLoop.npz"),
            legVel=np.array(self.legVelList).reshape(len(self.legVelList), self.legVelList[0].shape[0]),
            legPos=np.array(self.legPosList).reshape(len(self.legPosList), self.legPosList[0].shape[0]),
            markersPos=np.array(self.markersPosList).reshape(len(self.markersPosList), self.markersPosList[0].shape[0]),
            motorPos=np.array(self.motorPosList).reshape(len(self.motorPosList), self.motorPosList[0].shape[0]),
            observerState=np.array(self.observerStateList).reshape(len(self.observerStateList), self.observerStateList[0].shape[0]),
            observerOutput=np.array(self.observerOutputList).reshape(len(self.observerOutputList), self.observerOutputList[0].shape[0]),
            commandMode=np.array(self.commandModeList).reshape(len(self.commandModeList), 1),
            fps=1 / self.root.dt.value,
        )

    def compute_state_matrix(self):
        dt = self.root.dt.value
        m = np.array([[self.mass]])
        d = np.array([[self.damping]])
        s = np.array([[self.stiffness]])
        alpha = np.linalg.inv(m + dt * d + dt**2 * s)
        self.A_ref = np.block([[alpha @ m, -dt * alpha @ s], [dt * alpha @ m, np.eye(1) - (dt**2) * alpha * s]])
        self.E_ref = np.block([[dt * alpha], [(dt**2) * alpha]])
        self.B_ref = np.block([[dt*alpha@s], [(dt**2)*alpha@s]])
