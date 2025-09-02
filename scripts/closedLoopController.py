from .baseController import *


ControlMode = {"Open Loop": False, "State Feedback": True}


class ClosedLoopController(BaseController):
    def __init__(self, leg, motor, markers, load, motorInit, motorMin, motorMax, cutoffFreq, order, controller_type, observer_type):
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

        self.setup_additional_variables(order, controller_type, observer_type)
        self.setup_additional_gui()


    def setup_additional_variables(self, order, controller_type, observer_type):

        self.controller_type = controller_type
        self.observer_type = observer_type

        # Control and Observer data
        model = np.load(os.path.join(data_path, f"model_order{order}.npz"))
        self.A, self.B, self.C = model["stateMatrix"], model["inputMatrix"], model["outputMatrix"]
        if "force" in observer_type:
            self.E = model["forceMatrix"]

        control = np.load(os.path.join(data_path, f"controller_order{order}.npz"))
        self.K_state = control["statefeedbackGain"]
        if controller_type == "state_feedback":
            self.G = control["feedforwardGain"]
        elif controller_type == "state_feedback_integral":
            self.K_int = control["integralfeedbackGain"]
            self.integral = np.zeros((self.B.shape[1], 1))

        observer = np.load(os.path.join(data_path, f"observer_order{order}.npz"))
        self.L_state = observer["stateGain"]
        if "force" in observer_type:
            self.L_force = observer["forceGain"]
            self.observerForce = np.zeros((self.E.shape[1], 1))
            self.filterForce = np.zeros((self.E.shape[1], 1))
        if "perturbation" in observer_type:
            self.L_perturbation = observer["perturbationGain"]
            self.observerPerturbation = np.zeros((self.B.shape[1], 1))
            self.filterPerturbation = np.zeros((self.B.shape[1], 1))

        # additional states for closed-loop control
        self.reference_pos = np.zeros((self.B.shape[1], 1))
        self.observerState = np.zeros((self.A.shape[0], 1))
        self.observerOutput = np.zeros((self.C.shape[0], 1))
        self.measurePrev = np.zeros((self.C.shape[0], 1))
        self.filterMeasure = np.zeros((self.C.shape[0], 1))

        # additional data storage
        self.observerStateList = []
        self.observerOutputList = []
        self.commandModeList = []


    def update_observer(self, cmd):
        self.observerOutput = self.C @ self.observerState
        measureNoised = self.measurePrev + np.random.normal(0, self.guiNode.noise.value, self.markersPos.shape)
        self.filterMeasure = self.filter(measureNoised, self.filterMeasure, cutoffFreq=self.guiNode.cutoffMeasureFreq.value)
        if "force" == self.observer_type:
            self.observerState = self.A @ self.observerState + self.B @ cmd + self.E @ self.observerForce + self.L_state @ (self.filterMeasure - self.observerOutput)
            self.observerForce += self.L_force @ (self.filterMeasure - self.observerOutput)
            self.filterForce = self.filter(self.observerForce, self.filterForce, cutoffFreq=self.guiNode.cutoffForceFreq.value)
        elif "perturbation" == self.observer_type:
            self.observerState = self.A @ self.observerState + self.B @ cmd + self.L_state @ (self.filterMeasure - self.observerOutput)
            self.observerPerturbation += self.L_perturbation @ (self.filterMeasure - self.observerOutput)
            print(f"observerPerturbation: {self.observerPerturbation}")
            self.filterPerturbation = self.filter(self.observerPerturbation, self.filterPerturbation, cutoffFreq=0.01*self.guiNode.cutoffPerturbationFreq.value)
            print(f"filterPerturbation: {self.filterPerturbation}")
        elif "perturbation_force" == self.observer_type:
            self.observerState = self.A @ self.observerState + self.B @ cmd + self.E @ self.observerForce + self.L_state @ (self.filterMeasure - self.observerOutput)
            self.observerPerturbation += self.L_perturbation @ (self.filterMeasure - self.observerOutput)
            self.observerForce += self.L_force @ (self.filterMeasure - self.observerOutput)
            self.filterPerturbation = self.filter(self.observerPerturbation, self.filterPerturbation, cutoffFreq=0.01*self.guiNode.cutoffPerturbationFreq.value)
            self.filterForce = self.filter(self.observerForce, self.filterForce, cutoffFreq=self.guiNode.cutoffForceFreq.value)
        else:
            self.observerState = self.A @ self.observerState + self.B @ cmd + self.L_state @ (self.filterMeasure - self.observerOutput)


    def compute_command(self):
        if self.controller_type == "state_feedback_integral":
            desiredMotorPos = (-self.K_state @ self.observerState - self.K_int @ self.integral).flatten()
            self.integral += self.reference_pos - self.filterMeasure[[1]]
        elif self.controller_type == "state_feedback" and "perturbation" in self.observer_type:
            desiredMotorPos = ( self.G @ self.reference_pos - self.K_state @ self.observerState - self.filterPerturbation).flatten()
        else:
            desiredMotorPos = ( self.G @ self.reference_pos - self.K_state @ self.observerState).flatten()
        return desiredMotorPos[0]


    def execute_control_at_camera_frame(self):
        # observer
        cmd = self.currentMotorPos.reshape(-1, 1)
        self.update_observer(cmd)
        self.measurePrev = self.markersPos.copy()


        if self.guiNode.active.value:
            self.reference_pos = np.array([[self.guiNode.reference_pos.value]])
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


    def execute_control_at_simu_frame(self):
        super().execute_control_at_simu_frame()
        self.guiNode.output.value = self.markersPos[1, 0]
        if "force" in self.observer_type:
            self.guiNode.observerForce.value = self.filterForce[0, 0] / 9.81


    def setup_additional_gui(self):
        self.guiNode.addData(name="noise", type="float", value=0.)
        self.guiNode.addData(name="reference_pos", type="float", value=0.)
        self.guiNode.addData(name="output", type="float", value=0.)
        self.guiNode.addData(name="controlMode", type="bool", value=ControlMode["Open Loop"])
        self.guiNode.addData(name="cutoffMotorFreq", type="float", value=self.cutoffFreq)
        self.guiNode.addData(name="cutoffMeasureFreq", type="float", value=600.)


        MyGui.MyRobotWindow.addSettingInGroup("Reference (mm)", self.guiNode.reference_pos, -50, 50, "Control Law")
        MyGui.MyRobotWindow.addSettingInGroup("Observer Noise (mm)", self.guiNode.noise, 0, 3, "Control Law")
        MyGui.MyRobotWindow.addSettingInGroup("Control Mode", self.guiNode.controlMode, 0, 1, "Buttons")
        MyGui.MyRobotWindow.addSettingInGroup("Motor (Hz)", self.guiNode.cutoffMotorFreq, 0, 100, "Cutoff Frequency")
        MyGui.MyRobotWindow.addSettingInGroup("Measurement (0.1Hz)", self.guiNode.cutoffMeasureFreq, 0, 1000, "Cutoff Frequency")
        if "perturbation" in self.observer_type:
            self.guiNode.addData(name="cutoffPerturbationFreq", type="float", value=60.)
            MyGui.MyRobotWindow.addSettingInGroup("Perturbation (0.01Hz)", self.guiNode.cutoffPerturbationFreq, 0, 100, "Cutoff Frequency")

        # Plotting data
        MyGui.PlottingWindow.addData("Reference (mm)", self.guiNode.reference_pos)
        MyGui.PlottingWindow.addData("Output (mm)", self.guiNode.output)
        MyGui.PlottingWindow.addData("Motor (rad)", self.motor.JointActuator.value)

        if "force" in self.observer_type:
            self.guiNode.addData(name="observerForce", type="float", value=0.)
            self.guiNode.addData(name="cutoffForceFreq", type="float", value=600.)
            MyGui.MyRobotWindow.addSettingInGroup("Force (0.1Hz)", self.guiNode.cutoffForceFreq, 0, 1000, "Cutoff Frequency")
            MyGui.PlottingWindow.addData("Force (g)", self.guiNode.force)
            MyGui.PlottingWindow.addData("Force obs (g)", self.guiNode.observerForce)


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
