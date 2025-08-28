from .baseController import *


ControlMode = {"Open Loop": False, "State Feedback": True}


class ClosedLoopController(BaseController):
    def __init__(self, leg, motor, markers, load, motorInit, motorMin, motorMax, cutoffFreq, order, use_observer=1):
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

        self.setup_additional_variables(use_observer, order)
        self.setup_additional_gui(use_observer)


    def setup_additional_variables(self, use_observer, order):
        # Control and Observer data
        model = np.load(os.path.join(data_path, f"model_order{order}.npz"))
        self.A, self.B, self.C = model["stateMatrix"], model["inputMatrix"], model["outputMatrix"]
        control = np.load(os.path.join(data_path, f"controller_order{order}.npz"))
        self.K, self.G = control["feedbackGain"], control["feedForwardGain"]
        if use_observer:
            observer = np.load(os.path.join(data_path, f"observer_order{order}.npz"))
            self.L = observer["observerGain"]
        else:
            reduction = np.load(os.path.join(data_path, f"reduction_order{order}.npz"))
            self.R = reduction["reductionMatrix"]

        # additional states for closed-loop control
        self.reference = np.zeros((self.B.shape[1], 1))
        self.observerState = np.zeros((self.A.shape[0], 1))
        self.observerOutput = np.zeros((self.C.shape[0], 1))
        self.measurePrev = np.zeros((self.C.shape[0], 1))

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
            measureNoised = self.measurePrev + np.random.normal(0, self.guiNode.noise.value, self.markersPos.shape)
            self.observerState = self.A @ self.observerState + self.B @ cmd + self.L @ (measureNoised - self.observerOutput)
            self.measurePrev = self.markersPos.copy()
            state4Control = self.observerState.copy()


        if self.guiNode.active.value:
            self.reference = np.array([[self.guiNode.reference.value]])
            markersPos = self.markers.position.value.flatten()
            self.refMo.position.value = np.array([[self.initRefMo[0], markersPos[1], self.initRefMo[2] + self.reference[0, 0]]])

        # control
        if self.guiNode.controlMode.value == ControlMode["State Feedback"]:
            if not self.use_observer:
                fullState = np.vstack([self.legVel, self.legPos])
                state4Control = self.R.T @ fullState
            desiredMotorPos = ( self.G @ self.reference - self.K @ state4Control).flatten()
            self.motor.position.value = desiredMotorPos[0]*1e2
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


    def setup_additional_gui(self, use_observer):
        # Specific gui setup
        if use_observer:
            self.guiNode.addData(name="noise", type="float", value=0.)
        self.guiNode.addData(name="reference", type="float", value=0.)
        self.guiNode.addData(name="output", type="float", value=0.)
        self.guiNode.addData(name="controlMode", type="bool", value=ControlMode["Open Loop"])
        MyGui.MyRobotWindow.addSettingInGroup("Reference (mm)", self.guiNode.reference, -50, 50, "Control Law")
        if use_observer:
            MyGui.MyRobotWindow.addSettingInGroup("Observer Noise (mm)", self.guiNode.noise, 0, 3, "Control Law")
        MyGui.MyRobotWindow.addSettingInGroup("Control Mode", self.guiNode.controlMode, 0, 1, "Buttons")

        # Plotting data
        MyGui.PlottingWindow.addData("Reference", self.guiNode.reference)
        MyGui.PlottingWindow.addData("Output", self.guiNode.output)


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
