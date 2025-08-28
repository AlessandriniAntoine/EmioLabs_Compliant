from .baseController import *


class OpenLoopController(BaseController):
    def __init__(self, leg, motor, markers, load, motorInit, motorMin, motorMax, cutoffFreq):
        super().__init__(leg, motor, markers, load, motorInit, motorMin, motorMax, cutoffFreq)

        # Specific gui setup
        group = "Mechanical Parameters"
        self.guiNode.addData(name="loadMass", type="float", value=self.load.UniformMass.totalMass.value*1e3)
        MyGui.MyRobotWindow.addSettingInGroup("Load Mass (g)", self.guiNode.loadMass, 0.0, 5000., group)

        self.guiNode.addData(name="noise", type="float", value=0.)
        MyGui.MyRobotWindow.addSettingInGroup("Noise (% pi/4)", self.guiNode.noise, 0, 100, "Motor Movement")


    def execute_control_at_camera_frame(self):
        desiredMotorPos = self.currentMotorPos.copy()
        if self.guiNode.active.value:
            noise = self.guiNode.noise.value / 100 * np.pi / 4
            desiredMotorPos[0] = self.motor.position.value*1e-2 + np.random.normal(0, noise, 1)
        self.command = self.filter(
            desiredMotorPos, self.command,
            cutoffFreq=self.cutoffFreq, samplingFreq=self.samplingFreq)


    def execute_control_at_simu_frame(self):
        super().execute_control_at_simu_frame()
        self.load.UniformMass.totalMass.value = self.guiNode.loadMass.value*1e-3


    def save(self):
        print("Saving data...")
        np.savez(
            os.path.join(data_path, "sofa_openLoop.npz"),
            legVel=np.array(self.legVelList).reshape(len(self.legVelList), self.legVelList[0].shape[0]),
            legPos=np.array(self.legPosList).reshape(len(self.legPosList), self.legPosList[0].shape[0]),
            markersPos=np.array(self.markersPosList).reshape(len(self.markersPosList), self.markersPosList[0].shape[0]),
            motorPos=np.array(self.motorPosList).reshape(len(self.motorPosList), self.motorPosList[0].shape[0]),
            fps=1 / self.root.dt.value,
        )
