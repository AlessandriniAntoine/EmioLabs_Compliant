import os
from math import pi
import numpy as np
import Sofa
import Sofa.ImGui as MyGui

lab_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
data_path = os.path.join(lab_path, "data")


class BaseController(Sofa.Core.Controller):
    def __init__(self, leg, motor, markers, load, motorInit, motorMin, motorMax, cutoffFreq):
        Sofa.Core.Controller.__init__(self)
        self.name = "Control Strategy"
        self.root = leg.getRoot()
        self.leg = leg
        self.motor = motor
        self.markers = markers.MechanicalObject
        self.load = load

        # Motor setup
        speedLimit = 75.57
        self.speedLimit = speedLimit * 2 * pi / 60 * self.root.dt.value
        self.motorInit = np.array(motorInit)
        self.motorMin = np.array(motorMin)
        self.motorMax = np.array(motorMax)
        self.cutoffFreq = cutoffFreq
        self.samplingFreq = 60.
        self.motor.addData(name="position", type="float", value=0.)

        # Forces setup
        self.root.VisualStyle.displayFlags.value = "showVisualModels showForceFields"
        self.force = self.load.addObject('ConstantForceField', name="User", indices=[0], showArrowSize=0.03)
        self.force.forces.value = [7 * [0.]]
        self.maxforce = 40

        # GUI setup
        self.guiNode = self.root.addChild("guiNode")
        self.setup_gui()

        # Initialize simulation variables
        self.setup_variables()


    def setup_gui(self):
        MyGui.MyRobotWindow.addSettingInGroup("Motor (rad/100)", self.motor.position, self.motorMin*1e2, self.motorMax*1e2, "Motor Movement")

        self.guiNode.addData(name="force", type="float", value=0.)
        self.guiNode.addData(name="record", type="bool", value=False)
        self.guiNode.addData(name="saving", type="bool", value=False)
        self.guiNode.addData(name="active", type="bool", value=False)

        MyGui.MyRobotWindow.addSettingInGroup("Horizontal", self.guiNode.force, -self.maxforce, self.maxforce, "User Actions")
        MyGui.MyRobotWindow.addSettingInGroup("Active", self.guiNode.active, 0, 1, "Buttons")
        MyGui.MyRobotWindow.addSettingInGroup("Record", self.guiNode.record, 0, 1, "Buttons")
        MyGui.MyRobotWindow.addSettingInGroup("Saving", self.guiNode.saving, 0, 1, "Buttons")


    def setup_variables(self):

        # constants
        self.initThreshold = 5e-2
        self.step = 0
        self.dtRatio = int(round(1 / (60 * self.root.dt.value)))
        self.mo = self.leg.getChild("LegDeformablePart").MechanicalObject

        # measures
        self.legVel = np.zeros_like(self.mo.velocity.value.copy())
        self.legPos = np.zeros_like(self.mo.position.value.copy())
        self.lastLegPos = np.zeros_like(self.mo.position.value.copy())
        self.markersPos = np.zeros_like(self.markers.position.value.copy()[:, [1, 2]].reshape(-1, 1))
        self.currentMotorPos = np.zeros((1, ))
        self.current_force = np.zeros((1, ))

        # states
        self.command = np.zeros((1, ))

        # boolean
        self.start = False
        self.hasSaved = False

        # data storage
        self.legPosList = []
        self.legVelList = []
        self.markersPosList = []
        self.motorPosList = []
        self.forceList = []


    def onAnimateBeginEvent(self, e):
        if self.start:
            self.measure_data()

            if self.step % self.dtRatio == 0:
                self.step = 0
                self.execute_control_at_camera_frame()

            self.execute_control_at_simu_frame()

            if self.step % self.dtRatio == 0 and self.guiNode.record.value:
                self.record_data()

            self.step += 1
        else:
            self.initialize_simulation()

        if self.hasSaved:
            self.hasSaved = False
            self.guiNode.saving.value = False
        if self.guiNode.saving.value and not self.hasSaved:
            self.hasSaved = True
            self.save()
        if self.guiNode.record.value and not self.guiNode.active.value:
            self.guiNode.active.value = True


    def initialize_simulation(self):
        legVel = self.mo.velocity.value.copy()
        legPos = self.mo.position.value.copy()
        if np.linalg.norm(legPos - self.lastLegPos) < self.initThreshold:
            self.start = True
            self.guiNode.active.value = True
            print("Simulation started")
        self.lastLegPos = legPos
        self.initialLegVel = legVel[:, [1, 2]].reshape(-1, 1)
        self.initialLegPos = legPos[:, [1, 2]].reshape(-1, 1)
        self.initialMarkersPos = self.markers.position.value.copy()[:, [1, 2]].reshape(-1, 1)


    def measure_data(self):
        self.legVel = self.mo.velocity.value.copy()
        self.legVel = self.legVel[:, [1, 2]].reshape(-1, 1) - self.initialLegVel
        self.legPos = self.mo.position.value.copy()
        self.legPos = self.legPos[:, [1, 2]].reshape(-1, 1) - self.initialLegPos
        markersPos = self.markers.position.value.copy()[:, [1, 2]].reshape(-1, 1)
        self.markersPos = markersPos - self.initialMarkersPos
        self.currentMotorPos = np.array([self.motor.JointActuator.value.value]) - self.motorInit
        self.current_force = self.force.forces.value.flatten()[[2]]


    def execute_control_at_simu_frame(self):
        motorDisplacement = np.clip(self.command - self.currentMotorPos, -self.speedLimit, self.speedLimit)
        command = self.currentMotorPos + motorDisplacement
        command = np.clip(command, self.motorMin, self.motorMax)
        self.motor.JointActuator.value.value = command[0] + self.motorInit
        self.force.forces = [[ 0, 0, self.guiNode.force.value*1e2, 0, 0, 0, 0]]

    def execute_control_at_camera_frame(self):
       raise NotImplementedError("execute_control_at_camera_frame method must be implemented in the derived class.")


    def record_data(self):
        self.legVelList.append(self.legVel.copy())
        self.legPosList.append(self.legPos.copy())
        self.markersPosList.append(self.markersPos.copy())
        self.motorPosList.append(np.array([self.command]).reshape(-1, 1))
        self.forceList.append(self.current_force)


    def save(self):
        raise NotImplementedError("save method must be implemented in the derived class.")


    def filter(self, signal, signalFilter, cutoffFreq=1., samplingFreq=60.):
        samplingPeriod = 1 / samplingFreq
        tau = 1 / (2 * np.pi * cutoffFreq)
        alpha = samplingPeriod / (tau + samplingPeriod)
        return alpha * signal + (1 - alpha) * signalFilter
