import os
import time
import json
import numpy as np
import tkinter as tk
import multiprocessing
import cv2 as cv

from emioapi import EmioMotors
from emioapi._depthcamera import DepthCamera

from utils import Observer, StateFeedbackIntegral

lab_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
data_path = os.path.join(lab_path, "data")


nbMarkers = 2  # Number of markers to track
ControlMode = {"Open Loop": False, "Closed Loop": True}


def arg_parse():
    import sys
    import argparse
    parser = argparse.ArgumentParser(prog=sys.argv[0],
                                     description='Simulate two legs.')

    parser.add_argument('-mi', '--motorInit', type=str,
                        help="Initial values for the motors (in radians). ",
                        default='0.', dest="motorInit")
    parser.add_argument('-mx', '--motorMax', type=str,
                        help="Maximum values for the motors (in radians). ",
                        default='1.57', dest="motorMax")
    parser.add_argument('-mn', '--motorMin', type=str,
                        help="Minimum values for the motors (in radians). ",
                        default='-1.57', dest="motorMin")
    parser.add_argument("--motorCutoffFreq", type=str,
                        help="Motor cutoff frequency",
                        default='20.', dest="motorCutoffFreq")

    parser.add_argument('--order', type=str,
                        help="Specify the order of the reduction",
                        default='7', dest="order")


    try:
        args = parser.parse_args()
    except SystemExit:
        Sofa.msg_error(sys.argv[0], "Invalid arguments, get defaults instead.")
        args = parser.parse_args([])

    return args

###########################################################################
# TKINTER APP
###########################################################################
class ClosedLoopLinearApp:
    def __init__(self, root, sharedMotorPos, sharedRefPos, sharedStart, sharedControlMode, sharedRecord, sharedSave):
        self.root = root
        self.root.title('Emio Linear Closed Loop Control')
        self.root.geometry("500x250")

        # shared Variables
        self.sharedMotorPos = sharedMotorPos
        self.sharedRefPos = sharedRefPos
        self.sharedStart = sharedStart
        self.sharedControlMode = sharedControlMode
        self.sharedRecord = sharedRecord
        self.sharedSave = sharedSave

        # app variable
        self.start = False
        self.active = True

        # Information Frame
        infoFrame = tk.Frame(root)
        infoFrame.pack()
        info = ["Start", "Closed Loop", "Active"]
        infoValue = ["False", "False", "True"]
        self.labelInfo = {}
        for i, item in enumerate(info):
            self.labelInfo[item] = tk.Label(infoFrame, text=f"{item}: {infoValue[i]}", font=("Arial", 12))
            self.labelInfo[item].pack(side="left", padx=5, pady=5)

        # Motor Sliders
        motorLabelFrame = tk.Frame(root)
        motorLabelFrame.pack()
        motorSliderFrame = tk.Frame(root)
        motorSliderFrame.pack()

        self.labelMotor = tk.Label(motorLabelFrame, text='Motor: 0.00 (rad)')
        self.labelMotor.pack(side="left", padx=3, pady=5)
        self.sliderMotor = tk.Scale(motorSliderFrame, from_=-100, to=100, orient='horizontal', command=self.sliderMotorAction)
        self.sliderMotor.pack(side="left", padx=3, pady=5)
        self.sliderMotor.set(0)

        # Reference Sliders
        refLabelFrame = tk.Frame(root)
        refLabelFrame.pack()
        refSliderFrame = tk.Frame(root)
        refSliderFrame.pack()

        self.labelRef = tk.Label(refLabelFrame, text='Ref: 0.00 (mm)')
        self.labelRef.pack(side="left", padx=3, pady=5)
        self.sliderRef = tk.Scale(refSliderFrame, from_=-100, to=100, orient='horizontal', command=self.sliderRefAction)
        self.sliderRef.pack(side="left", padx=3, pady=5)
        self.sliderRef.set(0)

        # Buttons
        buttonFrame = tk.Frame(root)
        buttonFrame.pack(pady=10)
        self.startButton = tk.Button(buttonFrame, text="Start", command=self.startAction)
        self.startButton.pack(side='left', padx=3)

        self.controlModeButton = tk.Button(buttonFrame, text="Control Mode", command=self.controlModeAction)
        self.controlModeButton.pack(side='right', padx=3)

        self.activeButton = tk.Button(buttonFrame, text="Active", command=self.activeAction)
        self.activeButton.pack(side='right', padx=3)

        self.recordingButton = tk.Button(buttonFrame, text="Record", command=self.recordAction)
        self.recordingButton.pack(side='right', padx=3)

        self.saveButton = tk.Button(buttonFrame, text="Save", command=self.saveAction)
        self.saveButton.pack(side='right', padx=3)


    def sliderMotorAction(self, val):
        motorPos = int(val) * np.pi / 2 * 1 / 100
        self.labelMotor.config(text=f'Motor: {motorPos:.2f} (rad)')
        with self.sharedMotorPos.get_lock():
            if self.active:
                self.sharedMotorPos[0] = motorPos


    def sliderRefAction(self, val):
        refPos = float(val)
        self.labelRef.config(text=f'Ref: {refPos:.2f} (mm)')
        with self.sharedRefPos.get_lock():
            if self.active:
                self.sharedRefPos[0] = refPos

    def startAction(self):
        self.start = True
        with self.sharedStart.get_lock():
            self.sharedStart.value = True
            self.labelInfo["Start"].config(text="Start: True")

    def controlModeAction(self):
        with self.sharedControlMode.get_lock():
            if self.sharedControlMode.value == ControlMode["Open Loop"]:
                self.sharedControlMode.value = ControlMode["Closed Loop"]
                self.labelInfo["Closed Loop"].config(text="Closed Loop: True")
            else:
                self.sharedControlMode.value = ControlMode["Open Loop"]
                self.labelInfo["Closed Loop"].config(text="Closed Loop: False")

    def activeAction(self):
        self.active = not self.active
        if self.active:
            self.labelInfo["Active"].config(text="Active: True")
            if self.sharedControlMode.value == ControlMode["Open Loop"]:
                self.sharedMotorPos[0] = self.sliderMotor.get() * np.pi / 2 * 1 / 100
            else:
                self.sharedRefPos[0] = self.sliderRef.get()
        else:
            self.labelInfo["Active"].config(text="Active: False")


    def recordAction(self):
        with self.sharedRecord.get_lock():
            self.sharedRecord.value = not self.sharedRecord.value
            if self.sharedRecord.value:
                self.active = True
                if self.sharedControlMode.value == ControlMode["Open Loop"]:
                    self.sharedMotorPos[0] = self.sliderMotor.get() * np.pi / 2 * 1 / 100
                else:
                    self.sharedRefPos[0] = self.sliderRef.get()
                print("Recording started...")

    def saveAction(self):
        with self.sharedSave.get_lock():
            self.sharedSave.value = True

    def close_app(self):
        self.root.destroy()


############################################################################
# Low-pass filter function
#############################################################################
def filterFirstOder(signal, signalFiltered, cutoffFreq=1., samplingFreq=60.):
    """Apply a first-order low-pass filter to the signal."""
    timeConstant = 1 / (2 * np.pi * cutoffFreq)
    samplingPeriod = 1 / samplingFreq
    alpha = samplingPeriod / (timeConstant + samplingPeriod)
    signalFiltered = alpha * signal + (1 - alpha) * signalFiltered
    return signalFiltered

############################################################################
# Process
#############################################################################
def processCamera(trackerPos, event):
    """Update tracker position."""
    json_path = os.path.join(data_path, "cameraparameter.json")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Camera parameter file not found at {json_path}")
    with open(json_path, 'r') as fp:
        json_parameters = json.load(fp)

    camera = DepthCamera(
        show_video_feed=True,
        tracking=True,
        compute_point_cloud=False,
        parameter=json_parameters)
    camera.set_fps(60)
    camera.set_depth_max(800)
    camera.set_depth_min(0)
    camera.open()

    while True:
        camera.update()
        if len(camera.trackers_pos) == nbMarkers:
            with trackerPos.get_lock():
                trackerPos[:] = np.array(camera.trackers_pos).flatten()
        event.set()

        k = cv.waitKey(1)
        if k == ord('q'):
            camera.quit()
            break

def processSlider(sharedMotorPos, sharedRefPos,
                  sharedStart, sharedControlMode, sharedRecord, sharedSave):
    root = tk.Tk()
    app = ClosedLoopLinearApp(root, sharedMotorPos, sharedRefPos,
        sharedStart, sharedControlMode, sharedRecord, sharedSave)
    root.protocol("WM_DELETE_WINDOW", app.close_app)
    root.mainloop()

def processController(trackerPos, sharedMotorPos, sharedRefPos, sharedStart, sharedControlMode, sharedRecord, sharedSave, event):

    dt = 1/60
    m = np.array([[0.1]])
    d = np.array([[10]])
    s = np.array([[100]])
    alpha = np.linalg.inv(m + dt * d + dt**2 * s)
    A_ref = np.block([[alpha @ m, -dt * alpha @ s], [dt * alpha @ m, np.eye(1) - (dt**2) * alpha * s]])
    E_ref = np.block([[dt * alpha], [(dt**2) * alpha]])
    B_ref = np.block([[dt*alpha@s], [(dt**2)*alpha@s]])

    args = arg_parse()
    motors = EmioMotors()
    while not motors.open():
        print("Waiting for motors to open...")
        time.sleep(1)
    print("Motors opened successfully.")

    # Control mode
    counterRecord = 0
    maxRecord = 400
    start = False
    cutoffFreqMeasure = 5.  # Hz
    cutoffFreqMotor = float(args.motorCutoffFreq)  # Hz
    samplingFreq = 60.  # Hz

    # Motors Limits
    motorInit = float(args.motorInit)
    motorMin = -float(args.motorMin)
    motorMax = float(args.motorMax)

    # control data
    order = int(args.order)

    controller = StateFeedbackIntegral(os.path.join(data_path, f"controller_order{order}.npz"))
    observer = Observer(os.path.join(data_path, f"observer_order{order}_default.npz"))
    observer_force = Observer(os.path.join(data_path, f"observer_order{order}_force.npz"))
    force_filter = np.zeros((observer.B.shape[1], 1))

    # Initialize variables
    initialMarkersPos = np.zeros((3*nbMarkers, ))
    measureFiltered = np.zeros((3*nbMarkers, ))
    markersPosPrev = np.zeros((3*nbMarkers, ))
    motorPos = np.zeros((observer.B.shape[1],))
    motorPosPrev = np.zeros((observer.B.shape[1],))
    currentMotorPos = np.zeros((observer.B.shape[1],))
    reference_pos = np.zeros((2*observer.B.shape[1],1))
    desired_pos = np.zeros((observer.B.shape[1],1))

    # recorded data
    markersPositions = []
    motorPositions = []
    referenceList = []
    outputList = []
    observerStatesList = []
    observerOutputList = []
    controlModeList = []

    motors.angles = [motorInit, 0, 0, 0]
    print(f"Motors initialized to {motors.angles}")

    while True:
        # t1 = time.time()
        # print(f"fps: {1/(t1 - t0)}")
        # t0 = t1

        event.wait()
        event.clear()

        # Measurements
        with trackerPos.get_lock():
            measure = np.array(trackerPos[:])

        # order the markers
        pos = measure.reshape(nbMarkers, 3)
        i_sorted_y = sorted([0, 1], key=lambda i: pos[i, 1])
        new_order = [i_sorted_y[0], i_sorted_y[1]]
        pos_sorted = pos[new_order]
        measure = pos_sorted.flatten()
        # print(f"Measurements: {measure}")

        # Filter the measurements
        measureFiltered = filterFirstOder(measure, measureFiltered,
                                          cutoffFreq=cutoffFreqMeasure, samplingFreq=samplingFreq)

        angles = motors.angles
        if angles is not None:
            currentMotorPos = np.array([angles[0] - motorInit])

        if start:
            markersPos = measureFiltered - initialMarkersPos
            if np.linalg.norm(markersPos - markersPosPrev) > 70:
                print("Filtering markers ...")
                markersPos = markersPosPrev.copy()
            # print(f"markersPos: {markersPos.flatten()}")


            output = (markersPos.reshape(-1,1))[[x for i in range(nbMarkers) for x in [3*i+1, 3*i+2]]]
            outputPrev = (markersPosPrev.reshape(-1,1))[[x for i in range(nbMarkers) for x in [3*i+1, 3*i+2]]]

            # observer update
            cmd = np.array([currentMotorPos]).reshape(-1, 1)
            observer.update(cmd, outputPrev)
            observer_force.update(cmd, outputPrev)
            force_filter = filterFirstOder(observer_force.state[2*order:], force_filter, cutoffFreq=60.)
            desired_pos = np.array([sharedRefPos[:]]).reshape(-1, 1)

            if sharedControlMode.value == ControlMode["Open Loop"]:
                with sharedMotorPos.get_lock():
                    command = np.array(sharedMotorPos[:])
            else:
                controller.update(observer.state, reference_pos[[1]], output[[1]])
                command = controller.command.flatten()
                reference_pos = A_ref @ reference_pos + B_ref @ desired_pos + E_ref @ force_filter

            # apply the motor position
            motorPos = filterFirstOder(command, motorPos, cutoffFreq=cutoffFreqMotor, samplingFreq=samplingFreq)
            motorPos = np.clip(motorPos, motorMin, motorMax)
            if (motorPosPrev - motorPos).any():
                motors.angles = [motorInit + motorPos[0], 0, 0, 0]
                motorPosPrev = motorPos

            # save the data
            with sharedRecord.get_lock():
                if sharedRecord.value and counterRecord < maxRecord:
                    markersPositions.append(markersPos.flatten())
                    motorPositions.append(currentMotorPos)
                    referenceList.append(reference_pos.flatten())
                    outputList.append(output.flatten())
                    observerStatesList.append(observerState.flatten())
                    observerOutputList.append(observerOutput.flatten())
                    controlModeList.append(sharedControlMode.value)
                    counterRecord +=1

            if counterRecord == maxRecord:
                print("Recording finished")
                sharedRecord.value = False
                counterRecord = 0

            markersPosPrev = markersPos.copy()
        else:
            motorPosPrev = currentMotorPos
            initialMarkersPos = measureFiltered.copy()


        if sharedSave.value:
            print("Data saved...")
            np.savez(os.path.join(data_path, "hardware_closedLoop.npz"),
                markersPos=markersPositions,
                motorPos=motorPositions,
                reference=referenceList,
                outputsPos=outputList,
                observerState=observerStatesList,
                observerOutput=observerOutputList,
                controlMode=controlModeList,
                initialMarkersPos=initialMarkersPos,
            )
            sharedSave.value = False
        if sharedStart.value and not start:
            start = True
            markersPosPrev = measureFiltered - initialMarkersPos


def main():
    import time

    # shared variables
    trackerPos = multiprocessing.Array("d", 3 * nbMarkers * [0.])
    sharedMotorPos = multiprocessing.Array("d", 1 * [0.0])
    sharedRefPos = multiprocessing.Array("d", 1 * [0.0])
    sharedStart = multiprocessing.Value("i", False)
    sharedControlMode = multiprocessing.Value("i", False)
    sharedRecord = multiprocessing.Value("i", False)
    sharedSave = multiprocessing.Value("i", False)
    event = multiprocessing.Event()

    # Create the GUI application


    # Create processes
    p1 = multiprocessing.Process(target=processCamera, args=(trackerPos, event))
    p2 = multiprocessing.Process(target=processSlider, args=(sharedMotorPos, sharedRefPos,
        sharedStart, sharedControlMode, sharedRecord, sharedSave))
    p3 = multiprocessing.Process(target=processController, args=(
        trackerPos, sharedMotorPos, sharedRefPos,
        sharedStart, sharedControlMode, sharedRecord, sharedSave, event
    ))

    p1.start()
    p2.start()
    p3.start()

    try:
        while True:
            time.sleep(5)
    except KeyboardInterrupt:
        p1.terminate()
        p2.terminate()
        p3.terminate()


def createScene(root):
    """Create the scene for the direct control application."""
    main()

if __name__ == "__main__":
    main()
