import os
import sys

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../../")

import Sofa
from math import pi, sin, cos

from utils.header import addHeader, addSolvers

from splib3.animation import AnimationManager, animate
from splib3.loaders import getLoadingLocation
from splib3.numerics import to_radians

from parts.motor import Motor
from parts.leg import Leg
from parts.camera import Camera
from parts.centerpart import CenterPart
from utils.topology import applyTranslation, applyRotation
from utils import getColorFromFilename, getListFromArgs
from utils import RGBAColor


def arg_parse():
    import sys
    import argparse
    parser = argparse.ArgumentParser(prog=sys.argv[0],
                                     description='Simulate two legs.')

    parser.add_argument('-ln', '--legName', type=str, nargs='*',
                        help="name of the leg (ex: blueleg, greenleg, greyleg)",
                        default='yellowleg', dest="legName")
    parser.add_argument('-lm', '--legModel', type=str, nargs='*',
                        help="name of the model (beam, cosserat, or tetra)",
                        default='tetra', dest="legModel")
    parser.add_argument('-lp', '--legPositionOnMotor', type=str, nargs='*',
                        help="position on motor (clockwiseup, counterclockwiseup, clockwisedown, or counterclockwisedown)",
                        default='clockwisedown', dest="legPositionOnMotor")

    parser.add_argument('-mi', '--motorInit', type=str,
                        help="Initial values for the motors (in radians). ",
                        default='0.', dest="motorInit")
    parser.add_argument('-mx', '--motorMax', type=str,
                        help="Maximum values for the motors (in radians). ",
                        default='1.57', dest="motorMax")
    parser.add_argument('-mn', '--motorMin', type=str,
                        help="Minimum values for the motors (in radians). ",
                        default='1.57', dest="motorMin")
    parser.add_argument("--motorCutoffFreq", type=str,
                        help="Motor cutoff frequency",
                        default='30.', dest="motorCutoffFreq")

    parser.add_argument('-ctr', "--controller", type=str,
                        help="Select the controller that will be used.",
                        default="openloop", dest="controller")
    parser.add_argument('--order', type=str,
                        help="Specify the order of the reduction",
                        default='7', dest="order")
    parser.add_argument('--useObserver', type=str,
                        help="Specify the order of the reduction",
                        default='1', dest="useObserver")

    parser.add_argument('--no-connection', dest="connection", action='store_false',
                        help="use when you want to run the simulation without the robot")
    try:
        args = parser.parse_args()
    except SystemExit:
        Sofa.msg_error(sys.argv[0], "Invalid arguments, get defaults instead.")
        args = parser.parse_args([])

    return args


def attachLoad(centerpart):
    load = centerpart.addChild("Load")
    load.addObject("MechanicalObject", template="Rigid3", position=[0, 33, 0, 0., 0, 0., 1],
                   showObject=False, showObjectScale=20)
    load.addObject("UniformMass", totalMass=0.315)
    visual = load.addChild("Visual")
    visual.addObject("MeshSTLLoader", filename="../../data/meshes/centerparts/greymass.stl",
                     translation=[0, -15, 0],
                     rotation=[-90, 90, 0])
    visual.addObject("OglModel", src=visual.MeshSTLLoader.linkpath, color=RGBAColor.grey)
    visual.addObject("RigidMapping")

    load.addObject('RigidMapping', name="load-center-rigid-mapping", input="@..", output="@.")


def createScene(rootnode):
    import myparameters

    args = arg_parse()

    ##############################################################################
    # Parameters
    ##############################################################################
    dt = 1 / 60
    motorInit = float(args.motorInit)
    motorMin = -float(args.motorMin)
    motorMax = float(args.motorMax)

    # Header of the simulation
    settings, modelling, simulation = addHeader(rootnode, withCollision=False)
    rootnode.addObject(AnimationManager(rootnode))
    rootnode.dt = dt
    rootnode.gravity = [0., -9810, 0.]
    addSolvers(simulation)
    settings.addObject('RequiredPlugin', name='Sofa.Component.Constraint.Projective')
    rootnode.Simulation.EulerImplicitSolver.rayleighStiffness = 0.009

    ##############################################################################
    # Motor
    ##############################################################################
    translation = [0, 0, 102.5]
    motor = Motor(name="Motor",
                  translation=translation,
                  rotation=[0, 180, 0],
                  tempvisurotation=[-90, 180, 0])
    motor.Parts.MotorVisual.activated.value = False
    simulation.addChild(motor)

    motor.addObject("JointConstraint", name="JointActuator", index=0, value=0, valueType="displacement", minDisplacement=-pi, maxDisplacement=pi)
    motor.JointActuator.value = motorInit

    ##############################################################################
    # Leg
    ##############################################################################
    leg = Leg(name="Leg",
              legName=args.legName,
              positionOnMotor=args.legPositionOnMotor,
              model=args.legModel,
              translation=translation,
              rotation=[0, 180, 0],
              youngModulus=myparameters.youngModulus,
              poissonRatio=myparameters.poissonRatio,
              thickness=myparameters.thickness,
              width=myparameters.width,
              )
    simulation.addChild(leg)

    # Attach the leg to motor
    leg.attachBase(motor.Parts, 1)

    ##############################################################################
    # Load
    ##############################################################################
    load = simulation.addChild("Load")
    load.addObject("MechanicalObject", template="Rigid3", position=[[0, -200, 80, 0.707, -0.707, 0., 0.]])
    load.addObject("UniformMass", totalMass=0.36)
    visual = load.addChild("Visual")
    visual.addObject("MeshSTLLoader",
                     filename=getLoadingLocation("../../data/meshes/centerparts/greymass.stl", __file__),
                     translation=[10, 0, 0],
                     rotation=[0, 90, 0])
    visual.addObject("OglModel", src=visual.MeshSTLLoader.linkpath, color=RGBAColor.grey)
    visual.addObject("RigidMapping")

    # Attach the load to the leg
    leg.attachExtremity(load, 0)

    ##############################################################################
    # Camera
    ##############################################################################
    camera = modelling.addChild(Camera(rotation=[180, -135, 0], translation=[0, 10, 0]))

    ##############################################################################
    # Base + Platform
    ##############################################################################
    box = modelling.addChild("Base")
    box.addObject("MeshSTLLoader",
                filename=getLoadingLocation("../../data/meshes/base-extended.stl", __file__))
    # Platform
    platform = modelling.addChild("Platform")
    platform.addObject("MeshSTLLoader",
                filename=getLoadingLocation("../../data/meshes/supports/platform.stl", __file__))
    platform.addObject("OglModel", src=platform.MeshSTLLoader.linkpath, color=[1, 1, 1, 0.1])
    box.addObject("OglModel", src=box.MeshSTLLoader.linkpath, color=[1, 1, 1, 0.1])

    ##############################################################################
    # Drums
    ##############################################################################
    for i in range(3):
        drum = modelling.addChild("Drum" + str(i+1))
        drum.addObject("MeshSTLLoader",
                      filename=getLoadingLocation("../../data/meshes/legmotorattachbase.stl", __file__),
                       rotation=[0, [90, 180, -90][i], 0],
                       translation=[[-100, 0, 0],[0, 0, -100],[100, 0, 0]][i])
        drum.addObject("OglModel", src=drum.MeshSTLLoader.linkpath, color=[1, 1, 1, 0.1])

    ##############################################################################
    # Markers
    ##############################################################################
    markers = leg.leg.addChild("Markers")
    markers.addObject("MechanicalObject",
                      position=[[-5, -191, -22.5], [-5, -100, -22.5]],
                      translation=translation,
                      showObject=True, showObjectScale=7, drawMode=1, showColor=[0, 1, 0, 1])
    markers.addObject("BarycentricMapping" if args.legModel == "tetra" else "SkinningMapping")

    ##############################################################################
    # Emio Gui
    ##############################################################################

    # Open Loop
    if args.controller == "openloop":
        from scripts.openLoopController import OpenLoopController
        rootnode.addObject(OpenLoopController(
            leg, motor, markers, load,
            motorInit, motorMin, motorMax, float(args.motorCutoffFreq)))
    elif args.controller == "closedloop":
        from scripts.closedLoopController import ClosedLoopController
        rootnode.addObject(ClosedLoopController(
            leg, motor, markers, load,
            motorInit, motorMin, motorMax, float(args.motorCutoffFreq),
            int(args.order), int(args.useObserver)))

    if args.connection:
        # Add RealSense camera tracker
        try:
            from parts.controllers.motorcontroller import MotorController
            from parts.controllers.trackercontroller import DotTracker
            rootnode.addObject(MotorController([motor.JointActuator.displacement, None, None, None],
                                            name="MotorController"))

            tracker = DotTracker(name="DotTracker",
                                 root=rootnode,
                                 configuration="extended",
                                 nb_tracker=2,
                                 show_video_feed=False,
                                 track_colors=True,
                                 comp_point_cloud=False,
                                 scale=1)

            rootnode.addObject(tracker)
        except RuntimeError:
            Sofa.msg_error(__file__, "Camera not detected")

    return rootnode
