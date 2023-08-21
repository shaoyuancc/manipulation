from pydrake.all import (
    Adder,
    AddMultibodyPlantSceneGraph,
    Demultiplexer,
    DiagramBuilder,
    InverseDynamicsController,
    MakeMultibodyStateToWsgStateSystem,
    ModelInstanceIndex,
    MultibodyPlant,
    Parser,
    PassThrough,
    SchunkWsgPositionController,
    StateInterpolatorWithDiscreteDerivative,
)

from manipulation.scenarios import (
    AddIiwa,
    AddPlanarIiwa,
    AddRgbdSensors,
    AddWsg,
)
from manipulation.utils import ConfigureParser


def MakeManipulationStation(
    model_directives=None,
    filename=None,
    time_step=0.002,
    iiwa_prefix="iiwa",
    wsg_prefix="wsg",
    camera_prefix="camera",
    prefinalize_callback=None,
    package_xmls=[],
):
    """
    Creates a manipulation station system, which is a sub-diagram containing:
      - A MultibodyPlant with populated via the Parser from the
        `model_directives` argument AND the `filename` argument.
      - A SceneGraph
      - For each model instance starting with `iiwa_prefix`, we add an
        additional iiwa controller system
      - For each model instance starting with `wsg_prefix`, we add an
        additional schunk controller system
      - For each body starting with `camera_prefix`, we add a RgbdSensor

    Args:
        builder: a DiagramBuilder

        model_directives: a string containing any model directives to be parsed

        filename: a string containing the name of an sdf, urdf, mujoco xml, or
        model directives yaml file.

        time_step: the standard MultibodyPlant time step.

        iiwa_prefix: Any model instances starting with `iiwa_prefix` will get
        an inverse dynamics controller, etc attached

        wsg_prefix: Any model instance starting with `wsg_prefix` will get a
        schunk controller

        camera_prefix: Any bodies in the plant (created during the
        plant_setup_callback) starting with this prefix will get a camera
        attached.

        prefinalize_callback: A function, setup(plant), that will be called
        with the multibody plant before calling finalize.  This can be useful
        for e.g. adding additional bodies/models to the simulation.

        package_xmls: A list of filenames to be passed to
        PackageMap.AddPackageXml().  This is useful if you need to add more
        models to your path (e.g. from your current working directory).
    """
    builder = DiagramBuilder()

    # Add (only) the iiwa, WSG, and cameras to the scene.
    plant, scene_graph = AddMultibodyPlantSceneGraph(
        builder, time_step=time_step
    )
    parser = Parser(plant)
    for p in package_xmls:
        parser.package_map().AddPackageXml(p)
    ConfigureParser(parser)
    if model_directives:
        parser.AddModelsFromString(model_directives, ".dmd.yaml")
    if filename:
        parser.AddModelsFromUrl(filename)
    if prefinalize_callback:
        prefinalize_callback(plant)
    plant.Finalize()

    for i in range(plant.num_model_instances()):
        model_instance = ModelInstanceIndex(i)
        model_instance_name = plant.GetModelInstanceName(model_instance)

        if model_instance_name.startswith(iiwa_prefix):
            num_iiwa_positions = plant.num_positions(model_instance)

            # I need a PassThrough system so that I can export the input port.
            iiwa_position = builder.AddSystem(PassThrough(num_iiwa_positions))
            builder.ExportInput(
                iiwa_position.get_input_port(),
                model_instance_name + "_position",
            )
            builder.ExportOutput(
                iiwa_position.get_output_port(),
                model_instance_name + "_position_commanded",
            )

            # Export the iiwa "state" outputs.
            demux = builder.AddSystem(
                Demultiplexer(2 * num_iiwa_positions, num_iiwa_positions)
            )
            builder.Connect(
                plant.get_state_output_port(model_instance),
                demux.get_input_port(),
            )
            builder.ExportOutput(
                demux.get_output_port(0),
                model_instance_name + "_position_measured",
            )
            builder.ExportOutput(
                demux.get_output_port(1),
                model_instance_name + "_velocity_estimated",
            )
            builder.ExportOutput(
                plant.get_state_output_port(model_instance),
                model_instance_name + "_state_estimated",
            )

            # Make the plant for the iiwa controller to use.
            controller_plant = MultibodyPlant(time_step=time_step)
            # TODO: Add the correct IIWA model (introspected from MBP)
            if plant.num_positions(model_instance) == 3:
                controller_iiwa = AddPlanarIiwa(controller_plant)
            else:
                controller_iiwa = AddIiwa(controller_plant)
            AddWsg(controller_plant, controller_iiwa, welded=True)
            controller_plant.Finalize()

            # Add the iiwa controller
            iiwa_controller = builder.AddSystem(
                InverseDynamicsController(
                    controller_plant,
                    kp=[100] * num_iiwa_positions,
                    ki=[1] * num_iiwa_positions,
                    kd=[20] * num_iiwa_positions,
                    has_reference_acceleration=False,
                )
            )
            iiwa_controller.set_name(model_instance_name + "_controller")
            builder.Connect(
                plant.get_state_output_port(model_instance),
                iiwa_controller.get_input_port_estimated_state(),
            )

            # Add in the feed-forward torque
            adder = builder.AddSystem(Adder(2, num_iiwa_positions))
            builder.Connect(
                iiwa_controller.get_output_port_control(),
                adder.get_input_port(0),
            )
            # Use a PassThrough to make the port optional (it will provide zero
            # values if not connected).
            torque_passthrough = builder.AddSystem(
                PassThrough([0] * num_iiwa_positions)
            )
            builder.Connect(
                torque_passthrough.get_output_port(), adder.get_input_port(1)
            )
            builder.ExportInput(
                torque_passthrough.get_input_port(),
                model_instance_name + "_feedforward_torque",
            )
            builder.Connect(
                adder.get_output_port(),
                plant.get_actuation_input_port(model_instance),
            )

            # Add discrete derivative to command velocities.
            desired_state_from_position = builder.AddSystem(
                StateInterpolatorWithDiscreteDerivative(
                    num_iiwa_positions,
                    time_step,
                    suppress_initial_transient=True,
                )
            )
            desired_state_from_position.set_name(
                model_instance_name + "_desired_state_from_position"
            )
            builder.Connect(
                desired_state_from_position.get_output_port(),
                iiwa_controller.get_input_port_desired_state(),
            )
            builder.Connect(
                iiwa_position.get_output_port(),
                desired_state_from_position.get_input_port(),
            )

            # Export commanded torques.
            builder.ExportOutput(
                adder.get_output_port(),
                model_instance_name + "_torque_commanded",
            )
            builder.ExportOutput(
                adder.get_output_port(),
                model_instance_name + "_torque_measured",
            )

            builder.ExportOutput(
                plant.get_generalized_contact_forces_output_port(
                    model_instance
                ),
                model_instance_name + "_torque_external",
            )

        elif model_instance_name.startswith(wsg_prefix):
            # Wsg controller.
            wsg_controller = builder.AddSystem(SchunkWsgPositionController())
            wsg_controller.set_name(model_instance_name + "_controller")
            builder.Connect(
                wsg_controller.get_generalized_force_output_port(),
                plant.get_actuation_input_port(model_instance),
            )
            builder.Connect(
                plant.get_state_output_port(model_instance),
                wsg_controller.get_state_input_port(),
            )
            builder.ExportInput(
                wsg_controller.get_desired_position_input_port(),
                model_instance_name + "_position",
            )
            builder.ExportInput(
                wsg_controller.get_force_limit_input_port(),
                model_instance_name + "_force_limit",
            )
            wsg_mbp_state_to_wsg_state = builder.AddSystem(
                MakeMultibodyStateToWsgStateSystem()
            )
            builder.Connect(
                plant.get_state_output_port(model_instance),
                wsg_mbp_state_to_wsg_state.get_input_port(),
            )
            builder.ExportOutput(
                wsg_mbp_state_to_wsg_state.get_output_port(),
                model_instance_name + "_state_measured",
            )
            builder.ExportOutput(
                wsg_controller.get_grip_force_output_port(),
                model_instance_name + "_force_measured",
            )

    # Cameras.
    AddRgbdSensors(
        builder, plant, scene_graph, model_instance_prefix=camera_prefix
    )

    # Export "cheat" ports.
    builder.ExportOutput(scene_graph.get_query_output_port(), "query_object")
    builder.ExportOutput(
        plant.get_contact_results_output_port(), "contact_results"
    )
    builder.ExportOutput(
        plant.get_state_output_port(), "plant_continuous_state"
    )
    builder.ExportOutput(plant.get_body_poses_output_port(), "body_poses")

    diagram = builder.Build()
    diagram.set_name("ManipulationStation")
    return diagram
