from pydrake.all import (
    AddDefaultVisualization,
    Adder,
    AddMultibodyPlantSceneGraph,
    ApplyVisualizationConfig,
    Demultiplexer,
    DiagramBuilder,
    DiscreteContactSolver,
    DrakeLcm,
    IiwaCommandSender,
    IiwaStatusReceiver,
    InverseDynamicsController,
    LcmInterfaceSystem,
    LcmPublisherSystem,
    LcmSubscriberSystem,
    MakeMultibodyStateToWsgStateSystem,
    ModelInstanceIndex,
    MultibodyPlant,
    MultibodyPositionToGeometryPose,
    Parser,
    PassThrough,
    SceneGraph,
    SchunkWsgPositionController,
    StateInterpolatorWithDiscreteDerivative,
    VisualizationConfig,
)
from drake import lcmt_iiwa_command, lcmt_iiwa_status

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
    meshcat=None,
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

        meshcat: If not None, then AddDefaultVisualization will be added to the subdiagram using this meshcat instance.
    """
    builder = DiagramBuilder()

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
    plant.set_discrete_contact_solver(DiscreteContactSolver.kSap)
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

    if meshcat:
        AddDefaultVisualization(builder, meshcat=meshcat)

    diagram = builder.Build()
    diagram.set_name("ManipulationStation")
    return diagram


def MakeManipulationStationHardwareInterface(
    model_directives=None,
    filename=None,
    time_step=0.002,
    iiwa_prefix="iiwa",
    wsg_prefix="wsg",
    camera_prefix="camera",
    prefinalize_callback=None,
    package_xmls=[],
    meshcat=None,
):
    builder = DiagramBuilder()

    lcm = DrakeLcm()
    lcm_system = builder.AddNamedSystem("lcm", LcmInterfaceSystem(lcm=lcm))

    # Visualization
    scene_graph = builder.AddNamedSystem("scene_graph", SceneGraph())
    plant = MultibodyPlant(time_step=0.0)
    plant.RegisterAsSourceForSceneGraph(scene_graph)
    Parser(plant).AddModelsFromString(model_directives, ".dmd.yaml")
    plant.Finalize()
    to_pose = builder.AddSystem(MultibodyPositionToGeometryPose(plant))
    builder.Connect(
        to_pose.get_output_port(),
        scene_graph.get_source_pose_port(plant.get_source_id()),
    )

    config = VisualizationConfig()
    config.publish_contacts = False
    config.publish_inertia = False
    ApplyVisualizationConfig(
        config, builder=builder, plant=plant, meshcat=meshcat
    )

    iiwa_model_instance = None
    wsg_model_instance = None
    for i in range(plant.num_model_instances()):
        model_instance = ModelInstanceIndex(i)
        model_instance_name = plant.GetModelInstanceName(model_instance)

        if model_instance_name.startswith(iiwa_prefix):
            assert (
                not iiwa_model_instance
            ), "Still need to support multiple iiwas here"
            iiwa_model_instance = model_instance

        if model_instance_name.startswith(wsg_prefix):
            assert (
                not wsg_model_instance
            ), "Still need to support multiple WSGs here"
            wsg_model_instance = model_instance

    # Publish IIWA command.
    iiwa_command_sender = builder.AddSystem(IiwaCommandSender())
    # Note on publish period: IIWA driver won't respond faster than 200Hz
    iiwa_command_publisher = builder.AddSystem(
        LcmPublisherSystem.Make(
            channel="IIWA_COMMAND",
            lcm_type=lcmt_iiwa_command,
            lcm=lcm,
            publish_period=0.005,
            use_cpp_serializer=True,
        )
    )
    builder.ExportInput(
        iiwa_command_sender.get_position_input_port(), "iiwa_position"
    )
    builder.ExportInput(
        iiwa_command_sender.get_torque_input_port(), "iiwa_feedforward_torque"
    )
    builder.Connect(
        iiwa_command_sender.get_output_port(),
        iiwa_command_publisher.get_input_port(),
    )

    # Receive IIWA status and populate the output ports.
    iiwa_status_receiver = builder.AddSystem(IiwaStatusReceiver())
    iiwa_status_subscriber = builder.AddSystem(
        LcmSubscriberSystem.Make(
            channel="IIWA_STATUS",
            lcm_type=lcmt_iiwa_status,
            lcm=lcm,
            use_cpp_serializer=True,
            wait_for_message_on_initialization_timeout=10,
        )
    )

    assert not wsg_model_instance, "Still need to support WSG"
    builder.Connect(
        iiwa_status_receiver.get_position_measured_output_port(),
        to_pose.get_input_port(),
    )

    builder.ExportOutput(
        iiwa_status_receiver.get_position_commanded_output_port(),
        "iiwa_position_commanded",
    )
    builder.ExportOutput(
        iiwa_status_receiver.get_position_measured_output_port(),
        "iiwa_position_measured",
    )
    builder.ExportOutput(
        iiwa_status_receiver.get_velocity_estimated_output_port(),
        "iiwa_velocity_estimated",
    )
    builder.ExportOutput(
        iiwa_status_receiver.get_torque_commanded_output_port(),
        "iiwa_torque_commanded",
    )
    builder.ExportOutput(
        iiwa_status_receiver.get_torque_measured_output_port(),
        "iiwa_torque_measured",
    )
    builder.ExportOutput(
        iiwa_status_receiver.get_torque_external_output_port(),
        "iiwa_torque_external",
    )
    builder.Connect(
        iiwa_status_subscriber.get_output_port(),
        iiwa_status_receiver.get_input_port(),
    )

    return builder.Build()
