import dataclasses as dc
import typing

import numpy as np

from pydrake.all import (
    AddDefaultVisualization,
    Adder,
    AddMultibodyPlant,
    ApplyLcmBusConfig,
    ApplyMultibodyPlantConfig,    
    ApplyVisualizationConfig,
    CameraConfig,
    Demultiplexer,
    DiagramBuilder,
    DiscreteContactSolver,
    DrakeLcm,
    DrakeLcmParams,
    IiwaCommandSender,
    IiwaCommandReceiver,
    IiwaDriver,
    IiwaStatusSender,
    IiwaStatusReceiver,
    InverseDynamicsController,
    LcmInterfaceSystem,
    LcmPublisherSystem,
    LcmSubscriberSystem,
    MakeMultibodyStateToWsgStateSystem,
    ModelInstanceIndex,
    ModelDirective,
    ModelDirectives,
    MultibodyPlant,
    MultibodyPlantConfig,
    MultibodyPositionToGeometryPose,
    Parser,
    PassThrough,
    ProcessModelDirectives,
    SceneGraph,
    SchunkWsgDriver,
    SchunkWsgStatusSender,
    SchunkWsgPositionController,
    SimulatorConfig,
    StateInterpolatorWithDiscreteDerivative,
    VisualizationConfig,
    ZeroForceDriver,
)
from pydrake.common.yaml import yaml_load_typed
from drake import lcmt_iiwa_command, lcmt_iiwa_status

from manipulation.scenarios import (
    AddIiwa,
    AddPlanarIiwa,
    AddRgbdSensors,
    AddWsg,
)
from manipulation.utils import ConfigureParser


@dc.dataclass
class Scenario:
    """Defines the YAML format for a (possibly stochastic) scenario to be
    simulated.
    """

    # Random seed for any random elements in the scenario. The seed is always
    # deterministic in the `Scenario`; a caller who wants randomness must
    # populate this value from their own randomness.
    random_seed: int = 0

    # The maximum simulation time (in seconds).  The simulator will attempt to
    # run until this time and then terminate.
    simulation_duration: float = np.inf

    # Simulator configuration (integrator and publisher parameters).
    simulator_config: SimulatorConfig = SimulatorConfig(
        max_step_size=1e-3,
        accuracy=1.0e-2,
        target_realtime_rate=1.0)

    # Plant configuration (time step and contact parameters).
    plant_config: MultibodyPlantConfig = MultibodyPlantConfig()

    # All of the fully deterministic elements of the simulation.
    directives: typing.List[ModelDirective] = dc.field(default_factory=list)

    # A map of {bus_name: lcm_params} for LCM transceivers to be used by
    # drivers, sensors, etc.
    lcm_buses: typing.Mapping[str, DrakeLcmParams] = dc.field(
        default_factory=lambda: dict(default=DrakeLcmParams()))

    # For actuated models, specifies where each model's actuation inputs come
    # from, keyed on the ModelInstance name.
    model_drivers: typing.Mapping[str, typing.Union[
        IiwaDriver,
        SchunkWsgDriver,
        ZeroForceDriver,
    ]] = dc.field(default_factory=dict)

    # Cameras to add to the scene (and broadcast over LCM). The key for each
    # camera is a helpful mnemonic, but does not serve a technical role. The
    # CameraConfig::name field is still the name that will appear in the
    # Diagram artifacts.
    cameras: typing.Mapping[str, CameraConfig] = dc.field(default_factory=dict)

    visualization: VisualizationConfig = VisualizationConfig()


def load_scenario(*, filename, scenario_name, scenario_text=""):
    """Implements the command-line handling logic for scenario data.
    Returns a `Scenario` object loaded from the given input arguments.
    """
    result = yaml_load_typed(
        schema=Scenario,
        filename=filename,
        child_name=scenario_name,
        defaults=Scenario())
    if scenario_text:
        result = yaml_load_typed(
            schema=Scenario,
            data=scenario_text,
            defaults=result)
    return result

# TODO(russt): Use the c++ version pending https://github.com/RobotLocomotion/drake/issues/20055
def ApplyDriverConfig(driver_config,
            model_instance_name,
            sim_plant,
            models_from_directives_map,
            lcm_buses,
            builder):
    model_instance = sim_plant.GetModelInstanceByName(model_instance_name)
    if isinstance(driver_config, IiwaDriver):
        num_iiwa_positions = sim_plant.num_positions(model_instance)

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
            sim_plant.get_state_output_port(model_instance),
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
            sim_plant.get_state_output_port(model_instance),
            model_instance_name + "_state_estimated",
        )

        # Make the plant for the iiwa controller to use.
        controller_plant = MultibodyPlant(time_step=sim_plant.time_step())
        # TODO: Add the correct IIWA model (introspected from MBP)
        if num_iiwa_positions == 3:
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
            sim_plant.get_state_output_port(model_instance),
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
            sim_plant.get_actuation_input_port(model_instance),
        )

        # Add discrete derivative to command velocities.
        desired_state_from_position = builder.AddSystem(
            StateInterpolatorWithDiscreteDerivative(
                num_iiwa_positions,
                sim_plant.time_step(),
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
            sim_plant.get_generalized_contact_forces_output_port(
                model_instance
            ),
            model_instance_name + "_torque_external",
        )

    if isinstance(driver_config, SchunkWsgDriver):
        # Wsg controller.
        wsg_controller = builder.AddSystem(SchunkWsgPositionController())
        wsg_controller.set_name(model_instance_name + "_controller")
        builder.Connect(
            wsg_controller.get_generalized_force_output_port(),
            sim_plant.get_actuation_input_port(model_instance),
        )
        builder.Connect(
            sim_plant.get_state_output_port(model_instance),
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
            sim_plant.get_state_output_port(model_instance),
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


def ApplyDriverConfigs(*, driver_configs, sim_plant, models_from_directives,
                       lcm_buses, builder):
    models_from_directives_map = dict([
        (info.model_name, info)
        for info in models_from_directives
    ])
    for model_instance_name, driver_config in driver_configs.items():
        ApplyDriverConfig(
            driver_config,
            model_instance_name,
            sim_plant,
            models_from_directives_map,
            lcm_buses,
            builder)
        
def MakeHardwareStation(
    scenario: Scenario,
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

    # Create the multibody plant and scene graph.
    sim_plant, scene_graph = AddMultibodyPlant(
        config=scenario.plant_config,
        builder=builder)

    parser = Parser(sim_plant)
    for p in package_xmls:
        parser.package_map().AddPackageXml(p)
    ConfigureParser(parser)

    # Add model directives.
    added_models = ProcessModelDirectives(
        directives=ModelDirectives(directives=scenario.directives),
        parser=parser)

    # Now the plant is complete.
    sim_plant.Finalize()

    print(scenario.model_drivers)

    # Add actuation inputs.
    ApplyDriverConfigs(
        driver_configs=scenario.model_drivers,
        sim_plant=sim_plant,
        models_from_directives=added_models,
        lcm_buses=scenario.lcm_buses,
        builder=builder)

    # for system in builder.GetMutableSystems():
    #     if isinstance(system, (LcmInterfaceSystem, LcmSubscriberSystem,LcmPublisherSystem, IiwaCommandReceiver, IiwaStatusSender, SchunkWsgStatusSender)):
    #         builder.RemoveSystem(system)

    # TODO(russt): Add scene cameras. https://github.com/RobotLocomotion/drake/issues/20055

    # Add visualization.
    ApplyVisualizationConfig(scenario.visualization, builder, meshcat=meshcat)

    # Export "cheat" ports.
    builder.ExportOutput(scene_graph.get_query_output_port(), "query_object")
    builder.ExportOutput(
        sim_plant.get_contact_results_output_port(), "contact_results"
    )
    builder.ExportOutput(
        sim_plant.get_state_output_port(), "plant_continuous_state"
    )
    builder.ExportOutput(sim_plant.get_body_poses_output_port(), "body_poses")

    diagram = builder.Build()
    diagram.set_name("ManipulationStation")
    return diagram


def MakeHardwareStationInterface(
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
