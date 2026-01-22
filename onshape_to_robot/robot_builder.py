import numpy as np
import os
import hashlib
import json
import fnmatch
from .geometry import Mesh
from .message import warning, info, success, error, dim, bright
from .assembly import Assembly
from .config import Config
from .robot import Part, Joint, Link, Robot, Relation, Closure
from .csg import process as csg_process


class RobotBuilder:
    def __init__(self, config: Config):
        self.config: Config = config
        self.assembly: Assembly = Assembly(config)
        self.robot: Robot = Robot(config.robot_name)

        for closure_type, frame1, frame2 in self.assembly.closures:
            self.robot.closures.append(Closure(closure_type, frame1, frame2))

        self.unique_names = {}
        self.stl_filenames: dict = {}
        
        # NEW: Store part colors for Isaac Sim color application
        self.part_colors: dict = {}  # part_name -> [r, g, b] (0-1 range)

        for node in self.assembly.root_nodes:
            link = self.build_robot(node)
            self.robot.base_links.append(link)
        
        # NEW: Save color mapping file after building robot
        self.save_color_mapping()

    def part_is_ignored(self, name: str, what: str) -> bool:
        """
        Checks if a given part should be ignored by config
        """
        ignored = False

        # Removing <1>, <2> etc. suffix
        name = "<".join(name.split("<")[:-1]).strip()

        for entry in self.config.ignore:
            to_ignore = True
            match_entry = entry
            if entry[0] == "!":
                to_ignore = False
                match_entry = entry[1:]

            if fnmatch.fnmatch(name.lower(), match_entry.lower()):
                if (
                    self.config.ignore[entry] == "all"
                    or self.config.ignore[entry] == what
                ):
                    ignored = to_ignore

        return ignored

    def slugify(self, value: str) -> str:
        """
        Turns a value into a slug that is valid for USD/SdfPath.
        USD paths cannot start with a digit, so prefix if needed.
        """
        slug = "".join(c if c.isalnum() else "_" for c in value).strip("_")
        # USD paths cannot start with a digit - prefix with "part_" if needed
        if slug and slug[0].isdigit():
            slug = "part_" + slug
        return slug

    def printable_configuration(self, instance: dict) -> str:
        """
        Retrieve configuration enums to replace "List_..." with proper enum names
        """
        configuration = instance["configuration"]

        if instance["configuration"] != "default":
            if "documentVersion" in instance:
                version = instance["documentVersion"]
                wmv = "v"
            else:
                version = instance["documentMicroversion"]
                wmv = "m"
            elements = self.assembly.client.elements_configuration(
                instance["documentId"],
                version,
                instance["elementId"],
                wmv=wmv,
                linked_document_id=self.config.document_id,
            )
            for entry in elements["configurationParameters"]:
                type_name = entry["typeName"]
                message = entry["message"]

                if type_name.startswith("BTMConfigurationParameterEnum"):
                    parameter_name = message["parameterName"]
                    parameter_id = message["parameterId"]
                    configuration = configuration.replace(parameter_id, parameter_name)

        return configuration

    def part_name(self, part: dict, include_configuration: bool = False) -> str:
        """
        Retrieve the name from a part.
        i.e "Base link <1>" -> "base_link"
        """
        name = part["name"]
        parts = name.split(" ")
        del parts[-1]
        base_part_name = self.slugify("_".join(parts).lower())

        if not include_configuration:
            return base_part_name

        # Only add configuration to name if its not default and not a very long configuration (which happens for library parts like screws)
        configuration = self.printable_configuration(part)
        if configuration != "default" and self.config.include_configuration_suffix:
            if len(configuration) < 40:
                parts += ["_" + configuration.replace("=", "_").replace(" ", "_")]
            else:
                parts += ["_" + hashlib.md5(configuration.encode("utf-8")).hexdigest()]

        return self.slugify("_".join(parts).lower())

    def unique_name(self, part: dict, type: str):
        """
        Get unique part name (plate, plate_2, plate_3, ...)
        In the case where multiple parts have the same name in Onshape, they will result in different names in the URDF
        """
        while True:
            name = self.part_name(part, include_configuration=True)

            if type not in self.unique_names:
                self.unique_names[type] = {}

            if name in self.unique_names[type]:
                self.unique_names[type][name] += 1
                name = f"{name}_{self.unique_names[type][name]}"
            else:
                self.unique_names[type][name] = 1
                name = name

            if name not in [frame.name for frame in self.assembly.frames]:
                return name

    def instance_request_params(self, instance: dict) -> dict:
        """
        Build parameters to make an API call for a given instance
        """
        params = {}

        if "documentVersion" in instance:
            params["wmvid"] = instance["documentVersion"]
            params["wmv"] = "v"
        else:
            params["wmvid"] = instance["documentMicroversion"]
            params["wmv"] = "m"

        params["did"] = instance["documentId"]
        params["eid"] = instance["elementId"]
        params["linked_document_id"] = self.config.document_id
        params["configuration"] = instance["configuration"]

        return params

    def get_stl_filename(self, instance: dict) -> str:
        """
        Get a STL filename unique to the instance
        """
        exact_instance = (
            instance["documentId"],
            instance["documentMicroversion"],
            instance["elementId"],
            instance["configuration"],
            instance["partId"],
        )

        if exact_instance not in self.stl_filenames:
            part_name_config = self.part_name(instance, True)
            stl_filename = part_name_config
            k = 1
            while stl_filename in self.stl_filenames.values():
                k += 1
                stl_filename = part_name_config + f"__{k}"
            if k != 1:
                print(
                    warning(
                        f'WARNING: Parts with same name "{part_name_config}", incrementing STL name to "{stl_filename}"'
                    )
                )
            self.stl_filenames[exact_instance] = stl_filename

        return self.stl_filenames[exact_instance]

    def get_stl(self, instance: dict) -> str:
        """
        Download and store STL file
        """
        os.makedirs(self.config.asset_path(""), exist_ok=True)

        stl_filename = self.get_stl_filename(instance)
        filename = stl_filename + ".stl"

        params = self.instance_request_params(instance)
        stl = self.assembly.client.part_studio_stl_m(
            **params,
            partid=instance["partId"],
        )
        with open(self.config.asset_path(filename), "wb") as stream:
            stream.write(stl)

        # Storing metadata for imported instances in the .part file
        stl_metadata = stl_filename + ".part"
        with open(
            self.config.asset_path(stl_metadata), "w", encoding="utf-8"
        ) as stream:
            json.dump(instance, stream, indent=4, sort_keys=True)

        return self.config.asset_path(filename)

    def get_color(self, instance: dict) -> np.ndarray:
        """
        Retrieve the color of a part.
        DEFAULT: Blue if no color specified (common for robotics visualization).
        """
        if self.config.color is not None:
            color = np.array(self.config.color)
        else:
            params = self.instance_request_params(instance)
            metadata = self.assembly.client.part_get_metadata(
                **params,
                partid=instance["partId"],
            )

            # Default to nice blue color for robotics (instead of gray)
            color = np.array([0.3, 0.5, 0.8, 1.0])  # Professional blue

            # XXX: There must be a better way to retrieve the part color
            for entry in metadata["properties"]:
                if (
                    "value" in entry
                    and type(entry["value"]) is dict
                    and "color" in entry["value"]
                ):
                    rgb = entry["value"]["color"]
                    a = entry["value"]["opacity"]
                    color = np.array([rgb["red"], rgb["green"], rgb["blue"], a]) / 255.0

        return color

    def get_dynamics(self, instance: dict) -> tuple:
        """
        Retrieve the dynamics (mass, com, inertia) of a given instance
        """
        if self.config.no_dynamics:
            mass = 0
            com = [0] * 3
            inertia = [0] * 12
        else:
            if instance["isStandardContent"]:
                mass_properties = self.assembly.client.standard_cont_mass_properties(
                    instance["documentId"],
                    instance["documentVersion"],
                    instance["elementId"],
                    instance["partId"],
                    configuration=instance["configuration"],
                    linked_document_id=self.config.document_id,
                )
            else:
                params = self.instance_request_params(instance)
                mass_properties = self.assembly.client.part_mass_properties(
                    **params,
                    partid=instance["partId"],
                )

            if instance["partId"] not in mass_properties["bodies"]:
                print(
                    warning(
                        f"WARNING: part {instance['name']} has no dynamics (maybe it is a surface)"
                    )
                )
                return
            mass_properties = mass_properties["bodies"][instance["partId"]]
            mass = mass_properties["mass"][0]
            com = mass_properties["centroid"]
            inertia = mass_properties["inertia"]

            if abs(mass) < 1e-9:
                # Default to aluminum material properties (no mass from Onshape API)
                # Aluminum density: 2700 kg/m³
                # For safety, use small default mass to avoid simulation instability
                mass = 0.1  # kg (100 grams - reasonable for small parts)
                # Keep COM at origin since we don't have geometry volume
                com = [0, 0, 0]
                # Simple default inertia for small mass
                inertia = [[1e-4, 0, 0], [0, 1e-4, 0], [0, 0, 1e-4]]

        return mass, com[:3], np.reshape(np.array(inertia).flatten()[:9], (3, 3))

    def add_part(self, occurrence: dict):
        """
        Add a part to the current link
        """
        instance = occurrence["instance"]

        if instance["suppressed"]:
            return

        if instance["partId"] == "":
            print(warning(f"WARNING: Part '{instance['name']}' has no partId"))
            return

        part_name = instance["name"]
        extra = ""
        if instance["configuration"] != "default":
            extra = dim(
                " (configuration: " + self.printable_configuration(instance) + ")"
            )
        symbol = "+"
        if self.part_is_ignored(part_name, "visual") or self.part_is_ignored(
            part_name, "collision"
        ):
            symbol = "-"
            extra += dim(" / ")
            if self.part_is_ignored(part_name, "visual"):
                extra += dim("(ignoring visual)")
            if self.part_is_ignored(part_name, "collision"):
                extra += dim(" (ignoring collision)")

        print(success(f"{symbol} Adding part {part_name}{extra}"))

        if self.part_is_ignored(part_name, "visual") and self.part_is_ignored(
            part_name, "collision"
        ):
            stl_file = None
        else:
            stl_file = self.get_stl(instance)

        # Obtain metadatas about part to retrieve color
        color = self.get_color(instance)
        
        # NEW: Store color for this part (use clean part name without instance suffix)
        clean_part_name = self.part_name(instance, include_configuration=False)
        self.part_colors[clean_part_name] = color[:3].tolist()  # Store RGB only (0-1 range)

        # Obtain the instance dynamics
        mass, com, inertia = self.get_dynamics(instance)

        # Obtain part pose
        T_world_part = np.array(occurrence["transform"]).reshape(4, 4)

        # Adding non-ignored meshes
        meshes = []
        mesh = Mesh(stl_file, color)
        if self.part_is_ignored(part_name, "visual"):
            mesh.visual = False
        if self.part_is_ignored(part_name, "collision"):
            mesh.collision = False
        if mesh.visual or mesh.collision:
            meshes.append(mesh)

        part = Part(
            self.unique_name(instance, "part"),
            T_world_part,
            mass,
            com,
            inertia,
            meshes,
        )

        self.robot.links[-1].parts.append(part)

    def build_robot(self, body_id: int):
        """
        Add recursively body nodes to the robot description.
        """
        instance = self.assembly.body_instance(body_id)

        if body_id in self.assembly.link_names:
            link_name = self.assembly.link_names[body_id]
        else:
            link_name = self.unique_name(instance, "link")

        # Adding all the parts in the current link
        link = Link(link_name)
        self.robot.links.append(link)
        for occurrence in self.assembly.body_occurrences(body_id):
            if occurrence["instance"]["type"] == "Part":
                self.add_part(occurrence)
            if occurrence["fixed"]:
                link.fixed = True

        # Adding frames to the link
        for frame in self.assembly.frames:
            if frame.body_id == body_id:
                self.robot.links[-1].frames[frame.name] = frame.T_world_frame

        for children_body in self.assembly.tree_children[body_id]:
            dof = self.assembly.get_dof(body_id, children_body)
            child_body = dof.other_body(body_id)
            T_world_axis = dof.T_world_mate.copy()

            properties = self.config.joint_properties.get("default", {})
            for joint_name in self.config.joint_properties:
                if fnmatch.fnmatch(dof.name, joint_name):
                    properties = {
                        **properties,
                        **self.config.joint_properties[joint_name],
                    }

            joint = Joint(
                dof.name,
                dof.joint_type,
                link,
                None,
                T_world_axis,
                properties,
                dof.limits,
                dof.axis,
            )
            if dof.name in self.assembly.relations:
                source, ratio = self.assembly.relations[dof.name]
                joint.relation = Relation(source, ratio)

            # The joint is added before the recursive call, ensuring items in robot.joints has the
            # same order as recursive calls on the tree
            self.robot.joints.append(joint)

            joint.child = self.build_robot(child_body)

        return link
    
    def save_color_mapping(self):
        """
        Save part colors to JSON file for Isaac Sim automatic color application.
        Colors are stored in 0-1 RGB format matching Isaac Sim's expected input.
        """
        if not self.part_colors:
            print(info("  No colors captured (all parts may be using default colors)"))
            return
        
        color_file_path = self.config.asset_path("colors.json")
        
        # Create color mapping with metadata
        color_mapping = {
            "_metadata": {
                "description": "Part colors extracted from Onshape assembly",
                "format": "RGB values in 0-1 range (Isaac Sim format)",
                "robot_name": self.config.robot_name,
                "assembly_name": getattr(self.config, 'assembly_name', 'unknown')
            },
            "colors": self.part_colors
        }
        
        with open(color_file_path, 'w', encoding='utf-8') as f:
            json.dump(color_mapping, f, indent=2, sort_keys=True)
        
        print(success(f"✓ Saved {len(self.part_colors)} part colors to colors.json"))
        
        # Print a sample of captured colors for verification
        if self.part_colors:
            print(bright("  Color samples:"))
            for part_name, rgb in list(self.part_colors.items())[:5]:
                rgb_255 = [int(c * 255) for c in rgb]
                print(info(f"    {part_name}: RGB{tuple(rgb_255)} (0-255) = {rgb} (0-1)"))
            if len(self.part_colors) > 5:
                print(info(f"    ... and {len(self.part_colors) - 5} more"))
