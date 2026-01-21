
from __future__ import annotations
import numpy as np
from .config import Config
from .message import error, info, bright, success, warning
from .onshape_api.client import Client
from .robot import Joint
from .expression import ExpressionParser

INSTANCE_IGNORE = -1


class Frame:
    """
    Represents a frame attached
    """

    def __init__(self, body_id: int, name: str, T_world_frame: np.ndarray):
        self.body_id: int = body_id
        self.name: str = name
        self.T_world_frame: np.ndarray = T_world_frame


class DOF:
    """
    Represents a DOF
    """

    def __init__(
        self,
        body1_id: int,
        body2_id: int,
        name: str,
        joint_type: str,
        T_world_mate: np.ndarray,
        limits: tuple | None,
        axis: np.ndarray = np.array([0.0, 0.0, 1.0]),
    ):
        self.body1_id: int = body1_id
        self.body2_id: int = body2_id
        self.name: str = name
        self.joint_type: str = joint_type
        self.T_world_mate: np.ndarray = T_world_mate
        self.limits: tuple | None = limits
        self.axis: np.ndarray = axis

    def flip(self, flip_limits: bool = True):
        if flip_limits and self.limits is not None:
            self.limits = (-self.limits[1], -self.limits[0])

        # Flipping the joint around X axis
        flip = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        self.T_world_mate[:3, :3] = self.T_world_mate[:3, :3] @ flip

    def other_body(self, body_id: int):
        if body_id == self.body1_id:
            return self.body2_id
        elif body_id == self.body2_id:
            return self.body1_id
        else:
            raise Exception(f"ERROR: body {body_id} is not part of this DOF")


class Assembly:
    """
    Main entry point to process an assembly
    """

    def __init__(self, config: Config):
        self.config: Config = config

        # Creating Onshape API client
        self.client = Client(logging=False, creds=self.config.config_file)
        self.expression_parser = ExpressionParser()
        self.expression_parser.variables_lazy_loading = self.load_variables

        self.document_id: str = config.document_id
        self.workspace_id: str | None = config.workspace_id
        self.version_id: str | None = config.version_id

        # All (raw) data from assembly
        self.assembly_data: dict = {}
        # Map a (top-level) instance id to a body id
        self.current_body_id: int = 0
        self.instance_body: dict[str, int] = {}
        # Frames object
        self.frames: list[Frame] = []
        # Loop closure constraints
        self.closures: list = []
        # Degrees of freedom
        self.dofs: list[DOF] = []
        # Features data
        self.features: dict = {}
        # Configuration values
        self.configuration_parameters: dict = {}
        # Dictionnary mapping items to their children in the tree
        self.tree_children: dict = {}
        # Root nodes
        self.root_nodes: list = []
        # Overriden link names
        self.link_names: dict[int, str] = {}
        # Relation indexed by target joints, values are [source joint, ratio]
        self.relations: dict = {}

        self.ensure_workspace_or_version()
        self.find_assembly()
        self.check_configuration()
        self.retrieve_assembly()
        self.find_instances()
        self.load_features()
        self.load_configuration()
        self.process_mates()
        self.build_trees()
        self.find_relations()
        print("")

    def ensure_workspace_or_version(self):
        """
        Ensure either a workspace id or a version id is set
        If none, try to retrieve the current workspace ID from API
        """
        if self.version_id:
            print(bright(f"* Using configuration version ID {self.version_id} ..."))
        elif self.workspace_id:
            print(bright(f"* Using configuration workspace ID {self.workspace_id} ..."))
        else:
            print(
                bright(
                    "* Not workspace ID specified, retrieving the current workspace ..."
                )
            )
            document = self.client.get_document(self.config.document_id)
            self.workspace_id = document["defaultWorkspace"]["id"]
            print(success(f"+ Using workspace id: {self.workspace_id}"))

    def find_assembly(self):
        """
        Find the wanted assembly from the document
        """
        if self.config.element_id:
            print(
                bright(f"* Using configuration element ID {self.config.element_id} ...")
            )
            self.element_id = self.config.element_id
            return

        print(
            bright(
                "\n* Retrieving elements in the document, searching for the assembly..."
            )
        )

        elements = self.client.list_elements(
            self.document_id,
            self.version_id if self.version_id else self.workspace_id,
            "v" if self.version_id else "w",
        )

        self.element_id = None
        assemblies: dict = {}
        for element in elements:
            if element["type"] == "Assembly":
                assemblies[element["name"]] = element["id"]

        if self.config.assembly_name:
            if self.config.assembly_name in assemblies:
                self.element_id = assemblies[self.config.assembly_name]
            else:
                raise Exception(
                    f"ERROR: Unable to find required assembly {self.config.assembly_name} in this document"
                )
        else:
            if len(assemblies) == 0:
                raise Exception("ERROR: No assembly found in this document\n")
            elif len(assemblies) == 1:
                self.element_id = list(assemblies.values())[0]
            else:
                raise Exception(
                    f"ERROR: Multiple assemblies found, please specify the assembly name\n"
                    + '       to export (use "assemblyName" in the configuration file)\n'
                    + f"       Available assemblies: {', '.join(assemblies.keys())}"
                )

        if self.element_id == None:
            raise Exception(f"ERROR: Unable to find assembly in this document")

    def check_configuration(self):
        """
        Retrieve configuration items for given assembly and parsing config configuration
        """

        if self.config.configuration != "default":
            # Retrieving available config parameters
            elements = self.client.elements_configuration(
                self.document_id,
                self.version_id if self.version_id else self.workspace_id,
                self.element_id,
                wmv=("v" if self.version_id else "w"),
            )

            parameters = {}
            for entry in elements["configurationParameters"]:
                type_name = entry["typeName"]
                message = entry["message"]

                if type_name.startswith("BTMConfigurationParameterEnum"):
                    # The very first label typed is kept as the internal name for the enum, under the "option"
                    # key. However, the user label that can be changed later is "optionName"
                    option_names = [
                        option["message"]["optionName"] for option in message["options"]
                    ]
                    options = [
                        option["message"]["option"] for option in message["options"]
                    ]
                    parameters[message["parameterName"]] = [
                        "enum",
                        message["parameterId"],
                        option_names,
                        options,
                    ]
                elif type_name.startswith("BTMConfigurationParameterBoolean"):
                    parameters[message["parameterName"]] = ["bool"]
                elif type_name.startswith("BTMConfigurationParameterQuantity"):
                    parameters[message["parameterName"]] = ["quantity"]

            # Parsing configuration
            parts = self.config.configuration.split(";")
            processed_configuration = []
            for part in parts:
                kv = part.split("=")
                if len(kv) == 2:
                    key, value = kv
                    if key not in parameters:
                        raise Exception(
                            f'ERROR: Unknown configuration parameter "{key}" in the configuration'
                        )
                    if parameters[key][0] == "enum":
                        if value not in parameters[key][2]:
                            raise Exception(
                                f'ERROR: Unknown value "{value}" for configuration parameter "{key}"'
                            )

                        value = parameters[key][3][parameters[key][2].index(value)]
                        key = parameters[key][1]
                    processed_configuration.append(f"{key}={value.replace(' ', '+')}")

            # Re-writing the configuration
            self.config.configuration = ";".join(processed_configuration)

    def retrieve_assembly(self):
        """
        Retrieve all assembly data
        """
        print(bright(f"* Retrieving assembly with id {self.element_id}"))

        self.assembly_data: dict = self.client.get_assembly(
            self.document_id,
            self.version_id if self.version_id else self.workspace_id,
            self.element_id,
            wmv=("v" if self.version_id else "w"),
            configuration=self.config.configuration,
        )

        self.microversion_id: str = self.assembly_data["rootAssembly"][
            "documentMicroversion"
        ]
        self.occurrences: dict = {}
        for occurrence in self.assembly_data["rootAssembly"]["occurrences"]:
            self.occurrences[tuple(occurrence["path"])] = occurrence
        
        # Build comprehensive mapping for subassembly path resolution
        self._build_subassembly_mappings()
        
        # RECURSIVE: Fetch full assembly data for each subassembly independently
        # This gives us the subassembly's own occurrence list with LOCAL IDs that match feature references
        self._fetch_all_subassemblies_recursively()
        
        # Debug: Print subassembly count
        subasm_count = len(self.assembly_data.get("subAssemblies", []))
        print(bright(f"* Assembly data loaded: {subasm_count} subassemblies, {len(self.occurrence_id_to_path)} mapped occurrences"))

    def _build_subassembly_mappings(self):
        """
        Build mappings needed for recursive subassembly path resolution.
        Handles infinite nesting by using the flat occurrence list.
        """
        # Map leaf occurrence ID → full path from root
        self.occurrence_id_to_path: dict = {}
        for occurrence in self.assembly_data["rootAssembly"]["occurrences"]:
            path = occurrence["path"]
            leaf_id = path[-1] if path else None
            if leaf_id:
                self.occurrence_id_to_path[leaf_id] = path
        
        # Map subassembly definition key → list of (prefix_path, instance_id) tuples
        # This tells us WHERE each subassembly is instantiated at any nesting level
        self.subassembly_instances: dict = {}
        
        # Helper: get subassembly key from definition
        def sub_key(sub):
            return (sub.get("documentId"), sub.get("elementId"), 
                    sub.get("configuration"), sub.get("documentMicroversion"))
        
        # First pass: Find top-level subassembly instances (in rootAssembly)
        for instance in self.assembly_data["rootAssembly"]["instances"]:
            if instance.get("type") == "Assembly" and not instance.get("suppressed", False):
                key = (instance.get("documentId"), instance.get("elementId"),
                       instance.get("configuration"), instance.get("documentMicroversion"))
                if key not in self.subassembly_instances:
                    self.subassembly_instances[key] = []
                # Top-level: prefix is empty, instance ID is the path root
                self.subassembly_instances[key].append(([], instance["id"]))
        
        # Second pass: Find nested subassembly instances (within other subassemblies)
        for sub_assembly in self.assembly_data.get("subAssemblies", []):
            parent_key = sub_key(sub_assembly)
            
            # Get prefix paths where this parent subassembly is instantiated
            parent_locations = self.subassembly_instances.get(parent_key, [])
            
            # Check instances within this subassembly for nested subassemblies
            for instance in sub_assembly.get("instances", []):
                if instance.get("type") == "Assembly" and not instance.get("suppressed", False):
                    child_key = (instance.get("documentId"), instance.get("elementId"),
                                 instance.get("configuration"), instance.get("documentMicroversion"))
                    if child_key not in self.subassembly_instances:
                        self.subassembly_instances[child_key] = []
                    
                    # For each location where parent is instantiated, 
                    # the child is at: parent_prefix + [parent_instance_id] + child_instance_id
                    for prefix, parent_inst_id in parent_locations:
                        new_prefix = prefix + [parent_inst_id]
                        self.subassembly_instances[child_key].append((new_prefix, instance["id"]))
        
        # Keep iterating until no new nested subassemblies are found (handles deeper nesting)
        changed = True
        max_iterations = 10  # Prevent infinite loops
        iteration = 0
        while changed and iteration < max_iterations:
            changed = False
            iteration += 1
            
            for sub_assembly in self.assembly_data.get("subAssemblies", []):
                parent_key = sub_key(sub_assembly)
                parent_locations = self.subassembly_instances.get(parent_key, [])
                
                for instance in sub_assembly.get("instances", []):
                    if instance.get("type") == "Assembly" and not instance.get("suppressed", False):
                        child_key = (instance.get("documentId"), instance.get("elementId"),
                                     instance.get("configuration"), instance.get("documentMicroversion"))
                        
                        for prefix, parent_inst_id in parent_locations:
                            new_prefix = prefix + [parent_inst_id]
                            new_entry = (new_prefix, instance["id"])
                            
                            # Check if this entry already exists
                            if child_key not in self.subassembly_instances:
                                self.subassembly_instances[child_key] = []
                            if new_entry not in self.subassembly_instances[child_key]:
                                self.subassembly_instances[child_key].append(new_entry)
                                changed = True
    
    def _fetch_all_subassemblies_recursively(self):
        """
        Fetch full assembly data for each subassembly independently via API.
        This gives us each subassembly's LOCAL occurrence IDs that match feature references.
        Then build a mapping from local IDs to global paths in the main assembly.
        """
        # Storage for fetched subassembly data
        self.subassembly_full_data: dict = {}  # key: element_id → full assembly data
        self.subassembly_local_to_global: dict = {}  # key: element_id → {local_id: global_path}
        
        # Track which subassemblies we need to fetch (by element_id)
        element_ids_to_fetch = set()
        
        # Collect all unique subassembly element_ids from instances
        for instance in self.assembly_data["rootAssembly"]["instances"]:
            if instance.get("type") == "Assembly" and not instance.get("suppressed", False):
                element_ids_to_fetch.add((
                    instance.get("documentId"),
                    instance.get("elementId"),
                    instance.get("configuration")
                ))
        
        # Also collect nested subassembly element_ids from existing subAssemblies list
        for sub_assembly in self.assembly_data.get("subAssemblies", []):
            for instance in sub_assembly.get("instances", []):
                if instance.get("type") == "Assembly" and not instance.get("suppressed", False):
                    element_ids_to_fetch.add((
                        instance.get("documentId"),
                        instance.get("elementId"),
                        instance.get("configuration")
                    ))
        
        # Fetch each subassembly independently
        fetched_count = 0
        for doc_id, element_id, config in element_ids_to_fetch:
            if element_id in self.subassembly_full_data:
                continue  # Already fetched
            
            # Skip cross-document (linked) subassemblies - they have different doc_id
            # These require special handling with microversion access
            if doc_id != self.document_id:
                print(info(f"  Skipping external subassembly {element_id[:8]}... (different document)"))
                continue
            
            try:
                print(info(f"  Fetching subassembly {element_id[:8]}... via API"))
                
                sub_data = self.client.get_assembly(
                    doc_id,
                    self.version_id if self.version_id else self.workspace_id,
                    element_id,
                    wmv=("v" if self.version_id else "w"),
                    configuration=config or "default",
                )
                
                self.subassembly_full_data[element_id] = sub_data
                fetched_count += 1
                
                # Build local-to-global mapping for this subassembly
                # Local IDs are in sub_data["rootAssembly"]["occurrences"]
                # We need to map these to global paths in the main assembly
                self._build_local_to_global_mapping(element_id, sub_data)
                
            except Exception as e:
                print(warning(f"  WARNING: Failed to fetch subassembly {element_id[:8]}: {e}"))
        
        if fetched_count > 0:
            print(success(f"+ Fetched {fetched_count} subassembly(s) for local ID mapping"))
    
    def _build_local_to_global_mapping(self, element_id: str, sub_data: dict):
        """
        Build a mapping from subassembly's local occurrence IDs to global paths.
        For a part with local path [local_id] in the subassembly, find its global path in main assembly.
        The global path will be [parent_instance_id, ..., global_leaf_id].
        """
        # Get all locations where this subassembly is instantiated
        parent_locations = []
        for key, locations in self.subassembly_instances.items():
            if key[1] == element_id:  # key[1] is elementId
                parent_locations.extend(locations)
        
        if not parent_locations:
            return
        
        # Build mapping: local_occurrence_name → list of global_leaf_ids
        # Using PART NAME matching since IDs won't match
        self.subassembly_local_to_global[element_id] = {}
        
        # Get local occurrences from the fetched subassembly
        local_occurrences = sub_data.get("rootAssembly", {}).get("occurrences", [])
        local_instances = sub_data.get("rootAssembly", {}).get("instances", [])
        
        # Build local ID to name mapping
        local_id_to_name = {}
        for instance in local_instances:
            local_id_to_name[instance["id"]] = instance.get("name", "")
        
        for local_occ in local_occurrences:
            local_path = local_occ.get("path", [])
            if not local_path:
                continue
            
            local_leaf_id = local_path[-1]
            local_name = local_id_to_name.get(local_leaf_id, "")
            
            if not local_name:
                continue
            
            # Find matching global occurrences by NAME
            for prefix_path, parent_inst_id in parent_locations:
                full_prefix = prefix_path + [parent_inst_id]
                
                # Search for global occurrence with matching name AND correct prefix
                for global_occ in self.assembly_data["rootAssembly"]["occurrences"]:
                    global_path = global_occ["path"]
                    
                    # Check if global path starts with our prefix
                    if not self._path_starts_with(global_path, full_prefix):
                        continue
                    
                    # Get global leaf instance info
                    global_leaf_id = global_path[-1]
                    
                    # Find the instance name for this global ID
                    global_name = None
                    for sub_asm in self.assembly_data.get("subAssemblies", []):
                        for inst in sub_asm.get("instances", []):
                            if inst["id"] == global_leaf_id:
                                global_name = inst.get("name", "")
                                break
                        if global_name:
                            break
                    
                    # Match by name
                    if global_name == local_name:
                        # Found a match! Map local_leaf_id → global_leaf_id
                        if local_leaf_id not in self.subassembly_local_to_global[element_id]:
                            self.subassembly_local_to_global[element_id][local_leaf_id] = []
                        self.subassembly_local_to_global[element_id][local_leaf_id].append(global_leaf_id)

    def find_instances(self, prefix: list = [], instances=None):
        """
        Walking all the instances and associating them with their occurrences
        """
        if instances is None:
            instances = self.assembly_data["rootAssembly"]["instances"]

        for instance in instances:
            if "type" in instance:
                path = prefix + [instance["id"]]
                self.get_occurrence(path)["instance"] = instance

                if instance["type"] == "Assembly":
                    if not instance["suppressed"]:
                        d = instance["documentId"]
                        m = instance["documentMicroversion"]
                        e = instance["elementId"]
                        c = instance["configuration"]
                        for sub_assembly in self.assembly_data["subAssemblies"]:
                            if (
                                sub_assembly["documentId"] == d
                                and sub_assembly["documentMicroversion"] == m
                                and sub_assembly["elementId"] == e
                                and sub_assembly["configuration"] == c
                            ):
                                self.find_instances(
                                    prefix + [instance["id"]], sub_assembly["instances"]
                                )

    def load_features(self):
        """
        Load features
        """

        self.features = self.client.get_features(
            self.document_id,
            self.microversion_id,
            self.element_id,
            wmv="m",
            configuration=self.config.configuration,
        )

        self.matevalues = self.client.matevalues(
            self.document_id,
            self.version_id if self.version_id else self.workspace_id,
            self.element_id,
            wmv="v" if self.version_id else "w",
            configuration=self.config.configuration,
        )

    def load_configuration(self):
        """
        Load configuration parameters
        """

        self.variable_values = None

        # Extracting configuration variables
        parts = self.assembly_data["rootAssembly"]["fullConfiguration"].split(";")
        for part in parts:
            key_value = part.split("=")
            if len(key_value) == 2:
                key, value = key_value
                value = value.replace("+", " ")
                self.configuration_parameters[key] = value
                try:
                    param_value = self.expression_parser.eval_expr(value)
                    self.expression_parser.variables[key] = param_value
                except ValueError:
                    pass

    def load_variables(self):
        """
        Load variables values (only if needed) in the expression parser
        """
        variables = self.client.get_variables(
            self.document_id,
            self.version_id if self.version_id else self.workspace_id,
            self.element_id,
            wmv="v" if self.version_id else "w",
            configuration=self.config.configuration,
        )
        for entry in variables:
            for variable in entry["variables"]:
                if variable["value"] is not None:
                    self.expression_parser.variables[variable["name"]] = (
                        self.expression_parser.eval_expr(variable["value"])
                    )

    def get_occurrence(self, path: list):
        """
        Retrieve occurrence from its path
        """
        return self.occurrences[tuple(path)]

    def get_occurrence_transform(self, path: list) -> np.ndarray:
        """
        Retrieve occurrence transform from its path
        """
        T_world_part = np.array(self.get_occurrence(path)["transform"]).reshape(4, 4)

        return T_world_part

    def cs_to_transformation(self, cs: dict) -> np.ndarray:
        """
        Convert a coordinate system to a transformation matrix
        """
        T = np.eye(4)
        T[:3, :3] = np.stack(
            (
                np.array(cs["xAxis"]),
                np.array(cs["yAxis"]),
                np.array(cs["zAxis"]),
            )
        ).T
        T[:3, 3] = cs["origin"]

        return T

    def get_mate_transform(self, mated_entity: dict):
        return self.cs_to_transformation(mated_entity["matedCS"])

    def make_body(self, id: str):
        """
        Make the given instance id a body
        """
        self.instance_body[id] = self.current_body_id
        self.current_body_id += 1

    def merge_bodies(self, occurrence_A: str, occurrence_B: str):
        # Ensure occurrences are body
        if occurrence_A not in self.instance_body:
            self.make_body(occurrence_A)
        if occurrence_B not in self.instance_body:
            self.make_body(occurrence_B)

        # Merging bodies
        body1_id = self.instance_body[occurrence_A]
        body2_id = self.instance_body[occurrence_B]
        if body1_id > body2_id:
            body1_id, body2_id = body2_id, body1_id

        for occurrence in self.instance_body:
            if self.instance_body[occurrence] == body2_id:
                self.instance_body[occurrence] = body1_id

        for dof in self.dofs:
            if dof.body1_id == body2_id:
                dof.body1_id = body1_id
            if dof.body2_id == body2_id:
                dof.body2_id = body1_id

    def translation(self, x: float, y: float, z: float) -> np.ndarray:
        return np.array(
            [
                [1, 0, 0, x],
                [0, 1, 0, y],
                [0, 0, 1, z],
                [0, 0, 0, 1],
            ]
        )

    def process_mates(self):
        """
        Pre-assign all top-level instances to a separate body id
        """
        top_level_instances = self.assembly_data["rootAssembly"]["instances"]
        self.make_body(top_level_instances[0]["id"])
        
        # Debug: Print top-level instances for tracing
        print(bright(f"* Top-level instances:"))
        for inst in top_level_instances:
            inst_type = inst.get("type", "Unknown")
            inst_name = inst.get("name", "unnamed")
            inst_id = inst["id"][:8]
            print(bright(f"    - {inst_name} ({inst_type}) [{inst_id}...]"))

        # We first search for DOFs - using mate type detection instead of requiring "dof_" prefix
        # This allows using natural Onshape naming like "prismatic_*", "revolute_*", etc.
        for data, occurrence_A, occurrence_B in self.feature_mating_two_occurrences():
            # Determine if this is a DOF based on mate type and naming patterns
            mate_type = data["mateType"]
            mate_name = data["name"]
            
            # DOF detection logic:
            # 1. Explicit DOF naming: starts with "dof_"
            # 2. Joint type naming: starts with joint type (prismatic_, revolute_, slider_, cylindrical_)
            # 3. Exclude fixed mates unless explicitly marked as dof
            is_explicit_dof = mate_name.startswith("dof_")
            is_typed_joint = (
                mate_name.startswith("prismatic_") or 
                mate_name.startswith("revolute_") or 
                mate_name.startswith("slider_") or 
                mate_name.startswith("cylindrical_") or
                mate_name.startswith("ball_")
            )
            
            # Skip if this is a special-purpose mate (not a DOF)
            if (mate_name.startswith("fix_") or 
                mate_name.startswith("closing_") or 
                mate_name.startswith("frame_")):
                continue
            
            # Only process as DOF if explicitly marked or typed as a joint
            if not (is_explicit_dof or is_typed_joint):
                continue
            
            # Process the DOF name, removing dof prefix and inv suffix
            parts = mate_name.split("_")
            if parts[0] == "dof":
                del parts[0]
            
            data["inverted"] = False
            if len(parts) > 0 and (parts[-1] == "inv" or parts[-1] == "inverted"):
                data["inverted"] = True
                del parts[-1]
            
            name = "_".join(parts)
            
            if name == "":
                raise Exception(
                    f"ERROR: the following dof should have a name {mate_name}"
                )
            
            # Finding joint type and limits based on mate type
            limits = None
            if mate_type == "REVOLUTE" or mate_type == "CYLINDRICAL":
                if "wheel" in parts or "continuous" in parts:
                    joint_type = Joint.CONTINUOUS
                else:
                    joint_type = Joint.REVOLUTE
                
                if not self.config.ignore_limits:
                    limits = self.get_limits(joint_type, mate_name)
            elif mate_type == "SLIDER":
                joint_type = Joint.PRISMATIC
                if not self.config.ignore_limits:
                    limits = self.get_limits(joint_type, mate_name)
            elif mate_type == "FASTENED":
                joint_type = Joint.FIXED
            elif mate_type == "BALL":
                joint_type = Joint.BALL
                if not self.config.ignore_limits:
                    limits = self.get_limits(joint_type, mate_name)
            else:
                raise Exception(
                    f"ERROR: {name} is declared as a DOF but the mate type is {mate_type}\n"
                    + "       Only REVOLUTE, CYLINDRICAL, SLIDER and FASTENED are supported"
                )
            
            # We compute the axis in the world frame
            mated_entity = data["matedEntities"][0]
            T_world_part = self.get_occurrence_transform(
                mated_entity["matedOccurrence"]
            )

            # jointToPart is the (rotation only) matrix from joint to the part
            # it is attached to
            T_part_mate = self.get_mate_transform(mated_entity)

            T_world_mate = T_world_part @ T_part_mate

            limits_str = ""
            if limits is not None:
                limits_str = f"[{round(limits[0], 3)}: {round(limits[1], 3)}]"
            print(success(f"+ Found DOF: {name} ({joint_type}) {limits_str}"))

            # Ensure occurrences are body
            if occurrence_A not in self.instance_body:
                self.make_body(occurrence_A)
            if occurrence_B not in self.instance_body:
                self.make_body(occurrence_B)

            dof = DOF(
                self.instance_body[occurrence_A],
                self.instance_body[occurrence_B],
                name,
                joint_type,
                T_world_mate,
                limits,
            )

            if data["inverted"]:
                dof.flip()

            self.dofs.append(dof)

        # Merging fixed links
        for data, occurrence_A, occurrence_B in self.feature_mating_two_occurrences():
            if data["name"].startswith("fix_") or (
                data["mateType"] == "FASTENED"
                and not data["name"].startswith("dof_")
                and not data["name"].startswith("closing_")
                and not data["name"].startswith("frame_")
            ):
                self.merge_bodies(occurrence_A, occurrence_B)

        # Merging mate gorups
        for group in self.feature_mate_groups():
            for k in range(1, len(group)):
                occurrence_A = group[0]
                occurrence_B = group[k]

                self.merge_bodies(occurrence_A, occurrence_B)

        # Processing frame mates
        for data, occurrence_A, occurrence_B in self.feature_mating_two_occurrences():
            if data["name"].startswith("frame_"):
                name = "_".join(data["name"].split("_")[1:])
                if (
                    occurrence_A not in self.instance_body
                    and occurrence_B in self.instance_body
                ):
                    parent, child = occurrence_B, occurrence_A
                    mated_entity = data["matedEntities"][0]
                elif (
                    occurrence_B not in self.instance_body
                    and occurrence_A in self.instance_body
                ):
                    parent, child = occurrence_A, occurrence_B
                    mated_entity = data["matedEntities"][1]
                else:
                    raise Exception(
                        f"Frame {name} should mate an orphan body to a body in the kinematics tree"
                    )

                T_world_part = self.get_occurrence_transform(
                    mated_entity["matedOccurrence"]
                )

                self.frames.append(
                    Frame(self.instance_body[parent], name, T_world_part)
                )

                if self.config.draw_frames:
                    self.merge_bodies(parent, child)
                else:
                    self.instance_body[child] = INSTANCE_IGNORE

        # Checking that all instances are assigned to a body
        # SKIP subassembly instances if their internal parts already have bodies
        for instance in self.assembly_data["rootAssembly"]["instances"]:
            if instance["id"] not in self.instance_body and not instance["suppressed"]:
                # Check if this is a subassembly with internal DOFs
                is_subassembly_with_dofs = False
                if instance.get("type") == "Assembly":
                    # Check if any DOF uses a part inside this subassembly
                    for dof in self.dofs:
                        # Check if either body is inside this subassembly
                        for occ_id in self.instance_body:
                            if self.instance_body[occ_id] in [dof.body1_id, dof.body2_id]:
                                # Check if this occurrence is inside the subassembly
                                full_path = self.occurrence_id_to_path.get(occ_id, [occ_id])
                                if len(full_path) > 1 and full_path[0] == instance["id"]:
                                    is_subassembly_with_dofs = True
                                    break
                        if is_subassembly_with_dofs:
                            break
                
                # Only create body for instances that aren't subassemblies with internal DOFs
                if not is_subassembly_with_dofs:
                    self.make_body(instance["id"])
        
        # FALLBACK: Merge orphan nested parts into their top-level subassembly's body
        # This handles cases where nested group members couldn't be resolved by ID
        self._merge_orphan_nested_parts()

        # Processing loop closing frames
        for data, occurrence_A, occurrence_B in self.feature_mating_two_occurrences():
            is_hinge_closure = data["mateType"] == "REVOLUTE"

            if data["name"].startswith("closing_"):
                for k in 0, 1:
                    mated_entity = data["matedEntities"][k]
                    occurrence = mated_entity["matedOccurrence"][0]

                    T_world_part = self.get_occurrence_transform(
                        mated_entity["matedOccurrence"]
                    )
                    T_part_mate = self.get_mate_transform(mated_entity)
                    T_world_mate = T_world_part @ T_part_mate

                    self.frames.append(
                        Frame(
                            self.instance_body[occurrence],
                            f"{data['name']}_{k+1}",
                            T_world_mate,
                        )
                    )

                    if is_hinge_closure:
                        self.frames.append(
                            Frame(
                                self.instance_body[occurrence],
                                f"{data['name']}_{k+1}_z",
                                T_world_mate @ self.translation(0, 0, 0.1),
                            )
                        )

                closure_types = {
                    "FASTENED": "fixed",
                    "REVOLUTE": "revolute",
                    "BALL": "ball",
                    "SLIDER": "slider",
                }

                self.closures.append(
                    [
                        closure_types.get(data["mateType"], "unknown"),
                        f"{data['name']}_1",
                        f"{data['name']}_2",
                    ]
                )
                if is_hinge_closure:
                    self.closures.append(
                        [
                            closure_types.get(data["mateType"], "unknown"),
                            f"{data['name']}_1_z",
                            f"{data['name']}_2_z",
                        ]
                    )

        # Search for mate connector named "link_..." to override link names
        for feature in self.assembly_data["rootAssembly"]["features"]:
            if feature["featureType"] == "mateConnector" and feature["featureData"][
                "name"
            ].startswith("link_"):
                link_name = "_".join(feature["featureData"]["name"].split("_")[1:])
                body_id = self.instance_body[feature["featureData"]["occurrence"][0]]
                self.link_names[body_id] = link_name

            if feature["featureType"] == "mateConnector" and feature["featureData"][
                "name"
            ].startswith("frame_"):
                name = "_".join(feature["featureData"]["name"].split("_")[1:])
                occurrence = feature["featureData"]["occurrence"]
                T_world_occurrence = self.get_occurrence_transform(occurrence)
                body_id = self.instance_body[occurrence[0]]
                T_occurrence_mate = self.cs_to_transformation(
                    feature["featureData"]["mateConnectorCS"]
                )
                T_world_mate = T_world_occurrence @ T_occurrence_mate
                self.frames.append(Frame(body_id, name, T_world_mate))

        print(success(f"* Found total {len(self.dofs)} degrees of freedom"))

    def build_trees(self):
        """
        Perform checks on the produced tree
        """
        self.body_in_tree = []
        for body_id in self.instance_body.values():
            if body_id != INSTANCE_IGNORE and body_id not in self.body_in_tree:
                self.build_tree(body_id)

        print(success(f"* Found {len(self.root_nodes)} root nodes:"))
        for root_node in self.root_nodes:
            print(success(f"  - {self.body_instance(root_node)['name']}"))

    def build_tree(self, root_node: int):
        """
        Build kinematic tree starting from root_node.
        IMPROVED: Treats DOFs as UNDIRECTED graph edges.
        Automatically determines parent-child direction via traversal from base.
        This removes the requirement for specific mate entity ordering in Onshape.
        """
        # Append the root node
        self.root_nodes.append(root_node)

        # Treat DOF graph as UNDIRECTED - traverse from root and orient edges
        exploring = [root_node]
        dofs = self.dofs.copy()
        
        while len(exploring) > 0:
            current = exploring.pop()
            self.body_in_tree.append(current)

            children = []
            dofs_to_remove = []
            
            for dof in dofs:
                child_body = None
                needs_flip = False
                
                # Check if DOF connects to current body (treat as undirected edge)
                if dof.body1_id == current:
                    # Current is body1 → body2 is child
                    child_body = dof.body2_id
                    needs_flip = True  # Flip so body2 becomes body1 (parent → child convention)
                elif dof.body2_id == current:
                    # Current is body2 → body1 is child
                    child_body = dof.body1_id
                    needs_flip = False  # Already oriented correctly (body2=parent → body1=child)
                
                if child_body is not None:
                    # Found a connecting DOF - orient it from current (parent) to child
                    if needs_flip:
                        dof.flip(flip_limits=False)
                    
                    children.append(child_body)
                    dofs_to_remove.append(dof)
            
            # Remove processed DOFs
            for dof in dofs_to_remove:
                dofs.remove(dof)

            self.tree_children[current] = children
            
            # Add children to exploration queue
            for child in children:
                if child in self.body_in_tree:
                    raise Exception(
                        "The DOF graph is not a tree, check for loops in your DOFs"
                    )
                elif child not in exploring:
                    exploring.append(child)

    def collect_all_mates_once(self):
        """
        Collect ALL mates (root + subassembly) ONCE and cache them.
        This avoids issues with generator being consumed multiple times.
        """
        if hasattr(self, '_all_mates_cache'):
            return self._all_mates_cache
        
        self._all_mates_cache = []
        
        # Process root assembly features
        for feature in self.assembly_data["rootAssembly"]["features"]:
            if feature["featureType"] == "mate" and not feature["suppressed"]:
                import copy
                data = copy.deepcopy(feature["featureData"])

                if (
                    "matedEntities" not in data
                    or len(data["matedEntities"]) != 2
                    or len(data["matedEntities"][0]["matedOccurrence"]) == 0
                    or len(data["matedEntities"][1]["matedOccurrence"]) == 0
                ):
                    continue

                occ_path_A = data["matedEntities"][0]["matedOccurrence"]
                occ_path_B = data["matedEntities"][1]["matedOccurrence"]
                
                occurrence_A = occ_path_A[-1] if occ_path_A else None
                occurrence_B = occ_path_B[-1] if occ_path_B else None
                
                if occurrence_A is None or occurrence_B is None:
                    continue

                self._all_mates_cache.append((data, occurrence_A, occurrence_B))
        
        # Process subassembly features using RECURSIVE mapping
        subassembly_count = len(self.assembly_data.get("subAssemblies", []))
        if subassembly_count > 0:
            print(bright(f"* Scanning {subassembly_count} subassemblies for additional mates (recursive)..."))
        
        processed_subassemblies = set()
        
        for sub_assembly in self.assembly_data.get("subAssemblies", []):
            sub_doc_id = sub_assembly.get("documentId")
            sub_element_id = sub_assembly.get("elementId")
            sub_config = sub_assembly.get("configuration")
            sub_microversion = sub_assembly.get("documentMicroversion")
            
            sub_key = (sub_doc_id, sub_element_id, sub_config, sub_microversion)
            if sub_key in processed_subassemblies:
                continue
            processed_subassemblies.add(sub_key)
            
            # Use RECURSIVE subassembly_instances mapping to find ALL locations
            # where this subassembly is instantiated (at any nesting level)
            parent_locations = self.subassembly_instances.get(sub_key, [])
            
            if not parent_locations:
                # Fallback: Try old method for backward compatibility
                for instance in self.assembly_data["rootAssembly"]["instances"]:
                    if (instance.get("type") == "Assembly" and
                        instance.get("documentId") == sub_doc_id and
                        instance.get("elementId") == sub_element_id and
                        instance.get("configuration") == sub_config and
                        instance.get("documentMicroversion") == sub_microversion):
                        parent_locations.append(([], instance["id"]))
            
            if not parent_locations:
                print(warning(f"  WARNING: Could not find parent instance for subassembly (nested too deep?)"))
                continue
            
            features_in_subassembly = sub_assembly.get("features", [])
            mate_count = sum(1 for f in features_in_subassembly if f.get("featureType") == "mate")
            
            # Process for EACH location where this subassembly is instantiated
            for prefix_path, parent_inst_id in parent_locations:
                full_prefix = prefix_path + [parent_inst_id]
                depth_indicator = "  " * len(prefix_path)
                print(bright(f"  {depth_indicator}- Subassembly '{parent_inst_id[:8]}...' (depth {len(full_prefix)}) with {mate_count} mate features"))
                
                for feature in features_in_subassembly:
                    if feature["featureType"] == "mate" and not feature.get("suppressed", False):
                        import copy
                        data = copy.deepcopy(feature["featureData"])

                        if (
                            "matedEntities" not in data
                            or len(data["matedEntities"]) != 2
                            or len(data["matedEntities"][0]["matedOccurrence"]) == 0
                            or len(data["matedEntities"][1]["matedOccurrence"]) == 0
                        ):
                            continue

                        # Get local occurrence IDs from subassembly mate
                        relative_occ_A = data["matedEntities"][0]["matedOccurrence"][0]
                        relative_occ_B = data["matedEntities"][1]["matedOccurrence"][0]
                        
                        # Look up full paths using leaf ID mapping
                        full_path_A = self.occurrence_id_to_path.get(relative_occ_A)
                        full_path_B = self.occurrence_id_to_path.get(relative_occ_B)
                        
                        if full_path_A is None or full_path_B is None:
                            # For deeply nested parts, construct path from prefix + local ID
                            # Search occurrences that start with our prefix and end with the local ID
                            full_path_A = self._find_occurrence_path_with_prefix(full_prefix, relative_occ_A)
                            full_path_B = self._find_occurrence_path_with_prefix(full_prefix, relative_occ_B)
                        
                        if full_path_A is None or full_path_B is None:
                            print(warning(f"    {depth_indicator}Skipping {data.get('name')}: occurrence paths not found"))
                            continue
                        
                        # Validate paths start with expected prefix
                        if not self._path_starts_with(full_path_A, full_prefix):
                            print(warning(f"    {depth_indicator}Skipping {data.get('name')}: path A not in subassembly"))
                            continue
                        if not self._path_starts_with(full_path_B, full_prefix):
                            print(warning(f"    {depth_indicator}Skipping {data.get('name')}: path B not in subassembly"))
                            continue
                        
                        data["matedEntities"][0]["matedOccurrence"] = list(full_path_A)
                        data["matedEntities"][1]["matedOccurrence"] = list(full_path_B)
                        
                        occurrence_A = relative_occ_A
                        occurrence_B = relative_occ_B
                        
                        print(success(f"    {depth_indicator}+ Found subassembly mate: {data.get('name', 'unnamed')} ({occurrence_A[:8]}... <-> {occurrence_B[:8]}...)"))

                        self._all_mates_cache.append((data, occurrence_A, occurrence_B))
        
        return self._all_mates_cache
    
    def _find_occurrence_path_with_prefix(self, prefix, leaf_id):
        """
        Find an occurrence path that starts with the given prefix and ends with leaf_id.
        Used for deeply nested parts where direct leaf ID mapping may not work.
        """
        for occurrence in self.assembly_data["rootAssembly"]["occurrences"]:
            path = occurrence["path"]
            if len(path) > len(prefix) and path[-1] == leaf_id:
                if self._path_starts_with(path, prefix):
                    return path
        return None
    
    def _path_starts_with(self, path, prefix):
        """Check if path starts with the given prefix."""
        if len(path) < len(prefix):
            return False
        return path[:len(prefix)] == prefix
    
    def _build_name_to_global_id_mapping(self):
        """
        Build a mapping from part NAME to list of global occurrence IDs.
        Used for external subassemblies where we can't fetch local IDs but can match by name.
        """
        name_to_ids = {}
        
        # Build instance ID to name mapping from all subassemblies
        id_to_name = {}
        for sub_assembly in self.assembly_data.get("subAssemblies", []):
            for instance in sub_assembly.get("instances", []):
                instance_name = instance.get("name", "")
                instance_id = instance.get("id", "")
                if instance_name and instance_id:
                    id_to_name[instance_id] = instance_name
        
        # Also add top-level instances
        for instance in self.assembly_data["rootAssembly"]["instances"]:
            instance_name = instance.get("name", "")
            instance_id = instance.get("id", "")
            if instance_name and instance_id:
                id_to_name[instance_id] = instance_name
        
        # Map occurrence paths to names
        for occurrence in self.assembly_data["rootAssembly"]["occurrences"]:
            path = occurrence["path"]
            if not path:
                continue
            
            leaf_id = path[-1]
            part_name = id_to_name.get(leaf_id, "")
            
            if part_name:
                if part_name not in name_to_ids:
                    name_to_ids[part_name] = []
                name_to_ids[part_name].append(leaf_id)
        
        return name_to_ids
    
    def _merge_orphan_nested_parts(self):
        """
        Fallback: Merge nested parts that don't have a body into their parent subassembly's body.
        This handles cases where group members in depth-2+ subassemblies couldn't be resolved by ID.
        Parts inside a subassembly without explicit mates get merged into the first part's body.
        """
        merged_count = 0
        
        # Build mapping: top-level subassembly ID → body ID (of first assigned part in that subassembly)
        subassembly_body_mapping = {}
        for occ_id, body_id in self.instance_body.items():
            full_path = self.occurrence_id_to_path.get(occ_id)
            if full_path and len(full_path) > 1:
                top_level_id = full_path[0]
                if top_level_id not in subassembly_body_mapping:
                    subassembly_body_mapping[top_level_id] = body_id
        
        # Find orphan parts (occurrences without body assignment)
        for occurrence in self.assembly_data["rootAssembly"]["occurrences"]:
            path = occurrence["path"]
            if len(path) < 2:
                continue  # Skip top-level parts
            
            leaf_id = path[-1]
            top_level_id = path[0]
            
            # Check if this part has a body assignment
            if leaf_id in self.instance_body:
                continue  # Already assigned
            
            # Try to merge into parent subassembly's body
            if top_level_id in subassembly_body_mapping:
                parent_body_id = subassembly_body_mapping[top_level_id]
                self.instance_body[leaf_id] = parent_body_id
                merged_count += 1
            elif top_level_id in self.instance_body:
                # Use the top-level instance's body
                self.instance_body[leaf_id] = self.instance_body[top_level_id]
                merged_count += 1
        
        if merged_count > 0:
            print(success(f"+ Merged {merged_count} orphan nested parts into parent subassembly bodies"))

    def feature_mating_two_occurrences(self):
        """
        Iterate over all valid mating features with two occurrences.
        Returns cached list of mates (root + subassembly) that can be iterated multiple times.
        """
        for data, occurrence_A, occurrence_B in self.collect_all_mates_once():
            yield data, occurrence_A, occurrence_B

    def feature_mate_groups(self):
        """
        Find mate groups in the assembly (root + subassemblies with recursive path resolution).
        Returns list of groups, where each group is a list of occurrence IDs to be merged.
        """
        groups = []

        # Process root assembly mate groups
        for feature in self.assembly_data["rootAssembly"]["features"]:
            group = []
            if feature["featureType"] == "mateGroup" and not feature["suppressed"]:
                data = feature["featureData"]

                for occurrence in data.get("occurrences", []):
                    occ_path = occurrence.get("occurrence", [])
                    if occ_path:
                        # Use leaf ID for top-level groups
                        group.append(occ_path[-1])
            if group:
                groups.append(group)
        
        # Build name-to-global-id mapping for ALL parts in main assembly (for external subassemblies)
        name_to_global_ids = self._build_name_to_global_id_mapping()
        
        # Process subassembly mate groups with recursive path resolution
        processed_subassemblies = set()
        
        for sub_assembly in self.assembly_data.get("subAssemblies", []):
            sub_doc_id = sub_assembly.get("documentId")
            sub_element_id = sub_assembly.get("elementId")
            sub_config = sub_assembly.get("configuration")
            sub_microversion = sub_assembly.get("documentMicroversion")
            
            sub_key = (sub_doc_id, sub_element_id, sub_config, sub_microversion)
            if sub_key in processed_subassemblies:
                continue
            processed_subassemblies.add(sub_key)
            
            # Get all locations where this subassembly is instantiated
            parent_locations = self.subassembly_instances.get(sub_key, [])
            
            if not parent_locations:
                # Fallback for direct children of root
                for instance in self.assembly_data["rootAssembly"]["instances"]:
                    if (instance.get("type") == "Assembly" and
                        instance.get("documentId") == sub_doc_id and
                        instance.get("elementId") == sub_element_id and
                        instance.get("configuration") == sub_config and
                        instance.get("documentMicroversion") == sub_microversion):
                        parent_locations.append(([], instance["id"]))
            
            # Find mate groups in this subassembly
            features_in_subassembly = sub_assembly.get("features", [])
            
            for feature in features_in_subassembly:
                if feature.get("featureType") == "mateGroup" and not feature.get("suppressed", False):
                    data = feature.get("featureData", {})
                    
                    # Process for EACH location where this subassembly is instantiated
                    for prefix_path, parent_inst_id in parent_locations:
                        full_prefix = prefix_path + [parent_inst_id]
                        group = []
                        
                        for occurrence in data.get("occurrences", []):
                            relative_occ_path = occurrence.get("occurrence", [])
                            if not relative_occ_path:
                                continue
                            
                            relative_occ_id = relative_occ_path[0]
                            resolved_occ_id = relative_occ_id  # May get replaced with global ID
                            
                            # Try to find full path using leaf ID mapping
                            full_path = self.occurrence_id_to_path.get(relative_occ_id)
                            
                            if full_path is None:
                                # Search for path with correct prefix
                                full_path = self._find_occurrence_path_with_prefix(full_prefix, relative_occ_id)
                            
                            # Try using local-to-global mapping from fetched subassembly data
                            if full_path is None and sub_element_id in self.subassembly_local_to_global:
                                local_mapping = self.subassembly_local_to_global[sub_element_id]
                                if relative_occ_id in local_mapping:
                                    global_ids = local_mapping[relative_occ_id]
                                    if global_ids:
                                        global_leaf_id = global_ids[0]
                                        if global_leaf_id in self.occurrence_id_to_path:
                                            full_path = self.occurrence_id_to_path[global_leaf_id]
                                            resolved_occ_id = global_leaf_id
                            
                            # FALLBACK for external subassemblies: Match by NAME using main assembly data
                            if full_path is None:
                                # Get the instance info for this local ID from subassembly
                                local_name = None
                                for inst in sub_assembly.get("instances", []):
                                    if inst.get("id") == relative_occ_id:
                                        local_name = inst.get("name", "")
                                        break
                                
                                if local_name and local_name in name_to_global_ids:
                                    # Find matching global ID that's in our prefix
                                    for candidate_id in name_to_global_ids[local_name]:
                                        candidate_path = self.occurrence_id_to_path.get(candidate_id)
                                        if candidate_path and self._path_starts_with(candidate_path, full_prefix):
                                            full_path = candidate_path
                                            resolved_occ_id = candidate_id
                                            break
                            
                            if full_path is not None and self._path_starts_with(full_path, full_prefix):
                                group.append(resolved_occ_id)
                            else:
                                print(warning(f"    Group member {relative_occ_id[:8]}... not found in subassembly"))
                        
                        if len(group) >= 2:
                            depth_indicator = "  " * len(prefix_path)
                            print(success(f"  {depth_indicator}+ Found subassembly mateGroup with {len(group)} parts"))
                            groups.append(group)

        return groups

    def get_feature_by_id(self, feature_id: str):
        """
        Find a specific feature by its ID
        """
        for feature in self.features["features"]:
            if feature["message"]["featureId"] == feature_id:
                return feature

        return None

    def find_relations(self):
        """
        Finding relations features in the assembly
        """
        for feature in self.features["features"]:
            if feature["typeName"] == "BTMMateRelation":
                relation_name = feature["message"]["name"]

                mated_dofs = None
                ratio = None
                reverse = None
                for parameter in feature["message"]["parameters"]:
                    if parameter["message"]["parameterId"] == "matesQuery":
                        queries = parameter["message"]["queries"]
                        if len(queries) == 2:
                            dof1 = self.get_feature_by_id(
                                queries[0]["message"]["featureId"]
                            )["message"]["name"]
                            dof2 = self.get_feature_by_id(
                                queries[1]["message"]["featureId"]
                            )["message"]["name"]
                            if dof1.startswith("dof_") and dof2.startswith("dof_"):
                                mated_dofs = [dof1[4:], dof2[4:]]
                    elif parameter["message"]["parameterId"] == "relationRatio":
                        ratio = self.read_expression(parameter["message"]["expression"])
                    elif parameter["message"]["parameterId"] == "reverseDirection":
                        reverse = parameter["message"]["value"]

                if mated_dofs is not None and ratio is not None and reverse is not None:
                    if not reverse:
                        ratio = -ratio

                    print(
                        success(
                            f"+ Found relation {relation_name} mating {mated_dofs} with ratio {ratio}"
                        )
                    )
                    if mated_dofs[1] in self.relations:
                        print(
                            warning(
                                f"Multiple relations found with {mated_dofs[1]} as target"
                            )
                        )

                    self.relations[mated_dofs[1]] = [mated_dofs[0], ratio]

    def read_parameter_value(self, parameter: str, name: str):
        """
        Try to read a parameter value from Onshape
        """

        # This is an expression
        if parameter["typeName"] == "BTMParameterNullableQuantity":
            return self.read_expression(parameter["message"]["expression"])
        if parameter["typeName"] == "BTMParameterConfigured":
            message = parameter["message"]
            parameterValue = self.configuration_parameters[
                message["configurationParameterId"]
            ]

            for value in message["values"]:
                if value["typeName"] == "BTMConfiguredValueByBoolean":
                    booleanValue = parameterValue == "true"
                    if value["message"]["booleanValue"] == booleanValue:
                        return self.read_expression(
                            value["message"]["value"]["message"]["expression"]
                        )
                elif value["typeName"] == "BTMConfiguredValueByEnum":
                    if value["message"]["enumValue"] == parameterValue:
                        return self.read_expression(
                            value["message"]["value"]["message"]["expression"]
                        )
                else:
                    raise Exception(
                        "Can't read value of parameter {name} configured with {value['typeName']}"
                    )

            print(error(f"Coud not find the value for {name}"))
        else:
            raise Exception(f"Unknown feature type for {name}: {parameter['typeName']}")

    def read_expression(self, expression: str):
        """
        Reading an expression from Onshape
        """
        return self.expression_parser.eval_expr(expression)

    def get_offset(self, name: str):
        """
        Retrieve the offset from current joint position in the assembly
        Currently, this only works with workspace in the API
        """
        if self.matevalues is None:
            return None

        for entry in self.matevalues["mateValues"]:
            if entry["mateName"] == name:
                if "rotationZ" in entry:
                    return entry["rotationZ"]
                elif "translationZ" in entry:
                    return entry["translationZ"]
                else:
                    print(warning(f"Unknown offset type for {name}"))
        return None

    def get_limits(self, joint_type: str, name: str):
        """
        Retrieve (low, high) limits for a given joint.
        If no limits are specified in Onshape, returns large default limits.
        URDF requires limits for revolute/prismatic joints - unlimited motion uses large values.
        """
        # Default limits for unlimited joints (2 orders of magnitude less than URDF max)
        DEFAULT_REVOLUTE_LIMIT = 628.318  # ~100 * 2π radians (~100 full rotations)
        DEFAULT_PRISMATIC_LIMIT = 100.0   # ±100 meters
        
        enabled = False
        minimum, maximum = 0, 0
        for feature in self.features["features"]:
            # Find coresponding joint
            if name == feature["message"]["name"]:
                # Find min and max values
                for parameter in feature["message"]["parameters"]:
                    if parameter["message"]["parameterId"] == "limitsEnabled":
                        enabled = parameter["message"]["value"]

                if enabled:
                    for parameter in feature["message"]["parameters"]:
                        if joint_type == Joint.REVOLUTE:
                            if parameter["message"]["parameterId"] == "limitAxialZMin":
                                minimum = self.read_parameter_value(parameter, name)
                            if parameter["message"]["parameterId"] == "limitAxialZMax":
                                maximum = self.read_parameter_value(parameter, name)
                        elif joint_type == Joint.PRISMATIC:
                            if parameter["message"]["parameterId"] == "limitZMin":
                                minimum = self.read_parameter_value(parameter, name)
                            if parameter["message"]["parameterId"] == "limitZMax":
                                maximum = self.read_parameter_value(parameter, name)
                        elif joint_type == Joint.BALL:
                            if (
                                parameter["message"]["parameterId"]
                                == "limitEulerConeAngleMax"
                            ):
                                minimum = 0
                                maximum = self.read_parameter_value(parameter, name)
                        else:
                            print(
                                warning(
                                    f"WARNING: Can't read limits for a joint of type {joint_type}"
                                )
                            )
                            print(parameter)
        if enabled:
            if joint_type != Joint.BALL:
                offset = self.get_offset(name)
                if offset is not None:
                    minimum -= offset
                    maximum -= offset
            return (minimum, maximum)
        else:
            # No limits enabled - provide defaults for non-continuous joints
            if joint_type == Joint.CONTINUOUS:
                return None  # Continuous joints don't need limits
            elif joint_type == Joint.REVOLUTE:
                print(info(f"  Using default limits for {name}: ±{DEFAULT_REVOLUTE_LIMIT:.1f} rad (~100 rotations)"))
                return (-DEFAULT_REVOLUTE_LIMIT, DEFAULT_REVOLUTE_LIMIT)
            elif joint_type == Joint.PRISMATIC:
                print(info(f"  Using default limits for {name}: ±{DEFAULT_PRISMATIC_LIMIT} m"))
                return (-DEFAULT_PRISMATIC_LIMIT, DEFAULT_PRISMATIC_LIMIT)
            else:
                # For other joint types, no limits
                return None

    def body_instance(self, body_id: int):
        """
        Get the instance associated with a given body.
        IMPROVED: For subassembly parts, return the SUBASSEMBLY instance (with nice name)
        instead of the first part instance inside it.
        """
        # First, collect all occurrence IDs that belong to this body
        occ_ids_for_body = [occ_id for occ_id, bid in self.instance_body.items() if bid == body_id]
        
        # Check if all parts share a common top-level subassembly parent
        # If so, return the subassembly instance for better naming
        common_subassembly_id = None
        for occ_id in occ_ids_for_body:
            full_path = self.occurrence_id_to_path.get(occ_id)
            if full_path and len(full_path) > 1:
                # This is a nested part - get its top-level parent (subassembly ID)
                top_level_id = full_path[0]
                if common_subassembly_id is None:
                    common_subassembly_id = top_level_id
                elif common_subassembly_id != top_level_id:
                    # Parts come from different subassemblies - no common parent
                    common_subassembly_id = None
                    break
            else:
                # This is a top-level part, not from a subassembly
                common_subassembly_id = None
                break
        
        # If all parts share a common subassembly parent, return that subassembly instance
        if common_subassembly_id:
            for instance in self.assembly_data["rootAssembly"]["instances"]:
                if instance["id"] == common_subassembly_id:
                    return instance
        
        # Otherwise, check top-level instances directly
        for instance in self.assembly_data["rootAssembly"]["instances"]:
            if (
                instance["id"] in self.instance_body
                and self.instance_body[instance["id"]] == body_id
            ):
                return instance
        
        # Fallback: Check in subassembly instances using leaf IDs
        for sub_assembly in self.assembly_data.get("subAssemblies", []):
            for instance in sub_assembly.get("instances", []):
                if (
                    instance["id"] in self.instance_body
                    and self.instance_body[instance["id"]] == body_id
                ):
                    return instance

        return None

    def body_occurrences(self, body_id: int):
        """
        Retrieve all occurrences associated to a given body id.
        FIXED: Checks ALL path elements (top-level, intermediate, AND leaf IDs) for subassembly parts.
        This ensures deeply nested parts from external subassemblies are correctly associated.
        """
        for occurrence in self.assembly_data["rootAssembly"]["occurrences"]:
            path = occurrence["path"]
            if not path:
                continue
            
            # Check ALL elements in the path - any match means this occurrence belongs to the body
            # This handles: top-level, intermediate subassemblies, AND leaf parts
            matches_body = False
            for path_element in path:
                if path_element in self.instance_body and self.instance_body[path_element] == body_id:
                    matches_body = True
                    break
            
            if matches_body:
                yield occurrence

    def get_dof(self, body1_id: int, body2_id: int):
        """
        Get a DOF for given bodies
        """
        for dof in self.dofs:
            if (dof.body1_id == body1_id and dof.body2_id == body2_id) or (
                dof.body1_id == body2_id and dof.body2_id == body1_id
            ):
                return dof

        raise Exception(f"ERROR: no DOF found between {body1_id} and {body2_id}")
