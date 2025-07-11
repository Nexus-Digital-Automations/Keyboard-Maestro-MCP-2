from ...core.errors import MissingParameterError, ToolExecutionError
from ...core.types import Tool
from ..initialization import get_km_client


class GroupTools:
    """
    Macro Group Management Tools.
    Provides functionalities to list, create, delete, and manage Keyboard Maestro Macro Groups.
    """

    def __init__(self):
        self.km_client = get_km_client()

    def list_macro_groups(self) -> list:
        """
        Lists all available Keyboard Maestro Macro Groups.

        Returns:
            list: A list of dictionaries, where each dictionary represents a macro group
                  with keys like 'name', 'uuid', 'macros', etc.
        """
        try:
            return self.km_client.get_macro_groups()
        except Exception as e:
            raise ToolExecutionError(f"Failed to list macro groups: {e}") from e

    def create_macro_group(self, name: str, enabled: bool = True) -> dict:
        """
        Creates a new Keyboard Maestro Macro Group.

        Args:
            name (str): The name of the new macro group.
            enabled (bool): Whether the group should be enabled initially. Defaults to True.

        Returns:
            dict: The details of the newly created macro group.
        """
        if not name:
            raise MissingParameterError(
                "Group name is required to create a macro group."
            )
        try:
            # This is a placeholder for KM client's create group method.
            # Actual implementation might involve more parameters or a different call.
            group_data = self.km_client.create_macro_group(name=name, enabled=enabled)
            return group_data
        except Exception as e:
            raise ToolExecutionError(
                f"Failed to create macro group '{name}': {e}"
            ) from e

    def delete_macro_group(self, group_uuid: str) -> bool:
        """
        Deletes a Keyboard Maestro Macro Group by its UUID.

        Args:
            group_uuid (str): The UUID of the macro group to delete.

        Returns:
            bool: True if deletion was successful, False otherwise.
        """
        if not group_uuid:
            raise MissingParameterError(
                "Group UUID is required to delete a macro group."
            )
        try:
            return self.km_client.delete_macro_group(group_uuid)
        except Exception as e:
            raise ToolExecutionError(
                f"Failed to delete macro group '{group_uuid}': {e}"
            ) from e

    def get_tool_definitions(self) -> list[Tool]:
        return [
            Tool(
                name="list_macro_groups",
                description="Lists all Keyboard Maestro Macro Groups.",
                parameters=[],
            ),
            Tool(
                name="create_macro_group",
                description="Creates a new Keyboard Maestro Macro Group.",
                parameters=[
                    {
                        "name": "name",
                        "type": "str",
                        "description": "The name of the new macro group.",
                    },
                    {
                        "name": "enabled",
                        "type": "bool",
                        "description": "Whether the group should be enabled initially. Defaults to True.",
                        "optional": True,
                    },
                ],
            ),
            Tool(
                name="delete_macro_group",
                description="Deletes a Keyboard Maestro Macro Group by its UUID.",
                parameters=[
                    {
                        "name": "group_uuid",
                        "type": "str",
                        "description": "The UUID of the macro group to delete.",
                    }
                ],
            ),
        ]
