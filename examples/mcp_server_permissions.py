"""
Example of how to integrate permissions into your MCP server.

This shows practical patterns for permission management in a real application.
"""

from typing import Dict, List, Optional
from src.core import ExecutionContext, Permission, Duration, MacroEngine, MacroDefinition


class UserPermissionManager:
    """Manages permissions for different users/clients."""
    
    def __init__(self):
        # In a real application, this would come from a database or config
        self.user_permissions: Dict[str, frozenset[Permission]] = {
            'guest': frozenset([Permission.TEXT_INPUT]),
            'user': frozenset([
                Permission.TEXT_INPUT,
                Permission.SYSTEM_SOUND,
                Permission.FLOW_CONTROL
            ]),
            'power_user': frozenset([
                Permission.TEXT_INPUT,
                Permission.SYSTEM_SOUND,
                Permission.APPLICATION_CONTROL,
                Permission.SYSTEM_CONTROL,
                Permission.FLOW_CONTROL,
                Permission.CLIPBOARD_ACCESS
            ]),
            'admin': frozenset(Permission)  # All permissions
        }
    
    def get_permissions_for_user(self, user_type: str) -> frozenset[Permission]:
        """Get permissions for a user type."""
        return self.user_permissions.get(user_type, frozenset([Permission.TEXT_INPUT]))
    
    def create_context_for_user(self, user_type: str, timeout_seconds: int = 30) -> ExecutionContext:
        """Create an execution context for a specific user type."""
        permissions = self.get_permissions_for_user(user_type)
        return ExecutionContext.create_test_context(
            permissions=permissions,
            timeout=Duration.from_seconds(timeout_seconds)
        )
    
    def can_user_run_macro(self, user_type: str, macro: MacroDefinition) -> tuple[bool, List[Permission]]:
        """
        Check if a user can run a specific macro.
        
        Returns:
            (can_run, missing_permissions)
        """
        user_permissions = self.get_permissions_for_user(user_type)
        
        # Collect all required permissions from the macro's commands
        required_permissions = set()
        for command in macro.commands:
            required_permissions.update(command.get_required_permissions())
        
        missing_permissions = list(required_permissions - user_permissions)
        can_run = len(missing_permissions) == 0
        
        return can_run, missing_permissions


class MacroExecutionService:
    """Service that handles macro execution with permission checking."""
    
    def __init__(self):
        self.engine = MacroEngine()
        self.permission_manager = UserPermissionManager()
    
    def execute_macro_for_user(
        self, 
        macro: MacroDefinition, 
        user_type: str,
        timeout_seconds: Optional[int] = None
    ) -> Dict:
        """
        Execute a macro for a specific user with appropriate permissions.
        
        Returns a result dictionary with execution details.
        """
        # Check if user can run this macro
        can_run, missing_permissions = self.permission_manager.can_user_run_macro(user_type, macro)
        
        if not can_run:
            return {
                'success': False,
                'error': 'Insufficient permissions',
                'missing_permissions': [p.value for p in missing_permissions],
                'user_type': user_type
            }
        
        # Create context with user's permissions
        timeout = timeout_seconds or 30
        context = self.permission_manager.create_context_for_user(user_type, timeout)
        
        # Execute the macro
        result = self.engine.execute_macro(macro, context)
        
        return {
            'success': result.status.value == 'completed',
            'status': result.status.value,
            'execution_token': str(result.execution_token),
            'total_duration': result.total_duration.total_seconds() if result.total_duration else None,
            'error_details': result.error_details,
            'user_type': user_type,
            'permissions_used': [p.value for p in context.permissions]
        }
    
    def get_user_capabilities(self, user_type: str) -> Dict:
        """Get information about what a user can do."""
        permissions = self.permission_manager.get_permissions_for_user(user_type)
        
        return {
            'user_type': user_type,
            'permissions': [p.value for p in permissions],
            'can_run': {
                'text_macros': Permission.TEXT_INPUT in permissions,
                'sound_macros': Permission.SYSTEM_SOUND in permissions,
                'app_control': Permission.APPLICATION_CONTROL in permissions,
                'system_control': Permission.SYSTEM_CONTROL in permissions,
                'file_operations': Permission.FILE_ACCESS in permissions,
                'network_operations': Permission.NETWORK_ACCESS in permissions,
            }
        }


# Example usage in MCP server
def mcp_server_example():
    """Example of how you'd use this in your MCP server."""
    service = MacroExecutionService()
    
    # Create a sample macro that needs application control
    from src.core import create_test_macro, CommandType
    app_macro = create_test_macro("Launch Calculator", [CommandType.APPLICATION_CONTROL])
    sound_macro = create_test_macro("Play Alert", [CommandType.PLAY_SOUND])
    
    print("=== MCP Server Permission Example ===")
    
    # Test different user types
    for user_type in ['guest', 'user', 'power_user', 'admin']:
        print(f"\n--- {user_type.upper()} USER ---")
        
        # Show capabilities
        capabilities = service.get_user_capabilities(user_type)
        print(f"Permissions: {capabilities['permissions']}")
        
        # Try to run app control macro
        result = service.execute_macro_for_user(app_macro, user_type)
        print(f"App macro result: {'✅ Success' if result['success'] else '❌ Failed'}")
        if not result['success'] and 'missing_permissions' in result:
            print(f"  Missing: {result['missing_permissions']}")
        
        # Try to run sound macro
        result = service.execute_macro_for_user(sound_macro, user_type)
        print(f"Sound macro result: {'✅ Success' if result['success'] else '❌ Failed'}")


# Configuration-based permissions
def load_permissions_from_config():
    """Example of loading permissions from configuration."""
    # This could be a JSON file, environment variables, etc.
    config = {
        "default_timeout": 30,
        "user_roles": {
            "readonly": ["text_input"],
            "basic": ["text_input", "system_sound"],
            "advanced": ["text_input", "system_sound", "application_control"],
            "admin": "all"
        }
    }
    
    permission_map = {}
    for role, perms in config["user_roles"].items():
        if perms == "all":
            permission_map[role] = frozenset(Permission)
        else:
            permission_map[role] = frozenset([
                Permission[perm.upper()] for perm in perms
            ])
    
    return permission_map


if __name__ == "__main__":
    mcp_server_example()
    
    print("\n" + "="*50)
    print("CONFIG-BASED PERMISSIONS")
    config_perms = load_permissions_from_config()
    for role, perms in config_perms.items():
        print(f"{role}: {[p.value for p in perms]}")