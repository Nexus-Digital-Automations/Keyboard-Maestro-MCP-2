"""
Examples of how to set up execution contexts with different permission levels.

This file demonstrates practical permission configurations for various use cases.
"""

from src.core import ExecutionContext, Permission, Duration, MacroEngine, create_test_macro, CommandType


class PermissionProfiles:
    """Predefined permission profiles for common use cases."""
    
    @staticmethod
    def minimal() -> frozenset[Permission]:
        """Minimal permissions - text input only."""
        return frozenset([Permission.TEXT_INPUT])
    
    @staticmethod
    def basic() -> frozenset[Permission]:
        """Basic permissions - text and sound."""
        return frozenset([
            Permission.TEXT_INPUT,
            Permission.SYSTEM_SOUND
        ])
    
    @staticmethod
    def automation() -> frozenset[Permission]:
        """Automation permissions - apps and system control."""
        return frozenset([
            Permission.TEXT_INPUT,
            Permission.SYSTEM_SOUND,
            Permission.APPLICATION_CONTROL,
            Permission.SYSTEM_CONTROL,
            Permission.FLOW_CONTROL
        ])
    
    @staticmethod
    def media() -> frozenset[Permission]:
        """Media permissions - audio, video, screen capture."""
        return frozenset([
            Permission.TEXT_INPUT,
            Permission.SYSTEM_SOUND,
            Permission.AUDIO_OUTPUT,
            Permission.SCREEN_CAPTURE,
            Permission.FILE_ACCESS
        ])
    
    @staticmethod
    def admin() -> frozenset[Permission]:
        """Admin permissions - everything enabled."""
        return frozenset(Permission)  # All permissions
    
    @staticmethod
    def custom(*permissions: Permission) -> frozenset[Permission]:
        """Custom permissions - specify exactly what you need."""
        return frozenset(permissions)


def create_context_with_permissions(permission_profile: str, timeout_seconds: int = 30) -> ExecutionContext:
    """
    Create an execution context with a predefined permission profile.
    
    Args:
        permission_profile: One of 'minimal', 'basic', 'automation', 'media', 'admin'
        timeout_seconds: Execution timeout in seconds
    
    Returns:
        ExecutionContext with the specified permissions
    """
    profiles = {
        'minimal': PermissionProfiles.minimal(),
        'basic': PermissionProfiles.basic(),
        'automation': PermissionProfiles.automation(),
        'media': PermissionProfiles.media(),
        'admin': PermissionProfiles.admin()
    }
    
    if permission_profile not in profiles:
        raise ValueError(f"Unknown profile: {permission_profile}. Available: {list(profiles.keys())}")
    
    return ExecutionContext.create_test_context(
        permissions=profiles[permission_profile],
        timeout=Duration.from_seconds(timeout_seconds)
    )


# Example usage functions
def example_basic_usage():
    """Example: Basic macro execution with different permission levels."""
    engine = MacroEngine()
    
    # Create a macro that needs sound permission
    sound_macro = create_test_macro("Play Alert", [CommandType.PLAY_SOUND])
    
    print("=== Basic Permission Example ===")
    
    # Try with basic permissions (includes SYSTEM_SOUND)
    basic_context = create_context_with_permissions('basic')
    result = engine.execute_macro(sound_macro, basic_context)
    print(f"Basic context result: {result.status}")  # Should be COMPLETED
    
    # Try with minimal permissions (no SYSTEM_SOUND)
    minimal_context = create_context_with_permissions('minimal')
    result = engine.execute_macro(sound_macro, minimal_context)
    print(f"Minimal context result: {result.status}")  # Should be FAILED
    

def example_custom_permissions():
    """Example: Creating custom permission sets."""
    engine = MacroEngine()
    
    print("=== Custom Permissions Example ===")
    
    # Create context with exactly the permissions you need
    custom_context = ExecutionContext.create_test_context(
        permissions=PermissionProfiles.custom(
            Permission.TEXT_INPUT,
            Permission.APPLICATION_CONTROL,
            Permission.FILE_ACCESS
        ),
        timeout=Duration.from_seconds(45)
    )
    
    print(f"Custom context has TEXT_INPUT: {custom_context.has_permission(Permission.TEXT_INPUT)}")
    print(f"Custom context has SYSTEM_SOUND: {custom_context.has_permission(Permission.SYSTEM_SOUND)}")
    

def example_checking_permissions():
    """Example: How to check what permissions are available."""
    context = create_context_with_permissions('automation')
    
    print("=== Permission Checking Example ===")
    print(f"Available permissions: {[p.value for p in context.permissions]}")
    
    # Check individual permissions
    checks = [
        Permission.TEXT_INPUT,
        Permission.SYSTEM_SOUND,
        Permission.APPLICATION_CONTROL,
        Permission.NETWORK_ACCESS,
        Permission.SYSTEM_CONTROL
    ]
    
    for permission in checks:
        has_perm = context.has_permission(permission)
        print(f"Has {permission.value}: {'✅' if has_perm else '❌'}")


def example_runtime_permission_upgrade():
    """Example: How to add permissions at runtime (by creating new context)."""
    print("=== Runtime Permission Upgrade Example ===")
    
    # Start with basic permissions
    basic_context = create_context_with_permissions('basic')
    print(f"Basic permissions: {[p.value for p in basic_context.permissions]}")
    
    # Upgrade to include file access
    upgraded_permissions = basic_context.permissions | {Permission.FILE_ACCESS, Permission.NETWORK_ACCESS}
    upgraded_context = ExecutionContext.create_test_context(
        permissions=upgraded_permissions,
        timeout=basic_context.timeout
    )
    
    print(f"Upgraded permissions: {[p.value for p in upgraded_context.permissions]}")


if __name__ == "__main__":
    # Run examples
    try:
        example_basic_usage()
        print()
        example_custom_permissions()
        print()
        example_checking_permissions()
        print()
        example_runtime_permission_upgrade()
        
    except Exception as e:
        print(f"Example failed: {e}")
        print("Make sure you're running from the project root directory.")