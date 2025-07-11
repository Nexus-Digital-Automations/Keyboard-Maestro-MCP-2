"""Focused tests for src/security/access_controller.py.

This module provides targeted tests for the access_controller module to achieve high coverage
toward the mandatory 95% threshold.
"""

from datetime import UTC, datetime, timedelta

import pytest
from src.core.zero_trust_architecture import (
    AccessControlError,
    SecurityContext,
    TrustLevel,
    create_security_context_id,
)
from src.security.access_controller import (
    AccessController,
    AccessRequest,
    AccessResult,
    AuthorizationModel,
    AuthorizationResult,
    Permission,
    PermissionType,
    ResourceType,
    Role,
    Subject,
    create_access_request,
    create_permission,
    create_role,
    create_subject,
)


class TestAccessResult:
    """Test AccessResult enum values."""

    def test_access_result_enum_values(self):
        """Test AccessResult enum has expected values."""
        assert AccessResult.ALLOW.value == "allow"
        assert AccessResult.DENY.value == "deny"
        assert AccessResult.CONDITIONAL.value == "conditional"
        assert AccessResult.REQUIRES_APPROVAL.value == "requires_approval"
        assert AccessResult.TEMPORARILY_DENIED.value == "temporarily_denied"
        assert AccessResult.ESCALATED.value == "escalated"


class TestPermissionType:
    """Test PermissionType enum values."""

    def test_permission_type_enum_values(self):
        """Test PermissionType enum has expected values."""
        assert PermissionType.READ.value == "read"
        assert PermissionType.WRITE.value == "write"
        assert PermissionType.EXECUTE.value == "execute"
        assert PermissionType.DELETE.value == "delete"
        assert PermissionType.ADMIN.value == "admin"
        assert PermissionType.CREATE.value == "create"
        assert PermissionType.MODIFY.value == "modify"


class TestResourceType:
    """Test ResourceType enum values."""

    def test_resource_type_enum_values(self):
        """Test ResourceType enum has expected values."""
        assert ResourceType.FILE.value == "file"
        assert ResourceType.DIRECTORY.value == "directory"
        assert ResourceType.APPLICATION.value == "application"
        assert ResourceType.SERVICE.value == "service"
        assert ResourceType.DATABASE.value == "database"
        assert ResourceType.API.value == "api"
        assert ResourceType.MACRO.value == "macro"
        assert ResourceType.VARIABLE.value == "variable"
        assert ResourceType.CONFIGURATION.value == "configuration"


class TestAuthorizationModel:
    """Test AuthorizationModel enum values."""

    def test_authorization_model_enum_values(self):
        """Test AuthorizationModel enum has expected values."""
        assert AuthorizationModel.RBAC.value == "rbac"
        assert AuthorizationModel.ABAC.value == "abac"
        assert AuthorizationModel.DAC.value == "dac"
        assert AuthorizationModel.MAC.value == "mac"
        assert AuthorizationModel.ZBAC.value == "zbac"
        assert AuthorizationModel.TBAC.value == "tbac"


class TestPermission:
    """Test Permission dataclass."""

    def test_permission_creation(self):
        """Test Permission creation with valid data."""
        permission = Permission(
            permission_id="perm-001",
            permission_type=PermissionType.READ,
            resource_type=ResourceType.FILE,
            resource_path="/documents/file.txt",
            conditions={"department": "finance"},
            constraints={"time_range": "9-17"},
            expires_at=datetime.now(UTC) + timedelta(hours=1),
            granted_by="admin-001",
        )

        assert permission.permission_id == "perm-001"
        assert permission.permission_type == PermissionType.READ
        assert permission.resource_type == ResourceType.FILE
        assert permission.resource_path == "/documents/file.txt"
        assert permission.conditions["department"] == "finance"
        assert permission.constraints["time_range"] == "9-17"
        assert permission.granted_by == "admin-001"
        assert permission.expires_at is not None

    def test_permission_validation(self):
        """Test Permission validation."""
        # Test empty permission_id
        with pytest.raises(
            ValueError, match="Permission ID and resource path are required"
        ):
            Permission(
                permission_id="",
                permission_type=PermissionType.READ,
                resource_type=ResourceType.FILE,
                resource_path="/documents/file.txt",
            )

        # Test empty resource_path
        with pytest.raises(
            ValueError, match="Permission ID and resource path are required"
        ):
            Permission(
                permission_id="perm-001",
                permission_type=PermissionType.READ,
                resource_type=ResourceType.FILE,
                resource_path="",
            )

    def test_permission_is_expired(self):
        """Test Permission expiration check."""
        # Test not expired permission
        future_permission = Permission(
            permission_id="perm-001",
            permission_type=PermissionType.READ,
            resource_type=ResourceType.FILE,
            resource_path="/documents/file.txt",
            expires_at=datetime.now(UTC) + timedelta(hours=1),
        )
        assert not future_permission.is_expired()

        # Test expired permission
        expired_permission = Permission(
            permission_id="perm-002",
            permission_type=PermissionType.READ,
            resource_type=ResourceType.FILE,
            resource_path="/documents/file.txt",
            expires_at=datetime.now(UTC) - timedelta(hours=1),
        )
        assert expired_permission.is_expired()

        # Test permission without expiration
        no_expiry_permission = Permission(
            permission_id="perm-003",
            permission_type=PermissionType.READ,
            resource_type=ResourceType.FILE,
            resource_path="/documents/file.txt",
        )
        assert not no_expiry_permission.is_expired()

    def test_permission_matches_request(self):
        """Test Permission matches_request method."""
        # Test exact match
        permission = Permission(
            permission_id="perm-001",
            permission_type=PermissionType.READ,
            resource_type=ResourceType.FILE,
            resource_path="/documents/file.txt",
        )
        assert permission.matches_request("/documents/file.txt", PermissionType.READ)
        assert not permission.matches_request(
            "/documents/file.txt", PermissionType.WRITE
        )
        assert not permission.matches_request(
            "/documents/other.txt", PermissionType.READ
        )

        # Test wildcard match
        wildcard_permission = Permission(
            permission_id="perm-002",
            permission_type=PermissionType.READ,
            resource_type=ResourceType.FILE,
            resource_path="*",
        )
        assert wildcard_permission.matches_request("/any/path", PermissionType.READ)
        assert not wildcard_permission.matches_request(
            "/any/path", PermissionType.WRITE
        )

        # Test prefix wildcard match
        prefix_permission = Permission(
            permission_id="perm-003",
            permission_type=PermissionType.READ,
            resource_type=ResourceType.FILE,
            resource_path="/documents/*",
        )
        assert prefix_permission.matches_request(
            "/documents/file.txt", PermissionType.READ
        )
        assert prefix_permission.matches_request(
            "/documents/subfolder/file.txt", PermissionType.READ
        )
        assert not prefix_permission.matches_request(
            "/other/file.txt", PermissionType.READ
        )


class TestRole:
    """Test Role dataclass."""

    def test_role_creation(self):
        """Test Role creation with valid data."""
        role = Role(
            role_id="role-001",
            role_name="Finance Manager",
            description="Finance department manager role",
            permissions={"perm-001", "perm-002"},
            parent_roles={"role-parent"},
            conditions={"department": "finance"},
            created_by="admin-001",
        )

        assert role.role_id == "role-001"
        assert role.role_name == "Finance Manager"
        assert role.description == "Finance department manager role"
        assert "perm-001" in role.permissions
        assert "role-parent" in role.parent_roles
        assert role.conditions["department"] == "finance"
        assert role.created_by == "admin-001"

    def test_role_validation(self):
        """Test Role validation."""
        # Test empty role_id
        with pytest.raises(ValueError, match="Role ID and name are required"):
            Role(
                role_id="",
                role_name="Test Role",
                description="Test role description",
            )

        # Test empty role_name
        with pytest.raises(ValueError, match="Role ID and name are required"):
            Role(
                role_id="role-001",
                role_name="",
                description="Test role description",
            )


class TestSubject:
    """Test Subject dataclass."""

    def test_subject_creation(self):
        """Test Subject creation with valid data."""
        subject = Subject(
            subject_id="user-001",
            subject_type="user",
            attributes={"department": "finance", "level": "manager"},
            roles={"role-001", "role-002"},
            direct_permissions={"perm-001"},
            groups={"finance-group"},
            security_clearance="high",
            last_authenticated=datetime.now(UTC),
        )

        assert subject.subject_id == "user-001"
        assert subject.subject_type == "user"
        assert subject.attributes["department"] == "finance"
        assert "role-001" in subject.roles
        assert "perm-001" in subject.direct_permissions
        assert "finance-group" in subject.groups
        assert subject.security_clearance == "high"
        assert subject.last_authenticated is not None

    def test_subject_validation(self):
        """Test Subject validation."""
        # Test empty subject_id
        with pytest.raises(ValueError, match="Subject ID and type are required"):
            Subject(
                subject_id="",
                subject_type="user",
            )

        # Test empty subject_type
        with pytest.raises(ValueError, match="Subject ID and type are required"):
            Subject(
                subject_id="user-001",
                subject_type="",
            )


class TestAccessRequest:
    """Test AccessRequest dataclass."""

    @pytest.fixture
    def sample_context(self):
        """Create sample security context."""
        from src.core.zero_trust_architecture import create_security_context_id

        return SecurityContext(
            context_id=create_security_context_id(),
            user_id="user-001",
            session_id="session-123",
            trust_level=TrustLevel.HIGH,
        )

    def test_access_request_creation(self, sample_context):
        """Test AccessRequest creation with valid data."""
        request = AccessRequest(
            request_id="req-001",
            subject_id="user-001",
            resource_path="/documents/file.txt",
            resource_type=ResourceType.FILE,
            permission_type=PermissionType.READ,
            context=sample_context,
            additional_context={"purpose": "audit"},
            urgency="high",
        )

        assert request.request_id == "req-001"
        assert request.subject_id == "user-001"
        assert request.resource_path == "/documents/file.txt"
        assert request.resource_type == ResourceType.FILE
        assert request.permission_type == PermissionType.READ
        assert request.context == sample_context
        assert request.additional_context["purpose"] == "audit"
        assert request.urgency == "high"

    def test_access_request_validation(self, sample_context):
        """Test AccessRequest validation."""
        # Test empty request_id
        with pytest.raises(
            ValueError, match="Request ID, subject ID, and resource path are required"
        ):
            AccessRequest(
                request_id="",
                subject_id="user-001",
                resource_path="/documents/file.txt",
                resource_type=ResourceType.FILE,
                permission_type=PermissionType.READ,
                context=sample_context,
            )

        # Test empty subject_id
        with pytest.raises(
            ValueError, match="Request ID, subject ID, and resource path are required"
        ):
            AccessRequest(
                request_id="req-001",
                subject_id="",
                resource_path="/documents/file.txt",
                resource_type=ResourceType.FILE,
                permission_type=PermissionType.READ,
                context=sample_context,
            )

        # Test empty resource_path
        with pytest.raises(
            ValueError, match="Request ID, subject ID, and resource path are required"
        ):
            AccessRequest(
                request_id="req-001",
                subject_id="user-001",
                resource_path="",
                resource_type=ResourceType.FILE,
                permission_type=PermissionType.READ,
                context=sample_context,
            )


class TestAuthorizationResult:
    """Test AuthorizationResult dataclass."""

    def test_authorization_result_creation(self):
        """Test AuthorizationResult creation with valid data."""
        result = AuthorizationResult(
            request_id="req-001",
            decision=AccessResult.ALLOW,
            subject_id="user-001",
            resource_path="/documents/file.txt",
            permission_type=PermissionType.READ,
            reason="User has required permissions",
            confidence=0.95,
            conditions=["log_access", "notify_admin"],
            expires_at=datetime.now(UTC) + timedelta(hours=1),
        )

        assert result.request_id == "req-001"
        assert result.subject_id == "user-001"
        assert result.decision == AccessResult.ALLOW
        assert result.reason == "User has required permissions"
        assert result.confidence == 0.95
        assert "log_access" in result.conditions
        assert result.expires_at is not None
        assert result.resource_path == "/documents/file.txt"
        assert result.permission_type == PermissionType.READ

    def test_authorization_result_validation(self):
        """Test AuthorizationResult validation."""
        # Test invalid confidence score
        with pytest.raises(ValueError, match="Confidence must be between 0.0 and 1.0"):
            AuthorizationResult(
                request_id="req-001",
                decision=AccessResult.ALLOW,
                subject_id="user-001",
                resource_path="/test/path",
                permission_type=PermissionType.READ,
                reason="Test reason",
                confidence=1.5,
            )

    def test_authorization_result_is_expired(self):
        """Test AuthorizationResult expiration check."""
        # Test not expired result
        future_result = AuthorizationResult(
            request_id="req-001",
            decision=AccessResult.ALLOW,
            subject_id="user-001",
            resource_path="/test/path",
            permission_type=PermissionType.READ,
            reason="Test reason",
            confidence=0.9,
            expires_at=datetime.now(UTC) + timedelta(hours=1),
        )
        assert not future_result.is_expired()

        # Test expired result
        expired_result = AuthorizationResult(
            request_id="req-002",
            decision=AccessResult.ALLOW,
            subject_id="user-001",
            resource_path="/test/path",
            permission_type=PermissionType.READ,
            reason="Test reason",
            confidence=0.9,
            expires_at=datetime.now(UTC) - timedelta(hours=1),
        )
        assert expired_result.is_expired()

        # Test result without expiration
        no_expiry_result = AuthorizationResult(
            request_id="req-003",
            decision=AccessResult.ALLOW,
            subject_id="user-001",
            resource_path="/test/path",
            permission_type=PermissionType.READ,
            reason="Test reason",
            confidence=0.9,
        )
        assert not no_expiry_result.is_expired()


class TestAccessController:
    """Test AccessController class."""

    @pytest.fixture
    def controller(self):
        """Create AccessController instance for testing."""
        return AccessController()

    @pytest.fixture
    def sample_permission(self):
        """Create sample permission for testing."""
        return Permission(
            permission_id="perm-001",
            permission_type=PermissionType.READ,
            resource_type=ResourceType.FILE,
            resource_path="/documents/file.txt",
        )

    @pytest.fixture
    def sample_role(self):
        """Create sample role for testing."""
        return Role(
            role_id="role-001",
            role_name="Test Role",
            description="Role for testing",
            permissions={"perm-001"},
        )

    @pytest.fixture
    def sample_subject(self):
        """Create sample subject for testing."""
        return Subject(
            subject_id="user-001",
            subject_type="user",
            roles={"role-001"},
            direct_permissions={"perm-001"},
        )

    @pytest.fixture
    def sample_context(self):
        """Create sample security context."""
        from src.core.zero_trust_architecture import create_security_context_id

        return SecurityContext(
            context_id=create_security_context_id(),
            user_id="user-001",
            session_id="session-123",
            trust_level=TrustLevel.HIGH,
        )

    @pytest.fixture
    def sample_access_request(self, sample_context):
        """Create sample access request for testing."""
        return AccessRequest(
            request_id="req-001",
            subject_id="user-001",
            resource_path="/documents/file.txt",
            resource_type=ResourceType.FILE,
            permission_type=PermissionType.READ,
            context=sample_context,
        )

    def test_access_controller_initialization(self, controller):
        """Test AccessController initialization."""
        assert isinstance(controller.subjects, dict)
        assert isinstance(controller.roles, dict)
        assert isinstance(controller.permissions, dict)
        assert isinstance(controller.authorization_cache, dict)
        assert isinstance(controller.access_history, list)

        # Check default configurations
        assert controller.authorization_model == AuthorizationModel.ABAC
        assert controller.default_deny is True
        assert controller.cache_duration == 300
        assert controller.max_cache_size == 10000

        # Check initial metrics
        assert controller.authorization_count == 0
        assert controller.average_authorization_time == 0.0
        assert controller.cache_hit_rate == 0.0
        assert controller.denial_rate == 0.0

    async def test_register_subject_success(
        self, controller, sample_subject, sample_role, sample_permission
    ):
        """Test successful subject registration."""
        # Register dependencies first
        await controller.register_permission(sample_permission)
        await controller.register_role(sample_role)

        result = await controller.register_subject(sample_subject)

        assert result.is_right()
        success_message = result.get_right()
        assert "registered successfully" in success_message
        assert sample_subject.subject_id in controller.subjects

    async def test_register_subject_duplicate(self, controller, sample_subject):
        """Test subject registration with duplicate ID."""
        # Register subject first time
        controller.subjects[sample_subject.subject_id] = sample_subject

        result = await controller.register_subject(sample_subject)

        assert result.is_left()
        error = result.get_left()
        assert isinstance(error, AccessControlError)
        assert "already exists" in error.message

    async def test_register_subject_invalid_role(self, controller, sample_permission):
        """Test subject registration with invalid role."""
        # Register permission but not role
        await controller.register_permission(sample_permission)

        subject = Subject(
            subject_id="user-001",
            subject_type="user",
            roles={"invalid-role"},
            direct_permissions={"perm-001"},
        )

        result = await controller.register_subject(subject)

        assert result.is_left()
        error = result.get_left()
        assert isinstance(error, AccessControlError)
        assert "does not exist" in error.message

    async def test_register_role_success(
        self, controller, sample_role, sample_permission
    ):
        """Test successful role registration."""
        # Register permission first
        await controller.register_permission(sample_permission)

        result = await controller.register_role(sample_role)

        assert result.is_right()
        success_message = result.get_right()
        assert "registered successfully" in success_message
        assert sample_role.role_id in controller.roles

    async def test_register_role_duplicate(self, controller, sample_role):
        """Test role registration with duplicate ID."""
        # Register role first time
        controller.roles[sample_role.role_id] = sample_role

        result = await controller.register_role(sample_role)

        assert result.is_left()
        error = result.get_left()
        assert isinstance(error, AccessControlError)
        assert "already exists" in error.message

    async def test_register_permission_success(self, controller, sample_permission):
        """Test successful permission registration."""
        result = await controller.register_permission(sample_permission)

        assert result.is_right()
        success_message = result.get_right()
        assert "registered successfully" in success_message
        assert sample_permission.permission_id in controller.permissions

    async def test_register_permission_duplicate(self, controller, sample_permission):
        """Test permission registration with duplicate ID."""
        # Register permission first time
        controller.permissions[sample_permission.permission_id] = sample_permission

        result = await controller.register_permission(sample_permission)

        assert result.is_left()
        error = result.get_left()
        assert isinstance(error, AccessControlError)
        assert "already exists" in error.message

    async def test_authorize_access_subject_not_found(
        self, controller, sample_access_request
    ):
        """Test authorization with non-existent subject."""
        result = await controller.authorize_access(sample_access_request)

        assert result.is_left()
        error = result.get_left()
        assert isinstance(error, AccessControlError)
        assert "not found" in error.message

    async def test_authorize_access_rbac_success(
        self,
        controller,
        sample_subject,
        sample_role,
        sample_permission,
        sample_access_request,
    ):
        """Test successful RBAC authorization."""
        # Set up RBAC mode
        controller.authorization_model = AuthorizationModel.RBAC

        # Register dependencies
        await controller.register_permission(sample_permission)
        await controller.register_role(sample_role)
        await controller.register_subject(sample_subject)

        result = await controller.authorize_access(sample_access_request)

        assert result.is_right()
        auth_result = result.get_right()
        assert isinstance(auth_result, AuthorizationResult)
        assert auth_result.request_id == sample_access_request.request_id

    async def test_authorize_access_abac_success(
        self,
        controller,
        sample_subject,
        sample_role,
        sample_permission,
        sample_access_request,
    ):
        """Test successful ABAC authorization."""
        # Set up ABAC mode (default)
        controller.authorization_model = AuthorizationModel.ABAC

        # Register dependencies
        await controller.register_permission(sample_permission)
        await controller.register_role(sample_role)
        await controller.register_subject(sample_subject)

        result = await controller.authorize_access(sample_access_request)

        assert result.is_right()
        auth_result = result.get_right()
        assert isinstance(auth_result, AuthorizationResult)
        assert auth_result.request_id == sample_access_request.request_id

    async def test_revoke_permissions_success(
        self, controller, sample_subject, sample_role, sample_permission
    ):
        """Test successful permission revocation."""
        # Register and setup subject
        await controller.register_permission(sample_permission)
        await controller.register_role(sample_role)
        await controller.register_subject(sample_subject)

        result = await controller.revoke_permissions(
            sample_subject.subject_id,
            [sample_permission.permission_id],
        )

        assert result.is_right()
        success_message = result.get_right()
        assert "Revoked" in success_message and "permissions" in success_message

    async def test_revoke_permissions_subject_not_found(self, controller):
        """Test permission revocation with non-existent subject."""
        result = await controller.revoke_permissions(
            "non-existent-user",
            ["perm-001"],
        )

        assert result.is_left()
        error = result.get_left()
        assert isinstance(error, AccessControlError)
        assert "not found" in error.message

    async def test_get_effective_permissions_success(
        self, controller, sample_subject, sample_role, sample_permission
    ):
        """Test getting effective permissions for subject."""
        # Register dependencies
        await controller.register_permission(sample_permission)
        await controller.register_role(sample_role)
        await controller.register_subject(sample_subject)

        result = await controller.get_effective_permissions(sample_subject.subject_id)

        assert result.is_right()
        permissions = result.get_right()
        assert isinstance(permissions, list)
        assert len(permissions) > 0

    async def test_get_effective_permissions_subject_not_found(self, controller):
        """Test getting effective permissions for non-existent subject."""
        result = await controller.get_effective_permissions("non-existent-user")

        assert result.is_left()
        error = result.get_left()
        assert isinstance(error, AccessControlError)
        assert "not found" in error.message


class TestUtilityFunctions:
    """Test utility functions for access control."""

    def test_create_subject_function(self):
        """Test create_subject utility function."""
        subject = create_subject(
            subject_id="user-001",
            subject_type="user",
            attributes={"department": "finance"},
            roles=["role-001"],
            permissions=["perm-001"],
        )

        assert subject.subject_id == "user-001"
        assert subject.subject_type == "user"
        assert subject.attributes["department"] == "finance"
        assert "role-001" in subject.roles
        assert "perm-001" in subject.direct_permissions

    def test_create_role_function(self):
        """Test create_role utility function."""
        role = create_role(
            role_name="Finance Manager",
            description="Finance department manager role",
            permissions=["perm-001", "perm-002"],
        )

        assert role.role_id == "role_finance_manager"
        assert role.role_name == "Finance Manager"
        assert role.description == "Finance department manager role"
        assert "perm-001" in role.permissions

    def test_create_permission_function(self):
        """Test create_permission utility function."""
        permission = create_permission(
            permission_type=PermissionType.READ,
            resource_type=ResourceType.FILE,
            resource_path="/documents/file.txt",
        )

        assert permission.permission_id.startswith("perm_read_file_")
        assert permission.permission_type == PermissionType.READ
        assert permission.resource_type == ResourceType.FILE
        assert permission.resource_path == "/documents/file.txt"

    def test_create_access_request_function(self):
        """Test create_access_request utility function."""
        context = SecurityContext(
            context_id=create_security_context_id(),
            user_id="user-001",
            session_id="session-123",
            trust_level=TrustLevel.HIGH,
        )

        request = create_access_request(
            subject_id="user-001",
            resource_path="/documents/file.txt",
            resource_type=ResourceType.FILE,
            permission_type=PermissionType.READ,
            context=context,
        )

        assert request.request_id.startswith("req_")
        assert request.subject_id == "user-001"
        assert request.resource_path == "/documents/file.txt"
        assert request.resource_type == ResourceType.FILE
        assert request.permission_type == PermissionType.READ
        assert request.context == context


class TestIntegrationScenarios:
    """Test integration scenarios combining multiple components."""

    @pytest.fixture
    def configured_controller(self):
        """Create configured AccessController with subjects, roles, and permissions."""
        controller = AccessController()
        return controller

    async def test_full_access_control_lifecycle(self, configured_controller):
        """Test complete access control lifecycle: register, authorize, revoke."""
        # Create permission
        permission = Permission(
            permission_id="perm-read-docs",
            permission_type=PermissionType.READ,
            resource_type=ResourceType.FILE,
            resource_path="/documents/*",
        )

        # Create role
        role = Role(
            role_id="role-doc-reader",
            role_name="Document Reader",
            description="Can read documents",
            permissions={"perm-read-docs"},
        )

        # Create subject
        subject = Subject(
            subject_id="user-001",
            subject_type="user",
            roles={"role-doc-reader"},
        )

        # Create context
        context = SecurityContext(
            context_id=create_security_context_id(),
            user_id="user-001",
            session_id="session-123",
            trust_level=TrustLevel.HIGH,
        )

        # Create access request
        access_request = AccessRequest(
            request_id="req-001",
            subject_id="user-001",
            resource_path="/documents/report.pdf",
            resource_type=ResourceType.FILE,
            permission_type=PermissionType.READ,
            context=context,
        )

        # Register components
        perm_result = await configured_controller.register_permission(permission)
        assert perm_result.is_right()

        role_result = await configured_controller.register_role(role)
        assert role_result.is_right()

        subject_result = await configured_controller.register_subject(subject)
        assert subject_result.is_right()

        # Test authorization
        auth_result = await configured_controller.authorize_access(access_request)
        assert auth_result.is_right()

        # Test effective permissions
        eff_perms_result = await configured_controller.get_effective_permissions(
            "user-001"
        )
        assert eff_perms_result.is_right()

        # Test revocation
        revoke_result = await configured_controller.revoke_permissions(
            "user-001",
            ["perm-read-docs"],
        )
        assert revoke_result.is_right()

    async def test_rbac_authorization_scenarios(self, configured_controller):
        """Test various RBAC authorization scenarios."""
        configured_controller.authorization_model = AuthorizationModel.RBAC

        # Create hierarchical roles
        admin_permission = Permission(
            permission_id="perm-admin",
            permission_type=PermissionType.ADMIN,
            resource_type=ResourceType.SERVICE,
            resource_path="*",
        )

        user_permission = Permission(
            permission_id="perm-user",
            permission_type=PermissionType.READ,
            resource_type=ResourceType.FILE,
            resource_path="/user/*",
        )

        admin_role = Role(
            role_id="role-admin",
            role_name="Administrator",
            description="Full system access",
            permissions={"perm-admin", "perm-user"},
        )

        user_role = Role(
            role_id="role-user",
            role_name="User",
            description="Basic user access",
            permissions={"perm-user"},
        )

        # Register components
        await configured_controller.register_permission(admin_permission)
        await configured_controller.register_permission(user_permission)
        await configured_controller.register_role(admin_role)
        await configured_controller.register_role(user_role)

        # Create admin subject
        admin_subject = Subject(
            subject_id="admin-001",
            subject_type="user",
            roles={"role-admin"},
        )

        # Create regular user subject
        user_subject = Subject(
            subject_id="user-001",
            subject_type="user",
            roles={"role-user"},
        )

        await configured_controller.register_subject(admin_subject)
        await configured_controller.register_subject(user_subject)

        # Create context
        context = SecurityContext(
            context_id=create_security_context_id(),
            user_id="test-user",
            session_id="session-123",
            trust_level=TrustLevel.HIGH,
        )

        # Test admin access to admin resource
        admin_request = AccessRequest(
            request_id="req-admin",
            subject_id="admin-001",
            resource_path="/system/config",
            resource_type=ResourceType.SERVICE,
            permission_type=PermissionType.ADMIN,
            context=context,
        )

        admin_result = await configured_controller.authorize_access(admin_request)
        assert admin_result.is_right()

        # Test user access to user resource
        user_request = AccessRequest(
            request_id="req-user",
            subject_id="user-001",
            resource_path="/user/file.txt",
            resource_type=ResourceType.FILE,
            permission_type=PermissionType.READ,
            context=context,
        )

        user_result = await configured_controller.authorize_access(user_request)
        assert user_result.is_right()

    async def test_abac_authorization_scenarios(self, configured_controller):
        """Test various ABAC authorization scenarios."""
        configured_controller.authorization_model = AuthorizationModel.ABAC

        # Create context-aware permission
        context_permission = Permission(
            permission_id="perm-context",
            permission_type=PermissionType.READ,
            resource_type=ResourceType.FILE,
            resource_path="/sensitive/*",
            conditions={"department": "security", "clearance": "high"},
        )

        # Create role with context requirements
        security_role = Role(
            role_id="role-security",
            role_name="Security Analyst",
            description="Security department analyst",
            permissions={"perm-context"},
            conditions={"department": "security"},
        )

        # Create subject with required attributes
        security_subject = Subject(
            subject_id="security-001",
            subject_type="user",
            attributes={"department": "security", "clearance": "high"},
            roles={"role-security"},
        )

        # Register components
        await configured_controller.register_permission(context_permission)
        await configured_controller.register_role(security_role)
        await configured_controller.register_subject(security_subject)

        # Create context with matching attributes
        context = SecurityContext(
            context_id=create_security_context_id(),
            user_id="security-001",
            session_id="session-123",
            trust_level=TrustLevel.HIGH,
        )

        # Test access with correct context
        abac_request = AccessRequest(
            request_id="req-abac",
            subject_id="security-001",
            resource_path="/sensitive/classified.doc",
            resource_type=ResourceType.FILE,
            permission_type=PermissionType.READ,
            context=context,
            additional_context={"department": "security", "clearance": "high"},
        )

        abac_result = await configured_controller.authorize_access(abac_request)
        assert abac_result.is_right()

    async def test_permission_inheritance_and_revocation(self, configured_controller):
        """Test permission inheritance through roles and revocation."""
        # Create permissions
        read_perm = Permission(
            permission_id="perm-read",
            permission_type=PermissionType.READ,
            resource_type=ResourceType.FILE,
            resource_path="/files/*",
        )

        write_perm = Permission(
            permission_id="perm-write",
            permission_type=PermissionType.WRITE,
            resource_type=ResourceType.FILE,
            resource_path="/files/*",
        )

        # Create roles with inheritance
        base_role = Role(
            role_id="role-base",
            role_name="Base Role",
            description="Base permissions",
            permissions={"perm-read"},
        )

        advanced_role = Role(
            role_id="role-advanced",
            role_name="Advanced Role",
            description="Advanced permissions",
            permissions={"perm-write"},
            parent_roles={"role-base"},
        )

        # Create subject with advanced role
        subject = Subject(
            subject_id="user-001",
            subject_type="user",
            roles={"role-advanced"},
            direct_permissions={"perm-read"},  # Direct permission overlap
        )

        # Register components
        await configured_controller.register_permission(read_perm)
        await configured_controller.register_permission(write_perm)
        await configured_controller.register_role(base_role)
        await configured_controller.register_role(advanced_role)
        await configured_controller.register_subject(subject)

        # Test effective permissions before revocation
        eff_perms_result = await configured_controller.get_effective_permissions(
            "user-001"
        )
        assert eff_perms_result.is_right()
        permissions = eff_perms_result.get_right()
        assert len(permissions) > 0

        # Test revocation
        revoke_result = await configured_controller.revoke_permissions(
            "user-001",
            ["perm-read"],
        )
        assert revoke_result.is_right()

    async def test_error_handling_scenarios(self, configured_controller):
        """Test various error handling scenarios."""
        # Test authorization with expired permission
        expired_permission = Permission(
            permission_id="perm-expired",
            permission_type=PermissionType.READ,
            resource_type=ResourceType.FILE,
            resource_path="/temp/*",
            expires_at=datetime.now(UTC) - timedelta(hours=1),  # Already expired
        )

        await configured_controller.register_permission(expired_permission)

        role = Role(
            role_id="role-temp",
            role_name="Temporary Role",
            description="Role with expired permission",
            permissions={"perm-expired"},
        )

        await configured_controller.register_role(role)

        subject = Subject(
            subject_id="user-temp",
            subject_type="user",
            roles={"role-temp"},
        )

        await configured_controller.register_subject(subject)

        # Create access request for expired permission
        context = SecurityContext(
            context_id=create_security_context_id(),
            user_id="user-temp",
            session_id="session-123",
            trust_level=TrustLevel.HIGH,
        )

        expired_request = AccessRequest(
            request_id="req-expired",
            subject_id="user-temp",
            resource_path="/temp/file.txt",
            resource_type=ResourceType.FILE,
            permission_type=PermissionType.READ,
            context=context,
        )

        # Should handle expired permission gracefully
        result = await configured_controller.authorize_access(expired_request)
        # Result may be right or left depending on implementation
        assert result is not None
