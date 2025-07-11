"""Comprehensive test coverage for cloud integration core module.

Tests the complete cloud integration type system including branded types, enums,
dataclasses, and business logic following ADDER+ methodology for enterprise cloud automation.
"""

from datetime import UTC, datetime, timedelta

from hypothesis import given
from hypothesis import strategies as st
from src.core.cloud_integration import (
    CloudAuthMethod,
    CloudCredentials,
    CloudError,
    CloudOperation,
    CloudProvider,
    CloudRegion,
    CloudResource,
    CloudSecurityLevel,
    CloudServiceType,
    create_cloud_credentials,
    create_cloud_resource,
)


class TestCloudProviderEnum:
    """Test CloudProvider enum values and behavior."""

    def test_cloud_provider_values(self):
        """Test all CloudProvider enum values."""
        assert CloudProvider.AWS.value == "aws"
        assert CloudProvider.AZURE.value == "azure"
        assert CloudProvider.GOOGLE_CLOUD.value == "gcp"
        assert CloudProvider.ALIBABA_CLOUD.value == "alibaba"
        assert CloudProvider.DIGITAL_OCEAN.value == "digitalocean"
        assert CloudProvider.LINODE.value == "linode"
        assert CloudProvider.GENERIC.value == "generic"
        assert CloudProvider.MULTI_CLOUD.value == "multi"

    def test_cloud_provider_enum_complete(self):
        """Test CloudProvider enum completeness."""
        expected_providers = {
            "aws",
            "azure",
            "gcp",
            "alibaba",
            "digitalocean",
            "linode",
            "generic",
            "multi",
        }
        actual_providers = {cp.value for cp in CloudProvider}
        assert actual_providers == expected_providers


class TestCloudServiceTypeEnum:
    """Test CloudServiceType enum values and behavior."""

    def test_cloud_service_type_values(self):
        """Test all CloudServiceType enum values."""
        assert CloudServiceType.STORAGE.value == "storage"
        assert CloudServiceType.COMPUTE.value == "compute"
        assert CloudServiceType.DATABASE.value == "database"
        assert CloudServiceType.MESSAGING.value == "messaging"
        assert CloudServiceType.AI_ML.value == "ai_ml"
        assert CloudServiceType.MONITORING.value == "monitoring"
        assert CloudServiceType.NETWORKING.value == "networking"
        assert CloudServiceType.SECURITY.value == "security"
        assert CloudServiceType.ANALYTICS.value == "analytics"
        assert CloudServiceType.SERVERLESS.value == "serverless"

    def test_service_type_enum_complete(self):
        """Test CloudServiceType enum completeness."""
        expected_services = {
            "storage",
            "compute",
            "database",
            "messaging",
            "ai_ml",
            "monitoring",
            "networking",
            "security",
            "analytics",
            "serverless",
        }
        actual_services = {st.value for st in CloudServiceType}
        assert actual_services == expected_services


class TestCloudAuthMethodEnum:
    """Test CloudAuthMethod enum values and behavior."""

    def test_cloud_auth_method_values(self):
        """Test all CloudAuthMethod enum values."""
        assert CloudAuthMethod.API_KEY.value == "api_key"
        assert CloudAuthMethod.SERVICE_ACCOUNT.value == "service_account"
        assert CloudAuthMethod.MANAGED_IDENTITY.value == "managed_identity"
        assert CloudAuthMethod.ROLE_BASED.value == "role_based"
        assert CloudAuthMethod.OAUTH2.value == "oauth2"
        assert CloudAuthMethod.ACCESS_TOKEN.value == "access_token"
        assert CloudAuthMethod.CLIENT_CERTIFICATE.value == "client_certificate"


class TestCloudSecurityLevelEnum:
    """Test CloudSecurityLevel enum values and behavior."""

    def test_cloud_security_level_values(self):
        """Test all CloudSecurityLevel enum values."""
        assert CloudSecurityLevel.BASIC.value == "basic"
        assert CloudSecurityLevel.STANDARD.value == "standard"
        assert CloudSecurityLevel.HIGH.value == "high"
        assert CloudSecurityLevel.ENTERPRISE.value == "enterprise"
        assert CloudSecurityLevel.GOVERNMENT.value == "government"


class TestCloudRegionEnum:
    """Test CloudRegion enum values for global deployment."""

    def test_aws_region_values(self):
        """Test AWS region enum values."""
        assert CloudRegion.US_EAST_1.value == "us-east-1"
        assert CloudRegion.US_WEST_2.value == "us-west-2"
        assert CloudRegion.EU_WEST_1.value == "eu-west-1"
        assert CloudRegion.AP_SOUTHEAST_1.value == "ap-southeast-1"

    def test_azure_region_values(self):
        """Test Azure region enum values."""
        assert CloudRegion.AZURE_EAST_US.value == "eastus"
        assert CloudRegion.AZURE_WEST_EUROPE.value == "westeurope"
        assert CloudRegion.AZURE_SOUTHEAST_ASIA.value == "southeastasia"

    def test_gcp_region_values(self):
        """Test GCP region enum values."""
        assert CloudRegion.GCP_US_CENTRAL1.value == "us-central1"
        assert CloudRegion.GCP_EUROPE_WEST1.value == "europe-west1"
        assert CloudRegion.GCP_ASIA_SOUTHEAST1.value == "asia-southeast1"


class TestCloudError:
    """Test CloudError class methods and factory functions."""

    def test_cloud_error_creation(self):
        """Test basic CloudError creation."""
        error = CloudError("TEST_ERROR", "Test error message")
        assert error.error_type == "TEST_ERROR"
        assert error.message == "Test error message"
        assert error.details == {}
        assert isinstance(error.timestamp, datetime)

    def test_authentication_failed_error(self):
        """Test authentication failed error creation."""
        error = CloudError.authentication_failed("Invalid credentials")
        assert error.error_type == "AUTHENTICATION_FAILED"
        assert "Invalid credentials" in error.message
        assert "Cloud authentication failed" in error.message

    def test_connection_failed_error(self):
        """Test connection failed error creation."""
        error = CloudError.connection_failed("Network timeout")
        assert error.error_type == "CONNECTION_FAILED"
        assert "Network timeout" in error.message
        assert "Cloud connection failed" in error.message

    def test_unsupported_provider_error(self):
        """Test unsupported provider error creation."""
        error = CloudError.unsupported_provider(CloudProvider.ALIBABA_CLOUD)
        assert error.error_type == "UNSUPPORTED_PROVIDER"
        assert "alibaba" in error.message
        assert "not supported" in error.message

    def test_resource_creation_failed_error(self):
        """Test resource creation failed error."""
        error = CloudError.resource_creation_failed("Insufficient permissions")
        assert error.error_type == "RESOURCE_CREATION_FAILED"
        assert "Insufficient permissions" in error.message

    def test_session_not_found_error(self):
        """Test session not found error."""
        session_id = "session_123"
        error = CloudError.session_not_found(session_id)
        assert error.error_type == "SESSION_NOT_FOUND"
        assert session_id in error.message

    def test_insecure_auth_method_error(self):
        """Test insecure auth method error."""
        error = CloudError.insecure_auth_method(CloudAuthMethod.API_KEY)
        assert error.error_type == "INSECURE_AUTH_METHOD"
        assert "api_key" in error.message
        assert "not allowed for enterprise" in error.message

    def test_orchestration_failed_error(self):
        """Test orchestration failed error."""
        error = CloudError.orchestration_failed("Deployment conflict")
        assert error.error_type == "ORCHESTRATION_FAILED"
        assert "Multi-cloud orchestration failed" in error.message
        assert "Deployment conflict" in error.message

    def test_monitoring_failed_error(self):
        """Test monitoring failed error."""
        error = CloudError.monitoring_failed("Metrics unavailable")
        assert error.error_type == "MONITORING_FAILED"
        assert "Resource monitoring failed" in error.message
        assert "Metrics unavailable" in error.message

    def test_azure_specific_errors(self):
        """Test Azure-specific error methods."""
        no_subs_error = CloudError.no_subscriptions_found()
        assert no_subs_error.error_type == "NO_SUBSCRIPTIONS_FOUND"
        assert "No Azure subscriptions found" in no_subs_error.message

        missing_sub_error = CloudError.missing_subscription_id()
        assert missing_sub_error.error_type == "MISSING_SUBSCRIPTION_ID"
        assert "Azure subscription ID required" in missing_sub_error.message

    def test_gcp_specific_errors(self):
        """Test GCP-specific error methods."""
        error = CloudError.missing_service_account_file()
        assert error.error_type == "MISSING_SERVICE_ACCOUNT_FILE"
        assert "GCP service account file required" in error.message

    def test_security_specific_errors(self):
        """Test security-specific error methods."""
        incomplete_creds = CloudError.incomplete_credentials()
        assert incomplete_creds.error_type == "INCOMPLETE_CREDENTIALS"
        assert "Cloud credentials are incomplete" in incomplete_creds.message

        encryption_error = CloudError.encryption_required_for_storage()
        assert encryption_error.error_type == "ENCRYPTION_REQUIRED_FOR_STORAGE"
        assert "Encryption required for storage" in encryption_error.message

        public_access_error = CloudError.public_access_not_allowed()
        assert public_access_error.error_type == "PUBLIC_ACCESS_NOT_ALLOWED"
        assert "Public access not allowed" in public_access_error.message


class TestCloudCredentials:
    """Test CloudCredentials dataclass functionality."""

    def test_aws_api_key_credentials(self):
        """Test AWS API key credentials creation."""
        creds = CloudCredentials(
            provider=CloudProvider.AWS,
            auth_method=CloudAuthMethod.API_KEY,
            access_key="AKIATEST123",
            secret_key="secret123",  # noqa: S106 # Test credential
            region="us-east-1",
        )

        assert creds.provider == CloudProvider.AWS
        assert creds.auth_method == CloudAuthMethod.API_KEY
        assert creds.access_key == "AKIATEST123"
        assert creds.secret_key == "secret123"  # noqa: S105 # Test credential
        assert creds.region == "us-east-1"
        assert creds.security_level == CloudSecurityLevel.ENTERPRISE

    def test_azure_service_account_credentials(self):
        """Test Azure service account credentials creation."""
        creds = CloudCredentials(
            provider=CloudProvider.AZURE,
            auth_method=CloudAuthMethod.SERVICE_ACCOUNT,
            tenant_id="tenant-123",
            client_id="client-456",
            client_secret="secret-789",  # noqa: S106 # Test credential
            security_level=CloudSecurityLevel.HIGH,
        )

        assert creds.provider == CloudProvider.AZURE
        assert creds.auth_method == CloudAuthMethod.SERVICE_ACCOUNT
        assert creds.tenant_id == "tenant-123"
        assert creds.client_id == "client-456"
        assert creds.client_secret == "secret-789"  # noqa: S105 # Test credential
        assert creds.security_level == CloudSecurityLevel.HIGH

    def test_gcp_service_account_credentials(self):
        """Test GCP service account credentials creation."""
        creds = CloudCredentials(
            provider=CloudProvider.GOOGLE_CLOUD,
            auth_method=CloudAuthMethod.SERVICE_ACCOUNT,
            service_account_file="/path/to/service-account.json",
        )

        assert creds.provider == CloudProvider.GOOGLE_CLOUD
        assert creds.auth_method == CloudAuthMethod.SERVICE_ACCOUNT
        assert creds.service_account_file == "/path/to/service-account.json"

    def test_credentials_expiry_handling(self):
        """Test credentials expiry functionality."""
        # Test with explicit expiry
        future_time = datetime.now(UTC) + timedelta(hours=2)
        creds = CloudCredentials(
            provider=CloudProvider.AWS,
            auth_method=CloudAuthMethod.API_KEY,
            access_key="test",
            secret_key="test",  # noqa: S106 # Test credential
            expires_at=future_time,
        )

        assert not creds.is_expired()
        assert creds.expires_at == future_time

    def test_expired_credentials(self):
        """Test expired credentials detection."""
        past_time = datetime.now(UTC) - timedelta(hours=1)
        creds = CloudCredentials(
            provider=CloudProvider.AWS,
            auth_method=CloudAuthMethod.API_KEY,
            access_key="test",
            secret_key="test",  # noqa: S106 # Test credential
            expires_at=past_time,
        )

        assert creds.is_expired()

    def test_token_credentials_default_expiry(self):
        """Test token credentials get default 1-hour expiry."""
        creds = CloudCredentials(
            provider=CloudProvider.AZURE,
            auth_method=CloudAuthMethod.ACCESS_TOKEN,
            token="bearer_token_123",  # noqa: S106 # Test credential
        )

        assert creds.expires_at is not None
        # Should expire within approximately 1 hour
        expected_expiry = datetime.now(UTC) + timedelta(hours=1)
        time_diff = abs((creds.expires_at - expected_expiry).total_seconds())
        assert time_diff < 60  # Within 1 minute tolerance

    def test_safe_representation(self):
        """Test safe representation of credentials."""
        creds = CloudCredentials(
            provider=CloudProvider.AWS,
            auth_method=CloudAuthMethod.API_KEY,
            access_key="AKIATEST123",
            secret_key="secret123",  # noqa: S106 # Test credential
            region="us-west-2",
        )

        safe_repr = creds.get_safe_representation()

        assert safe_repr["provider"] == "aws"
        assert safe_repr["auth_method"] == "api_key"
        assert safe_repr["region"] == "us-west-2"
        assert safe_repr["security_level"] == "enterprise"
        assert safe_repr["has_access_key"] is True
        assert safe_repr["has_secret_key"] is True
        assert safe_repr["has_service_account"] is False
        assert safe_repr["has_token"] is False
        assert safe_repr["is_expired"] is False

        # Sensitive data should not be in safe representation
        assert "AKIATEST123" not in str(safe_repr)
        assert "secret123" not in str(safe_repr)

    def test_managed_identity_credentials(self):
        """Test managed identity credentials (no explicit secrets)."""
        creds = CloudCredentials(
            provider=CloudProvider.AZURE,
            auth_method=CloudAuthMethod.MANAGED_IDENTITY,
        )

        assert creds.provider == CloudProvider.AZURE
        assert creds.auth_method == CloudAuthMethod.MANAGED_IDENTITY
        # Managed identity doesn't require explicit credentials
        assert creds.access_key is None
        assert creds.secret_key is None


class TestCloudResource:
    """Test CloudResource dataclass functionality."""

    def test_cloud_resource_creation(self):
        """Test CloudResource creation with valid parameters."""
        resource = CloudResource(
            resource_id="bucket-123",
            provider=CloudProvider.AWS,
            service_type=CloudServiceType.STORAGE,
            resource_type="s3-bucket",
            region="us-east-1",
            configuration={"versioning": True, "encryption": "AES256"},
            tags={"environment": "production", "team": "engineering"},
        )

        assert resource.resource_id == "bucket-123"
        assert resource.provider == CloudProvider.AWS
        assert resource.service_type == CloudServiceType.STORAGE
        assert resource.resource_type == "s3-bucket"
        assert resource.region == "us-east-1"
        assert resource.configuration["versioning"] is True
        assert resource.tags["environment"] == "production"
        assert isinstance(resource.created_at, datetime)
        assert isinstance(resource.updated_at, datetime)
        assert resource.status == "unknown"

    def test_resource_cost_estimation(self):
        """Test resource cost estimation functionality."""
        storage_resource = CloudResource(
            resource_id="storage-1",
            provider=CloudProvider.AWS,
            service_type=CloudServiceType.STORAGE,
            resource_type="s3-bucket",
            region="us-east-1",
            configuration={},
        )

        compute_resource = CloudResource(
            resource_id="instance-1",
            provider=CloudProvider.AWS,
            service_type=CloudServiceType.COMPUTE,
            resource_type="ec2-instance",
            region="us-east-1",
            configuration={},
        )

        # Test cost estimates are assigned
        assert storage_resource.cost_estimate == 0.023  # per GB for storage
        assert compute_resource.cost_estimate == 72.0  # per month for compute

    def test_aws_arn_generation(self):
        """Test AWS ARN generation for resources."""
        # Test S3 resource ARN
        s3_resource = CloudResource(
            resource_id="my-bucket",
            provider=CloudProvider.AWS,
            service_type=CloudServiceType.STORAGE,
            resource_type="bucket",
            region="us-west-2",
            configuration={},
        )

        arn = s3_resource.get_arn()
        assert arn == "arn:aws:s3:us-west-2:*:bucket/my-bucket"

        # Test EC2 resource ARN
        ec2_resource = CloudResource(
            resource_id="i-1234567890abcdef0",
            provider=CloudProvider.AWS,
            service_type=CloudServiceType.COMPUTE,
            resource_type="instance",
            region="eu-west-1",
            configuration={},
        )

        arn = ec2_resource.get_arn()
        assert arn == "arn:aws:ec2:eu-west-1:*:instance/i-1234567890abcdef0"

        # Test Lambda resource ARN
        lambda_resource = CloudResource(
            resource_id="my-function",
            provider=CloudProvider.AWS,
            service_type=CloudServiceType.SERVERLESS,
            resource_type="function",
            region="ap-southeast-1",
            configuration={},
        )

        arn = lambda_resource.get_arn()
        assert arn == "arn:aws:lambda:ap-southeast-1:*:function/my-function"

    def test_non_aws_arn_generation(self):
        """Test ARN generation for non-AWS providers."""
        azure_resource = CloudResource(
            resource_id="storage-account-1",
            provider=CloudProvider.AZURE,
            service_type=CloudServiceType.STORAGE,
            resource_type="storage-account",
            region="eastus",
            configuration={},
        )

        arn = azure_resource.get_arn()
        assert arn is None  # Non-AWS providers don't have ARNs

    def test_unsupported_service_type_cost(self):
        """Test cost estimation for unsupported service types."""
        monitoring_resource = CloudResource(
            resource_id="monitor-1",
            provider=CloudProvider.AWS,
            service_type=CloudServiceType.MONITORING,
            resource_type="cloudwatch-dashboard",
            region="us-east-1",
            configuration={},
        )

        # Should use default cost for unsupported service types
        assert monitoring_resource.cost_estimate == 10.0


class TestCloudOperation:
    """Test CloudOperation dataclass functionality."""

    def test_cloud_operation_creation(self):
        """Test CloudOperation creation with valid parameters."""
        resource = CloudResource(
            resource_id="test-resource",
            provider=CloudProvider.AWS,
            service_type=CloudServiceType.COMPUTE,
            resource_type="ec2-instance",
            region="us-east-1",
            configuration={},
        )

        operation = CloudOperation(
            operation_id="op-123",
            operation_type="create_instance",
            provider=CloudProvider.AWS,
            resource=resource,
            parameters={"instance_type": "t3.medium", "ami_id": "ami-12345"},
        )

        assert operation.operation_id == "op-123"
        assert operation.operation_type == "create_instance"
        assert operation.provider == CloudProvider.AWS
        assert operation.resource == resource
        assert operation.parameters["instance_type"] == "t3.medium"
        assert operation.status == "pending"
        assert isinstance(operation.started_at, datetime)
        assert operation.completed_at is None
        assert operation.retry_count == 0

    def test_operation_completion_status(self):
        """Test operation completion status detection."""
        resource = CloudResource(
            resource_id="test-resource",
            provider=CloudProvider.AWS,
            service_type=CloudServiceType.STORAGE,
            resource_type="s3-bucket",
            region="us-east-1",
            configuration={},
        )

        # Test pending operation
        pending_op = CloudOperation(
            operation_id="op-1",
            operation_type="create_bucket",
            provider=CloudProvider.AWS,
            resource=resource,
            parameters={},
            status="pending",
        )
        assert not pending_op.is_completed()

        # Test completed operation
        completed_op = CloudOperation(
            operation_id="op-2",
            operation_type="create_bucket",
            provider=CloudProvider.AWS,
            resource=resource,
            parameters={},
            status="completed",
        )
        assert completed_op.is_completed()

        # Test failed operation
        failed_op = CloudOperation(
            operation_id="op-3",
            operation_type="create_bucket",
            provider=CloudProvider.AWS,
            resource=resource,
            parameters={},
            status="failed",
        )
        assert failed_op.is_completed()

    def test_operation_duration_calculation(self):
        """Test operation duration calculation."""
        resource = CloudResource(
            resource_id="test-resource",
            provider=CloudProvider.AWS,
            service_type=CloudServiceType.DATABASE,
            resource_type="rds-instance",
            region="us-west-2",
            configuration={},
        )

        start_time = datetime.now(UTC)
        end_time = start_time + timedelta(minutes=5, seconds=30)

        operation = CloudOperation(
            operation_id="op-duration-test",
            operation_type="create_database",
            provider=CloudProvider.AWS,
            resource=resource,
            parameters={},
            started_at=start_time,
            completed_at=end_time,
        )

        duration = operation.get_duration()
        assert duration is not None
        assert duration.total_seconds() == 330  # 5 minutes 30 seconds

        # Test incomplete operation
        incomplete_op = CloudOperation(
            operation_id="op-incomplete",
            operation_type="create_database",
            provider=CloudProvider.AWS,
            resource=resource,
            parameters={},
            started_at=start_time,
            completed_at=None,
        )

        assert incomplete_op.get_duration() is None


class TestFactoryFunctions:
    """Test factory functions for creating cloud objects."""

    def test_create_cloud_credentials_factory(self):
        """Test create_cloud_credentials factory function."""
        creds = create_cloud_credentials(
            provider=CloudProvider.AZURE,
            auth_method=CloudAuthMethod.SERVICE_ACCOUNT,
            tenant_id="tenant-123",
            client_id="client-456",
            client_secret="secret-789",  # noqa: S106 # Test credential
            security_level=CloudSecurityLevel.HIGH,
        )

        assert isinstance(creds, CloudCredentials)
        assert creds.provider == CloudProvider.AZURE
        assert creds.auth_method == CloudAuthMethod.SERVICE_ACCOUNT
        assert creds.tenant_id == "tenant-123"
        assert creds.security_level == CloudSecurityLevel.HIGH

    def test_create_cloud_resource_factory(self):
        """Test create_cloud_resource factory function."""
        resource = create_cloud_resource(
            resource_id="factory-test-resource",
            provider=CloudProvider.GOOGLE_CLOUD,
            service_type=CloudServiceType.AI_ML,
            resource_type="vertex-ai-model",
            region="us-central1",
            configuration={
                "model_type": "classification",
                "training_data": "gs://bucket/data",
            },
            tags={"project": "ml-pipeline", "stage": "production"},
        )

        assert isinstance(resource, CloudResource)
        assert resource.resource_id == "factory-test-resource"
        assert resource.provider == CloudProvider.GOOGLE_CLOUD
        assert resource.service_type == CloudServiceType.AI_ML
        assert resource.resource_type == "vertex-ai-model"
        assert resource.region == "us-central1"
        assert resource.configuration["model_type"] == "classification"
        assert resource.tags["project"] == "ml-pipeline"


class TestPropertyBasedValidation:
    """Property-based tests for cloud integration."""

    @given(st.text(min_size=1, max_size=100))
    def test_resource_id_properties(self, resource_id):
        """Property test for resource ID validation."""
        resource = CloudResource(
            resource_id=resource_id,
            provider=CloudProvider.AWS,
            service_type=CloudServiceType.STORAGE,
            resource_type="bucket",
            region="us-east-1",
            configuration={},
        )
        assert resource.resource_id == resource_id
        assert len(resource.resource_id) > 0

    @given(st.text(min_size=1, max_size=50))
    def test_operation_id_properties(self, operation_id):
        """Property test for operation ID validation."""
        resource = CloudResource(
            resource_id="test-resource",
            provider=CloudProvider.AWS,
            service_type=CloudServiceType.COMPUTE,
            resource_type="instance",
            region="us-east-1",
            configuration={},
        )

        operation = CloudOperation(
            operation_id=operation_id,
            operation_type="test_operation",
            provider=CloudProvider.AWS,
            resource=resource,
            parameters={},
        )
        assert operation.operation_id == operation_id
        assert len(operation.operation_id) > 0

    @given(st.integers(min_value=0, max_value=100))
    def test_retry_count_properties(self, retry_count):
        """Property test for retry count validation."""
        resource = CloudResource(
            resource_id="test-resource",
            provider=CloudProvider.AWS,
            service_type=CloudServiceType.STORAGE,
            resource_type="bucket",
            region="us-east-1",
            configuration={},
        )

        operation = CloudOperation(
            operation_id="test-op",
            operation_type="test_operation",
            provider=CloudProvider.AWS,
            resource=resource,
            parameters={},
            retry_count=retry_count,
        )
        assert operation.retry_count == retry_count
        assert operation.retry_count >= 0


class TestIntegrationScenarios:
    """Integration test scenarios for cloud operations."""

    def test_complete_aws_resource_lifecycle(self):
        """Test complete AWS resource lifecycle."""
        # Create credentials
        credentials = create_cloud_credentials(
            provider=CloudProvider.AWS,
            auth_method=CloudAuthMethod.API_KEY,
            access_key="AKIATEST123",
            secret_key="secret123",  # noqa: S106 # Test credential
            region="us-east-1",
            security_level=CloudSecurityLevel.ENTERPRISE,
        )

        # Create resource
        resource = create_cloud_resource(
            resource_id="production-bucket",
            provider=CloudProvider.AWS,
            service_type=CloudServiceType.STORAGE,
            resource_type="s3-bucket",
            region="us-east-1",
            configuration={
                "versioning": True,
                "encryption": "AES256",
                "lifecycle_policy": "enabled",
            },
            tags={"environment": "production", "backup": "daily"},
        )

        # Create operation
        operation = CloudOperation(
            operation_id="create-production-bucket",
            operation_type="create_s3_bucket",
            provider=CloudProvider.AWS,
            resource=resource,
            parameters={
                "bucket_name": "production-bucket",
                "acl": "private",
                "cors_enabled": False,
            },
        )

        # Validate complete workflow
        assert credentials.provider == CloudProvider.AWS
        assert not credentials.is_expired()
        assert (
            resource.get_arn() == "arn:aws:s3:us-east-1:*:s3-bucket/production-bucket"
        )
        assert resource.cost_estimate == 0.023
        assert operation.provider == CloudProvider.AWS
        assert not operation.is_completed()
        assert operation.resource == resource

    def test_multi_cloud_coordination_scenario(self):
        """Test multi-cloud coordination scenario."""
        # AWS Storage
        aws_resource = create_cloud_resource(
            resource_id="aws-data-lake",
            provider=CloudProvider.AWS,
            service_type=CloudServiceType.STORAGE,
            resource_type="s3-bucket",
            region="us-west-2",
            configuration={"replication": "cross-region"},
        )

        # Azure Compute
        azure_resource = create_cloud_resource(
            resource_id="azure-processing-vm",
            provider=CloudProvider.AZURE,
            service_type=CloudServiceType.COMPUTE,
            resource_type="virtual-machine",
            region="westeurope",
            configuration={"size": "Standard_D4s_v3", "os": "Ubuntu 20.04"},
        )

        # GCP Analytics
        gcp_resource = create_cloud_resource(
            resource_id="gcp-bigquery-dataset",
            provider=CloudProvider.GOOGLE_CLOUD,
            service_type=CloudServiceType.ANALYTICS,
            resource_type="bigquery-dataset",
            region="us-central1",
            configuration={"location": "US", "description": "Analytics pipeline"},
        )

        # Validate multi-cloud setup
        providers = {
            aws_resource.provider,
            azure_resource.provider,
            gcp_resource.provider,
        }
        assert len(providers) == 3
        assert CloudProvider.AWS in providers
        assert CloudProvider.AZURE in providers
        assert CloudProvider.GOOGLE_CLOUD in providers

        # Test ARN generation only for AWS
        assert aws_resource.get_arn() is not None
        assert azure_resource.get_arn() is None
        assert gcp_resource.get_arn() is None

    def test_security_validation_scenario(self):
        """Test comprehensive security validation scenario."""
        # Test enterprise-level credentials
        enterprise_creds = create_cloud_credentials(
            provider=CloudProvider.AZURE,
            auth_method=CloudAuthMethod.MANAGED_IDENTITY,
            security_level=CloudSecurityLevel.ENTERPRISE,
        )

        # Test government-level credentials
        gov_creds = create_cloud_credentials(
            provider=CloudProvider.AWS,
            auth_method=CloudAuthMethod.ROLE_BASED,
            security_level=CloudSecurityLevel.GOVERNMENT,
        )

        # Validate security levels
        assert enterprise_creds.security_level == CloudSecurityLevel.ENTERPRISE
        assert gov_creds.security_level == CloudSecurityLevel.GOVERNMENT

        # Test safe representation doesn't leak sensitive data
        safe_enterprise = enterprise_creds.get_safe_representation()
        safe_gov = gov_creds.get_safe_representation()

        # Should not contain actual credential values
        assert "managed_identity" in str(safe_enterprise)
        assert "role_based" in str(safe_gov)
        assert safe_enterprise["security_level"] == "enterprise"
        assert safe_gov["security_level"] == "government"

    def test_cost_optimization_scenario(self):
        """Test cost optimization across different service types."""
        # Create resources with different cost profiles
        storage_resource = create_cloud_resource(
            resource_id="cost-storage",
            provider=CloudProvider.AWS,
            service_type=CloudServiceType.STORAGE,
            resource_type="s3-bucket",
            region="us-east-1",
            configuration={},
        )

        compute_resource = create_cloud_resource(
            resource_id="cost-compute",
            provider=CloudProvider.AWS,
            service_type=CloudServiceType.COMPUTE,
            resource_type="ec2-instance",
            region="us-east-1",
            configuration={},
        )

        serverless_resource = create_cloud_resource(
            resource_id="cost-serverless",
            provider=CloudProvider.AWS,
            service_type=CloudServiceType.SERVERLESS,
            resource_type="lambda-function",
            region="us-east-1",
            configuration={},
        )

        # Validate cost estimates
        total_cost = (
            storage_resource.cost_estimate
            + compute_resource.cost_estimate
            + serverless_resource.cost_estimate
        )

        assert storage_resource.cost_estimate == 0.023  # Storage is cheapest
        assert compute_resource.cost_estimate == 72.0  # Compute is most expensive
        assert serverless_resource.cost_estimate == 0.20  # Serverless is moderate
        assert total_cost == 72.223

        # Test cost optimization recommendations
        cost_ranking = sorted(
            [storage_resource, compute_resource, serverless_resource],
            key=lambda r: r.cost_estimate,
        )

        assert cost_ranking[0] == storage_resource  # Cheapest
        assert cost_ranking[-1] == compute_resource  # Most expensive
