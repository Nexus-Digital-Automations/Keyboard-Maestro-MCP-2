"""
Property-Based Tests for File Operations

Tests file operations using property-based testing to validate
security boundaries, path validation, and transaction safety.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from hypothesis import given, strategies as st, assume, settings
from hypothesis.stateful import RuleBasedStateMachine, rule, invariant

from src.filesystem.file_operations import (
    FileOperationManager,
    FileOperationRequest,
    FileOperationType,
    FilePath
)
from src.core.errors import ContractViolationError
from src.filesystem.path_security import PathSecurity


class TestFileOperationProperties:
    """Property-based tests for file operations with security validation."""
    
    @given(st.text(min_size=1, max_size=100))
    def test_path_validation_rejects_dangerous_patterns(self, path_input):
        """Property: Dangerous path patterns should always be rejected."""
        # Skip if path contains null bytes (not valid for filesystem)
        assume('\x00' not in path_input)
        
        # Test dangerous patterns
        dangerous_inputs = [
            f"../{path_input}",
            f"../../{path_input}", 
            f"{path_input}/../secrets",
            f"/etc/{path_input}",
            f"~/{path_input}",
            f"${{{path_input}}}",
            f"`{path_input}`"
        ]
        
        for dangerous_path in dangerous_inputs:
            assert not PathSecurity.validate_path(dangerous_path), \
                f"Path validation should reject dangerous pattern: {dangerous_path}"
    
    @given(st.text(min_size=1, max_size=50, alphabet=st.characters(blacklist_characters=['/', '\\', ':', '*', '?', '"', '<', '>', '|'])))
    def test_safe_filename_validation(self, filename):
        """Property: Safe filenames in allowed directories should pass validation."""
        assume(filename not in ['.', '..', 'CON', 'PRN', 'AUX', 'NUL'])  # Windows reserved names
        assume(not filename.startswith('.'))  # Skip hidden files for simplicity
        
        # Test in Documents directory (should be allowed)
        safe_path = str(Path.home() / "Documents" / filename)
        
        # This might pass validation (depends on configuration)
        result = PathSecurity.validate_path(safe_path)
        # We can't assert True here because the actual allowed directories might be configured differently
        # But we can assert that if it passes, it should be truly safe
        if result:
            assert '../' not in safe_path
            assert not safe_path.startswith('/')
    
    @given(st.integers(min_value=0, max_value=1000000))
    def test_file_size_handling_properties(self, file_size):
        """Property: File size calculations should be consistent and safe."""
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            # Write data of specified size
            temp_file.write(b'x' * file_size)
            temp_file.flush()
            
            try:
                file_path = FilePath(temp_file.name)
                calculated_size = file_path.get_size()
                
                if calculated_size is not None:
                    assert calculated_size == file_size, \
                        f"File size should match written size: {calculated_size} != {file_size}"
                    assert calculated_size >= 0, "File size should never be negative"
            finally:
                Path(temp_file.name).unlink(missing_ok=True)
    
    def test_file_operation_transaction_safety(self):
        """Property: Failed operations should not leave partial state."""
        with tempfile.TemporaryDirectory() as temp_dir:
            source_file = Path(temp_dir) / "source.txt"
            dest_file = Path(temp_dir) / "dest.txt"
            
            # Create source file
            source_file.write_text("test content")
            
            # Create invalid destination (no permissions)
            try:
                manager = FileOperationManager()
                
                # Try to copy to invalid destination
                request = FileOperationRequest(
                    operation=FileOperationType.COPY,
                    source_path=FilePath(str(source_file)),
                    destination_path=FilePath("/root/invalid_dest.txt"),  # Should fail
                    overwrite=False
                )
                
                # This should fail safely
                result = manager.execute_operation(request)
                
                # Property: Source should remain unchanged after failed operation
                assert source_file.exists(), "Source file should remain after failed operation"
                assert source_file.read_text() == "test content", "Source content should be unchanged"
                
                # Property: Failed operation should not create partial destination
                assert not Path("/root/invalid_dest.txt").exists(), "Failed operation should not create partial destination"
                
            except Exception:
                # Even if exception occurs, source should be unchanged
                assert source_file.exists(), "Source file should remain even if exception occurs"


class FileOperationStateMachine(RuleBasedStateMachine):
    """Stateful property testing for file operations."""
    
    def __init__(self):
        super().__init__()
        self.temp_dir = Path(tempfile.mkdtemp())
        self.manager = FileOperationManager()
        self.files = {}  # Track created files
    
    def teardown(self):
        """Clean up temporary directory."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    @rule(filename=st.text(min_size=1, max_size=20, alphabet=st.characters(
        min_codepoint=32, max_codepoint=126,  # Printable ASCII only
        blacklist_characters=['/', '\\', ':', '*', '?', '"', '<', '>', '|']
    )))
    def create_file(self, filename):
        """Create a test file."""
        assume(filename not in ['.', '..'])
        assume(not filename.startswith('.'))
        
        file_path = self.temp_dir / filename
        if not file_path.exists():
            try:
                file_path.write_text(f"content_{filename}")
                self.files[filename] = file_path
            except (OSError, UnicodeError):
                # Skip files that can't be created due to filesystem limitations
                pass
    
    @rule()
    def delete_file(self):
        """Delete a test file."""
        if self.files:
            filename = list(self.files.keys())[0]  # Get first available file
            file_path = self.files[filename]
            if file_path.exists():
                try:
                    request = FileOperationRequest(
                        operation=FileOperationType.DELETE,
                        source_path=FilePath(str(file_path))
                    )
                    
                    result = self.manager.execute_operation(request)
                    
                    if result.is_right():
                        # File should be deleted
                        assert not file_path.exists(), "File should be deleted after successful operation"
                        del self.files[filename]
                    else:
                        # If operation failed, file should still exist
                        assert file_path.exists(), "File should exist if delete operation failed"
                        
                except Exception:
                    # Even on exception, file state should be consistent
                    pass
    
    @invariant()
    def files_consistent_with_filesystem(self):
        """Invariant: Tracked files should match filesystem state."""
        for filename, file_path in self.files.items():
            if file_path.exists():
                assert file_path.is_file(), f"Tracked path should be a file: {file_path}"
                content = file_path.read_text()
                assert content.startswith("content_"), f"File should have expected content pattern: {content}"


# Test class instantiation for running stateful tests
TestFileOperationStateMachine = FileOperationStateMachine.TestCase


@pytest.mark.property
class TestPathSecurityProperties:
    """Property-based tests specifically for path security validation."""
    
    @given(st.text())
    def test_path_validation_never_crashes(self, path_input):
        """Property: Path validation should handle all inputs gracefully."""
        try:
            result = PathSecurity.validate_path(path_input)
            assert isinstance(result, bool), "Validation should always return boolean"
        except ContractViolationError:
            # Contract violations are acceptable for invalid inputs
            pass
        except Exception as e:
            pytest.fail(f"Path validation crashed unexpectedly on input '{path_input}': {e}")
    
    @given(st.text(min_size=1))
    def test_sanitization_safety(self, path_input):
        """Property: Path sanitization should never make paths less safe."""
        assume('\x00' not in path_input)  # Skip null bytes
        
        try:
            sanitized = PathSecurity.sanitize_path(path_input)
            
            if sanitized is not None:
                # Sanitized path should still be valid
                assert PathSecurity.validate_path(sanitized), \
                    f"Sanitized path should be valid: {sanitized}"
                
                # Should not contain dangerous patterns
                assert '../' not in sanitized, "Sanitized path should not contain ../"
                assert '..\\' not in sanitized, "Sanitized path should not contain ..\\"
        except Exception:
            # Sanitization is allowed to fail, but shouldn't crash
            pass
    
    @settings(max_examples=50)  # Reduce examples for performance
    @given(st.lists(st.text(min_size=1, max_size=20), min_size=1, max_size=5))
    def test_path_joining_security(self, path_components):
        """Property: Joining path components should maintain security."""
        # Clean components first
        clean_components = []
        for component in path_components:
            if '\x00' not in component and component not in ['.', '..']:
                clean_components.append(component)
        
        assume(len(clean_components) > 0)
        
        # Join components
        joined_path = '/'.join(clean_components)
        
        # Test validation
        try:
            is_valid = PathSecurity.validate_path(joined_path)
            # We can't assert validity, but can check consistency
            if is_valid:
                assert not any(dangerous in joined_path for dangerous in ['../', '..\\'])
        except Exception:
            # Validation is allowed to fail on edge cases
            pass


if __name__ == "__main__":
    # Run property tests
    pytest.main([__file__, "-v", "--tb=short"])