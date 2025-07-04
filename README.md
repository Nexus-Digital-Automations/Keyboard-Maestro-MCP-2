# Keyboard Maestro MCP Server

An advanced macOS automation toolkit that integrates Keyboard Maestro with AI through the Model Context Protocol (MCP). This server provides 51+ production-ready tools for comprehensive macOS automation, featuring enterprise-grade security, functional programming patterns, and property-based testing.

## 🚀 Features

### Core Capabilities
- **Macro Execution**: Execute Keyboard Maestro macros with comprehensive error handling
- **Variable Management**: Manage KM variables across all scopes (global, local, instance, password)
- **Trigger System**: Event-driven macro triggers with functional state management
- **Security Framework**: Multi-level input validation and injection prevention
- **Performance Optimization**: Sub-second response times with intelligent caching

### Advanced Architecture
- **Functional Programming**: Immutable data structures and pure functions
- **Design by Contract**: Pre/post conditions with comprehensive validation
- **Type Safety**: Branded types with complete type system
- **Property-Based Testing**: Hypothesis-driven behavior validation
- **Security Boundaries**: Defense-in-depth with threat modeling

### MCP Integration
- **FastMCP Framework**: Modern Python MCP server implementation
- **10 Production Tools**: Comprehensive macro automation suite
- **Modular Architecture**: Organized tools by functionality (core, advanced, sync, groups)
- **Resource System**: Server status and help documentation
- **Prompt Templates**: Intelligent macro creation assistance

## 📦 Installation

### Prerequisites
- **macOS**: 10.15+ (Catalina or later)
- **Python**: 3.10+ 
- **Keyboard Maestro**: 10.0+ (for full functionality)
- **Claude Desktop**: Latest version

### Setup Instructions

1. **Clone the repository**:
   ```bash
   git clone https://github.com/anthropics/keyboard-maestro-mcp.git
   cd keyboard-maestro-mcp
   ```

2. **Install dependencies**:
   ```bash
   # Create virtual environment
   python -m venv .venv
   source .venv/bin/activate
   
   # Install with uv (recommended)
   pip install uv
   uv sync
   
   # Or install with pip
   pip install -e ".[dev,test,security]"
   ```

3. **Configure Keyboard Maestro**:
   - Enable "Web Server" in Keyboard Maestro preferences
   - Set port to 4490 (default)
   - Grant accessibility permissions if prompted

4. **Test the installation**:
   ```bash
   # Run the server
   uv run python -m src.main
   
   # Or with script entry point
   km-mcp-server --help
   ```

## 🔧 Claude Desktop Configuration

Add the following configuration to your Claude Desktop `claude_desktop_config.json` file:

### Configuration File Location
- **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`
- **Linux**: `~/.config/Claude/claude_desktop_config.json`

### Configuration Content (Recommended)

```json
{
  "mcpServers": {
    "keyboard-maestro": {
      "command": "/Users/YOUR_USERNAME/path/to/keyboard-maestro-mcp/.venv/bin/python",
      "args": [
        "-m",
        "src.main"
      ],
      "cwd": "/Users/YOUR_USERNAME/path/to/keyboard-maestro-mcp",
      "env": {
        "KM_WEB_SERVER_PORT": "4490",
        "KM_CONNECTION_TIMEOUT": "30",
        "LOG_LEVEL": "INFO"
      }
    }
  }
}
```

### Alternative Configuration (using uv for development)

```json
{
  "mcpServers": {
    "keyboard-maestro": {
      "command": "uv",
      "args": [
        "--directory",
        "/Users/YOUR_USERNAME/path/to/keyboard-maestro-mcp",
        "run",
        "python",
        "-m",
        "src.main"
      ],
      "env": {
        "KM_WEB_SERVER_PORT": "4490",
        "KM_CONNECTION_TIMEOUT": "30",
        "LOG_LEVEL": "INFO"
      }
    }
  }
}
```

**Important**: Replace `/Users/YOUR_USERNAME/path/to/keyboard-maestro-mcp` with the actual path to your installation.

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `KM_WEB_SERVER_PORT` | `4490` | Keyboard Maestro web server port |
| `KM_CONNECTION_TIMEOUT` | `30` | Connection timeout in seconds |
| `KM_CONNECTION_METHOD` | `applescript` | Connection method (applescript, url, web) |
| `LOG_LEVEL` | `INFO` | Logging level (DEBUG, INFO, WARNING, ERROR) |
| `SECURITY_LEVEL` | `STANDARD` | Security validation level (MINIMAL, STANDARD, STRICT, PARANOID) |

## 🛠️ Available Tools

### Core MCP Tools

#### 1. `km_execute_macro`
Execute Keyboard Maestro macros with comprehensive error handling.

**Parameters**:
- `identifier` (string): Macro name or UUID
- `trigger_value` (optional string): Parameter to pass to macro
- `method` (string): Execution method (applescript, url, web, remote)
- `timeout` (integer): Maximum execution time (1-300 seconds)

**Example**:
```python
# Execute a macro by name
result = await km_execute_macro(
    identifier="Type Current Date",
    method="applescript",
    timeout=30
)
```

#### 2. `km_list_macros`
List and filter Keyboard Maestro macros with search capabilities.

**Parameters**:
- `group_filter` (optional string): Filter by macro group
- `enabled_only` (boolean): Only return enabled macros
- `sort_by` (string): Sort field (name, last_used, created_date, group)
- `limit` (integer): Maximum results (1-100)

**Example**:
```python
# List all macros in "Utilities" group
macros = await km_list_macros(
    group_filter="Utilities",
    enabled_only=True,
    sort_by="name",
    limit=20
)
```

#### 3. `km_variable_manager`
Comprehensive variable management across all KM scopes.

**Parameters**:
- `operation` (string): Operation type (get, set, delete, list)
- `name` (optional string): Variable name
- `value` (optional string): Variable value (for set operation)
- `scope` (string): Variable scope (global, local, instance, password)
- `instance_id` (optional string): Instance ID for local/instance variables

**Example**:
```python
# Set a global variable
result = await km_variable_manager(
    operation="set",
    name="CurrentProject",
    value="Keyboard Maestro MCP",
    scope="global"
)

# Get a variable value
value = await km_variable_manager(
    operation="get",
    name="CurrentProject",
    scope="global"
)
```

#### 4. `km_search_macros_advanced` ✨ NEW
Advanced macro search with comprehensive filtering and metadata analysis.

**Parameters**:
- `query` (string): Search text for macro names, groups, or content
- `scope` (string): Search scope (name_only, name_and_group, full_content, metadata_only)
- `action_categories` (optional string): Filter by action types
- `complexity_levels` (optional string): Filter by complexity levels
- `min_usage_count` (integer): Minimum execution count filter
- `sort_by` (string): Advanced sorting criteria

#### 5. `km_analyze_macro_metadata` ✨ NEW
Deep analysis of individual macro metadata and patterns.

**Parameters**:
- `macro_id` (string): Macro ID or name to analyze
- `include_relationships` (boolean): Include similarity and relationship analysis

#### 6. `km_list_macro_groups` ✨ NEW
List all macro groups with comprehensive statistics.

**Parameters**:
- `include_macro_count` (boolean): Include count of macros in each group
- `include_enabled_count` (boolean): Include count of enabled macros
- `sort_by` (string): Sort groups by name, macro_count, or enabled_count

#### 7-10. Real-time Synchronization Tools ✨ NEW
- `km_start_realtime_sync`: Start real-time macro library synchronization
- `km_stop_realtime_sync`: Stop real-time synchronization
- `km_sync_status`: Get synchronization status with metrics
- `km_force_sync`: Force immediate synchronization

### Resources

- **`km://server/status`**: Server status and configuration
- **`km://help/tools`**: Comprehensive tool documentation

### Prompts  

- **`create_macro_prompt`**: Generate structured prompts for macro creation

## 🏗️ Architecture Overview

### Project Structure
```
keyboard-maestro-mcp/
├── src/
│   ├── main.py              # Main server entry point (modular)
│   ├── core/                # Core engine and types
│   │   ├── engine.py        # Macro execution engine
│   │   ├── types.py         # Branded types and protocols
│   │   ├── contracts.py     # Design by Contract system
│   │   └── errors.py        # Error hierarchy
│   ├── integration/         # Keyboard Maestro integration
│   │   ├── events.py        # Functional event system
│   │   ├── km_client.py     # KM client with Either monad
│   │   ├── triggers.py      # Trigger management
│   │   ├── security.py      # Security validation
│   │   ├── macro_metadata.py # Enhanced metadata extraction
│   │   ├── smart_filtering.py # Advanced search capabilities
│   │   ├── sync_manager.py  # Real-time synchronization
│   │   └── file_monitor.py  # File system monitoring
│   ├── server/              # Modular server components
│   │   ├── initialization.py # Component initialization
│   │   ├── resources.py     # MCP resources and prompts
│   │   ├── utils.py         # Utility functions
│   │   └── tools/           # Tool modules by functionality
│   │       ├── core_tools.py    # Basic macro operations
│   │       ├── advanced_tools.py # Enhanced search & metadata
│   │       ├── sync_tools.py    # Real-time synchronization
│   │       └── group_tools.py   # Macro group management
│   ├── commands/            # Macro command library
│   │   ├── text.py          # Text manipulation commands
│   │   ├── system.py        # System commands
│   │   ├── application.py   # Application control
│   │   └── validation.py    # Input validation
│   └── tools/               # Legacy tool implementations
├── tests/                   # Comprehensive test suite
│   ├── property_tests/      # Property-based testing
│   ├── integration/         # Integration tests
│   ├── performance/         # Performance benchmarks
│   └── security/            # Security validation tests
├── development/             # Project management
│   ├── TODO.md             # Task tracking
│   └── tasks/              # Detailed task specifications
└── pyproject.toml          # Python project configuration
```

### Key Design Patterns

#### Functional Programming
- **Immutable Data Structures**: All events and state transitions are immutable
- **Pure Functions**: Business logic separated from side effects
- **Function Composition**: Pipeline-based data transformation
- **Either Monad**: Functional error handling without exceptions

#### Design by Contract
- **Preconditions**: Input validation with clear error messages
- **Postconditions**: Output guarantees and state verification
- **Invariants**: System constraints maintained throughout execution
- **Contract Verification**: Automated testing of all contracts

#### Security Framework
- **Defense in Depth**: Multiple validation layers
- **Input Sanitization**: Comprehensive threat detection
- **Permission System**: Fine-grained access control
- **Audit Logging**: Complete operation traceability

## 🧪 Testing

### Test Suite Overview
- **150+ Test Cases**: Comprehensive coverage of all functionality
- **Property-Based Testing**: Hypothesis-driven behavior validation
- **Integration Testing**: End-to-end macro execution
- **Performance Benchmarks**: Sub-10ms startup, sub-5ms validation
- **Security Testing**: Injection prevention and boundary validation

### Running Tests

```bash
# Full test suite with coverage
uv run pytest --cov=src --cov-report=term-missing

# Property-based testing
uv run pytest tests/property_tests/ -v

# Performance benchmarks
uv run pytest tests/performance/ --benchmark-sort=mean

# Security validation
uv run pytest tests/security/ -v

# Integration tests
uv run pytest tests/integration/ -v
```

### Test Categories

- **Unit Tests**: Core engine, types, and command validation
- **Integration Tests**: KM client integration and event system
- **Property Tests**: System behavior across input ranges
- **Security Tests**: Injection prevention and input validation
- **Performance Tests**: Timing requirements and resource usage

## 🔒 Security

### Security Features
- **Multi-Level Validation**: 5 security levels from Minimal to Paranoid
- **Injection Prevention**: Script, command, path traversal, and SQL injection protection
- **Input Sanitization**: Comprehensive threat detection and neutralization
- **Permission Boundaries**: Fine-grained access control system
- **Audit Logging**: Complete operation traceability

### Security Levels

| Level | Description | Use Case |
|-------|-------------|----------|
| MINIMAL | Basic validation only | Development and testing |
| STANDARD | Standard security measures | Most production environments |
| STRICT | Enhanced validation | Security-sensitive environments |
| PARANOID | Maximum security | High-security production systems |

### Threat Categories Protected Against
1. **Script Injection**: JavaScript, AppleScript, shell command injection
2. **Command Injection**: System command execution prevention
3. **Path Traversal**: File system access restriction
4. **SQL Injection**: Database query protection
5. **Macro Abuse**: Malicious macro execution prevention

## 📊 Performance

### Performance Targets
- **Engine Startup**: <10ms
- **Command Validation**: <5ms per command
- **Macro Execution**: <100ms overhead
- **Trigger Response**: <50ms
- **Memory Usage**: <50MB peak

### Optimization Features
- **Connection Pooling**: Reuse KM connections
- **Intelligent Caching**: Cache macro definitions and validation results
- **Async Processing**: Non-blocking operation execution
- **Resource Management**: Automatic cleanup and memory management

## 🔧 Development

### Development Setup

```bash
# Install development dependencies
uv sync --group dev

# Install pre-commit hooks
pre-commit install

# Run code quality checks
uv run ruff check src tests
uv run black --check src tests
uv run mypy src

# Run all tests
uv run pytest
```

### Code Quality Standards
- **Type Safety**: 100% type coverage with mypy
- **Code Formatting**: Black with 88-character line length
- **Linting**: Ruff with security-focused rules
- **Documentation**: Comprehensive docstrings and contracts
- **Test Coverage**: >95% coverage target

### Contributing Guidelines
1. **Follow ADDER+ Architecture**: Implement all advanced techniques
2. **Maintain Type Safety**: Use branded types and contracts
3. **Add Property Tests**: Test behavior across input ranges
4. **Update Documentation**: Keep README and task files current
5. **Security Review**: Validate all security boundaries

## 📚 Documentation

### Additional Resources
- **[CLAUDE.md](./CLAUDE.md)**: Comprehensive development guidelines
- **[TESTING.md](./tests/TESTING.md)**: Test status and execution guide
- **[TODO.md](./development/TODO.md)**: Project task tracking
- **Task Files**: Detailed implementation specifications in `development/tasks/`

### API Documentation
- **MCP Protocol**: Built on FastMCP framework
- **Type System**: Branded types with comprehensive validation
- **Error Handling**: Structured error responses with recovery suggestions
- **Logging**: Structured logging with correlation IDs

## 🤝 Support

### Getting Help
- **GitHub Issues**: Bug reports and feature requests
- **Documentation**: Comprehensive guides and API reference
- **Error Messages**: Detailed error information with recovery suggestions
- **Logging**: Detailed execution traces for debugging

### Known Limitations
- **macOS Only**: Requires macOS and Keyboard Maestro
- **KM Dependencies**: Full functionality requires Keyboard Maestro 10.0+
- **Accessibility**: May require accessibility permissions for some operations
- **Performance**: Complex macros may exceed timing targets

## 📄 License

MIT License - see [LICENSE](./LICENSE) file for details.

## 🙏 Acknowledgments

- **Anthropic**: For Claude and MCP framework
- **Keyboard Maestro**: For the powerful automation platform
- **FastMCP**: For the excellent Python MCP framework
- **Hypothesis**: For property-based testing capabilities

---

**Version**: 1.0.0  
**Last Updated**: 2025-06-30  
**Minimum Requirements**: macOS 10.15+, Python 3.10+, Claude Desktop