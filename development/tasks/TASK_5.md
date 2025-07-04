# TASK_5: Real Keyboard Maestro API Integration

**Created By**: Agent_2 | **Priority**: HIGH | **Duration**: 3 hours
**Technique Focus**: Multiple API Integration + Error Handling + Data Transformation
**Size Constraint**: Target <250 lines/module, Max 400 if splitting awkward

## 🚦 Status & Assignment
**Status**: COMPLETED
**Assigned**: Agent_2
**Dependencies**: TASK_1 (Core engine), TASK_2 (KM integration)
**Blocking**: TASK_6 (Enhanced discovery), TASK_7 (State synchronization)

## 📖 Required Reading (Complete before starting)
- [ ] **src/integration/km_client.py**: Current KM client implementation and connection methods
- [ ] **src/server.py**: Mock data implementation in km_list_macros tool
- [ ] **Keyboard Maestro AppleScript Documentation**: getmacros and gethotkeys commands
- [ ] **KM Web API**: HTTP endpoints at localhost:4490 for JSON responses
- [ ] **tests/TESTING.md**: Current integration test status and framework

## 🎯 Implementation Overview
Replace mock macro data with real Keyboard Maestro integration using multiple API methods (AppleScript, Web API, URL schemes) to provide AI clients with complete visibility into user's actual macro library.

<thinking>
Current problem: Server returns only 2 hardcoded mock macros instead of real user macros
Solution approach:
1. AppleScript Integration: Primary method using "tell application Keyboard Maestro" 
2. Web API Integration: HTTP fallback to localhost:4490 for JSON responses
3. Error Handling: Graceful degradation when KM unavailable
4. Data Transformation: Convert KM responses to standardized MCP format
5. Performance: Efficient caching and connection reuse
</thinking>

## ✅ Implementation Subtasks (Sequential completion)

### Phase 1: AppleScript Integration ✅ COMPLETED
- [x] **AppleScript macro listing**: Implement getmacros command via osascript
- [x] **Macro metadata extraction**: Get name, ID, group, enabled status, triggers
- [x] **Group hierarchy mapping**: Extract macro group organization
- [x] **Error handling**: Handle KM not running or accessibility permissions

### Phase 2: Web API Integration ✅ COMPLETED  
- [x] **HTTP client setup**: Configure httpx client for KM web server
- [x] **JSON endpoint integration**: Parse /macros.json and /groups.json responses
- [x] **Fallback mechanism**: Use Web API when AppleScript fails
- [x] **Response validation**: Validate and sanitize JSON responses

### Phase 3: Data Transformation & Integration ✅ COMPLETED
- [x] **Unified response format**: Transform all API responses to consistent format
- [x] **Server integration**: Replace mock data in km_list_macros tool
- [x] **Caching layer**: Implement intelligent caching for performance
- [x] **Testing integration**: Update tests for real KM integration scenarios

## 🔧 Implementation Files & Specifications

### Core Integration Files to Enhance:

#### src/integration/km_client.py - Add Real Macro Listing
```python
async def list_macros_async(
    self, 
    group_filter: Optional[str] = None,
    enabled_only: bool = True
) -> Either[KMError, List[Dict[str, Any]]]:
    """Get real macro list from Keyboard Maestro."""
    
    # Try AppleScript first (most reliable)
    applescript_result = await self._list_macros_applescript(group_filter, enabled_only)
    if applescript_result.is_right():
        return applescript_result
    
    # Fallback to Web API
    web_api_result = await self._list_macros_web_api(group_filter, enabled_only)
    if web_api_result.is_right():
        return web_api_result
    
    # Both methods failed
    return Either.left(KMError.connection_error("Cannot connect to Keyboard Maestro"))

def _list_macros_applescript(
    self, 
    group_filter: Optional[str] = None,
    enabled_only: bool = True
) -> Either[KMError, List[Dict[str, Any]]]:
    """List macros using AppleScript getmacros command."""
    
    script = '''
    tell application "Keyboard Maestro"
        set macroList to {}
        set groupList to every macro group
        
        repeat with currentGroup in groupList
            set groupName to name of currentGroup
            set macroList to macroList & (every macro of currentGroup)
        end repeat
        
        set resultList to {}
        repeat with currentMacro in macroList
            set macroRecord to {¬
                id:(id of currentMacro as string), ¬
                name:(name of currentMacro), ¬
                group:(name of macro group of currentMacro), ¬
                enabled:(enabled of currentMacro), ¬
                triggerCount:(count of triggers of currentMacro), ¬
                actionCount:(count of actions of currentMacro)¬
            }
            set resultList to resultList & {macroRecord}
        end repeat
        
        return resultList
    end tell
    '''
    
    try:
        result = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True,
            text=True,
            timeout=self.config.timeout.total_seconds()
        )
        
        if result.returncode != 0:
            return Either.left(KMError.execution_error(f"AppleScript failed: {result.stderr}"))
        
        # Parse AppleScript record format and convert to dict
        macros = self._parse_applescript_records(result.stdout)
        
        # Apply filters
        if enabled_only:
            macros = [m for m in macros if m.get("enabled", False)]
        if group_filter:
            macros = [m for m in macros if group_filter.lower() in m.get("group", "").lower()]
        
        return Either.right(macros)
        
    except subprocess.TimeoutExpired:
        return Either.left(KMError.timeout_error("AppleScript timeout"))
    except Exception as e:
        return Either.left(KMError.execution_error(f"AppleScript error: {str(e)}"))

async def _list_macros_web_api(
    self, 
    group_filter: Optional[str] = None,
    enabled_only: bool = True
) -> Either[KMError, List[Dict[str, Any]]]:
    """List macros using KM Web API."""
    
    try:
        async with httpx.AsyncClient(timeout=self.config.timeout.total_seconds()) as client:
            # Get macros from web API
            response = await client.get(f"http://{self.config.web_api_host}:{self.config.web_api_port}/macros.json")
            response.raise_for_status()
            
            data = response.json()
            macros = data.get("macros", [])
            
            # Transform to standard format
            standardized_macros = []
            for macro in macros:
                standardized_macro = {
                    "id": macro.get("uid", ""),
                    "name": macro.get("name", ""),
                    "group": macro.get("group", ""),
                    "enabled": macro.get("enabled", True),
                    "triggerCount": len(macro.get("triggers", [])),
                    "actionCount": len(macro.get("actions", [])),
                    "last_used": macro.get("lastUsed"),
                    "created_date": macro.get("created")
                }
                standardized_macros.append(standardized_macro)
            
            # Apply filters
            if enabled_only:
                standardized_macros = [m for m in standardized_macros if m.get("enabled", False)]
            if group_filter:
                standardized_macros = [m for m in standardized_macros if group_filter.lower() in m.get("group", "").lower()]
            
            return Either.right(standardized_macros)
            
    except httpx.TimeoutException:
        return Either.left(KMError.timeout_error("Web API timeout"))
    except httpx.HTTPStatusError as e:
        return Either.left(KMError.connection_error(f"Web API HTTP error: {e.response.status_code}"))
    except Exception as e:
        return Either.left(KMError.execution_error(f"Web API error: {str(e)}"))

def _parse_applescript_records(self, applescript_output: str) -> List[Dict[str, Any]]:
    """Parse AppleScript record format into Python dictionaries."""
    # Implementation to parse AppleScript record syntax
    # This handles the specific format returned by AppleScript records
    import re
    
    records = []
    # Parse AppleScript record format: {id:"uuid", name:"Name", ...}
    record_pattern = r'\{([^}]+)\}'
    
    for match in re.finditer(record_pattern, applescript_output):
        record_str = match.group(1)
        record_dict = {}
        
        # Parse key:value pairs
        pair_pattern = r'(\w+):"([^"]*)"|\w+:(true|false|\d+)'
        for pair_match in re.finditer(pair_pattern, record_str):
            key = pair_match.group(1)
            if pair_match.group(2):  # String value
                value = pair_match.group(2)
            else:  # Boolean or number value
                raw_value = pair_match.group(3)
                if raw_value == "true":
                    value = True
                elif raw_value == "false":
                    value = False
                else:
                    value = int(raw_value)
            record_dict[key] = value
        
        records.append(record_dict)
    
    return records
```

#### src/server.py - Replace Mock Data with Real Integration
```python
@mcp.tool()
async def km_list_macros(
    group_filter: Annotated[Optional[str], Field(
        default=None,
        description="Filter by macro group name or UUID"
    )] = None,
    enabled_only: Annotated[bool, Field(
        default=True,
        description="Only return enabled macros"
    )] = True,
    sort_by: Annotated[str, Field(
        default="name",
        description="Sort field: name, last_used, created_date, or group"
    )] = "name",
    limit: Annotated[int, Field(
        default=20,
        ge=1,
        le=100,
        description="Maximum number of results"
    )] = 20,
    ctx: Context = None
) -> Dict[str, Any]:
    """
    List and filter Keyboard Maestro macros with comprehensive search capabilities.
    
    NOW RETURNS REAL USER MACROS from Keyboard Maestro instead of mock data.
    Supports filtering by group, enabled status, and custom sorting.
    """
    if ctx:
        await ctx.info(f"Listing real macros with filter: {group_filter or 'all groups'}")
    
    try:
        # Get real macro data from KM client
        km_client = get_km_client()
        
        if ctx:
            await ctx.report_progress(25, 100, "Connecting to Keyboard Maestro")
        
        # Query real macros using multiple API methods
        macros_result = await km_client.list_macros_async(
            group_filter=group_filter,
            enabled_only=enabled_only
        )
        
        if macros_result.is_left():
            # Connection failed - provide helpful error message
            error = macros_result.get_left()
            if ctx:
                await ctx.error(f"Cannot connect to Keyboard Maestro: {error}")
            
            return {
                "success": False,
                "error": {
                    "code": "KM_CONNECTION_FAILED",
                    "message": "Cannot connect to Keyboard Maestro",
                    "details": str(error),
                    "recovery_suggestion": "Ensure Keyboard Maestro is running and web server is enabled on port 4490"
                }
            }
        
        if ctx:
            await ctx.report_progress(75, 100, "Processing macro data")
        
        # Get successful result
        all_macros = macros_result.get_right()
        
        # Apply sorting
        sort_fields = {
            "name": lambda m: m.get("name", ""),
            "last_used": lambda m: m.get("last_used", ""),
            "created_date": lambda m: m.get("created_date", ""),
            "group": lambda m: m.get("group", "")
        }
        if sort_by in sort_fields:
            all_macros.sort(key=sort_fields[sort_by])
        
        # Apply limit
        limited_macros = all_macros[:limit]
        
        if ctx:
            await ctx.report_progress(100, 100, "Macro listing complete")
            await ctx.info(f"Found {len(limited_macros)} macros (total: {len(all_macros)})")
        
        return {
            "success": True,
            "data": {
                "macros": limited_macros,
                "total_count": len(all_macros),
                "filtered": group_filter is not None or enabled_only,
                "pagination": {
                    "limit": limit,
                    "returned": len(limited_macros),
                    "has_more": len(all_macros) > limit
                }
            },
            "metadata": {
                "timestamp": datetime.utcnow().isoformat(),
                "server_version": "1.0.0",
                "data_source": "keyboard_maestro_live",
                "query_params": {
                    "group_filter": group_filter,
                    "enabled_only": enabled_only,
                    "sort_by": sort_by
                }
            }
        }
        
    except Exception as e:
        logger.exception("Unexpected error in km_list_macros")
        if ctx:
            await ctx.error(f"Unexpected error: {e}")
        return {
            "success": False,
            "error": {
                "code": "SYSTEM_ERROR",
                "message": "Unexpected system error occurred",
                "details": str(e) if logger.isEnabledFor(logging.DEBUG) else "Contact support",
                "recovery_suggestion": "Check logs and ensure Keyboard Maestro is running"
            }
        }
```

## 🏗️ Modularity Strategy
- **km_client.py enhancements**: Add real macro listing methods (target: +150 lines)
- **AppleScript integration**: Separate module for script parsing (target: 100 lines)
- **Web API client**: HTTP integration module (target: 125 lines)
- **Data transformation**: Unified response formatter (target: 75 lines)
- **Caching layer**: Performance optimization module (target: 100 lines)

## ✅ Success Criteria
- AI client can see user's complete real macro library by name and group
- Multiple API methods provide robust connectivity (AppleScript + Web API + URL schemes)
- Graceful error handling when Keyboard Maestro unavailable
- Performance targets: <2 second response time for macro listing
- Comprehensive filtering and sorting maintained from mock implementation
- All existing server functionality preserved while replacing mock data
- Integration tests validate real KM connectivity scenarios
- Error messages provide clear recovery instructions for connection issues
- Caching optimizes repeated requests without stale data issues
- Security validation prevents injection through macro names/metadata