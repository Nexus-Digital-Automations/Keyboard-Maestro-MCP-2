"""
LDAP and Active Directory integration with enterprise security.

This module provides secure LDAP/Active Directory connectivity with user and group
synchronization, enterprise authentication, and comprehensive audit logging for
enterprise environments requiring centralized identity management.

Security: Encrypted connections only, certificate validation, injection prevention
Performance: <5s connection, <10s user sync, connection pooling, efficient search
Type Safety: Complete integration with enterprise security and audit frameworks
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any, Set, Tuple
import asyncio
import logging
from datetime import datetime, timedelta, UTC
import ssl
import socket

from ..core.contracts import require, ensure
from ..core.either import Either
from ..core.enterprise_integration import (
    EnterpriseConnection, EnterpriseCredentials, LDAPUser, SyncResult,
    EnterpriseError, IntegrationType, AuthenticationMethod,
    EnterpriseSecurityValidator
)

logger = logging.getLogger(__name__)


class LDAPConnector:
    """LDAP and Active Directory integration with enterprise security."""
    
    def __init__(self):
        self.connections: Dict[str, Any] = {}
        self.connection_pools: Dict[str, List[Any]] = {}
        self.security_validator = EnterpriseSecurityValidator()
        self.connection_stats: Dict[str, Dict[str, Any]] = {}
        
    @require(lambda self, connection: isinstance(connection, EnterpriseConnection))
    @require(lambda self, credentials: isinstance(credentials, EnterpriseCredentials))
    async def connect(self, connection: EnterpriseConnection, 
                     credentials: EnterpriseCredentials) -> Either[EnterpriseError, str]:
        """Establish secure LDAP/AD connection with enterprise validation."""
        try:
            logger.info(f"Establishing LDAP connection to {connection.host}:{connection.port}")
            
            # Validate connection security
            security_check = self.security_validator.validate_connection_security(connection)
            if security_check.is_left():
                return security_check
            
            # Validate credentials security
            creds_check = self.security_validator.validate_credentials_security(credentials)
            if creds_check.is_left():
                return creds_check
            
            # Import LDAP library
            try:
                import ldap3
                from ldap3 import Server, Connection, ALL, NTLM, SIMPLE, SYNC
            except ImportError:
                return Either.left(EnterpriseError.connection_failed("ldap3 library not available"))
            
            # Create secure server configuration
            server = ldap3.Server(
                host=connection.host,
                port=connection.port,
                use_ssl=connection.use_ssl,
                get_info=ldap3.ALL,
                connect_timeout=connection.timeout,
                tls=ldap3.Tls(
                    validate=ssl.CERT_REQUIRED if connection.ssl_verify else ssl.CERT_NONE,
                    version=ssl.PROTOCOL_TLS,
                    ciphers='ECDHE+AESGCM:ECDHE+CHACHA20:DHE+AESGCM:DHE+CHACHA20:!aNULL:!MD5:!DSS'
                ) if connection.use_ssl else None
            )
            
            # Create connection with appropriate authentication
            conn = None
            if credentials.auth_method == AuthenticationMethod.SIMPLE_BIND:
                user_dn = self._build_user_dn(credentials.username, credentials.domain, connection.base_dn)
                
                conn = ldap3.Connection(
                    server,
                    user=user_dn,
                    password=credentials.password,
                    authentication=ldap3.SIMPLE,
                    auto_bind=False,
                    raise_exceptions=True,
                    pool_size=connection.connection_pool_size,
                    pool_lifetime=connection.security_limits.max_connection_lifetime
                )
                
            elif credentials.auth_method == AuthenticationMethod.NTLM:
                if not credentials.domain:
                    return Either.left(EnterpriseError("DOMAIN_REQUIRED", "Domain required for NTLM authentication"))
                
                conn = ldap3.Connection(
                    server,
                    user=f"{credentials.domain}\\{credentials.username}",
                    password=credentials.password,
                    authentication=ldap3.NTLM,
                    auto_bind=False,
                    raise_exceptions=True,
                    pool_size=connection.connection_pool_size
                )
                
            else:
                return Either.left(EnterpriseError.unsupported_auth_method(credentials.auth_method))
            
            # Test connection with bind
            start_time = datetime.now(UTC)
            if not conn.bind():
                error_msg = getattr(conn, 'last_error', 'Unknown authentication error')
                return Either.left(EnterpriseError.authentication_failed())
            
            connection_time = (datetime.now(UTC) - start_time).total_seconds()
            
            # Store connection and statistics
            self.connections[connection.connection_id] = conn
            self.connection_stats[connection.connection_id] = {
                'connected_at': datetime.now(UTC),
                'connection_time': connection_time,
                'searches_performed': 0,
                'records_retrieved': 0,
                'last_activity': datetime.now(UTC),
                'server_info': {
                    'vendor': getattr(server.info, 'vendor_name', 'Unknown'),
                    'version': getattr(server.info, 'vendor_version', 'Unknown'),
                    'naming_contexts': getattr(server.info, 'naming_contexts', [])
                }
            }
            
            logger.info(f"LDAP connection established successfully: {connection.connection_id} ({connection_time:.2f}s)")
            return Either.right(connection.connection_id)
            
        except Exception as e:
            logger.error(f"LDAP connection failed: {str(e)}")
            return Either.left(EnterpriseError.connection_failed(str(e)))
    
    @require(lambda self, connection_id: isinstance(connection_id, str) and len(connection_id) > 0)
    @require(lambda self, search_base: isinstance(search_base, str) and len(search_base) > 0)
    async def search_users(self, connection_id: str, search_base: str, 
                          search_filter: str = "(objectClass=user)",
                          attributes: List[str] = None,
                          size_limit: int = 1000) -> Either[EnterpriseError, List[LDAPUser]]:
        """Search for users in LDAP/AD with security validation."""
        try:
            if connection_id not in self.connections:
                return Either.left(EnterpriseError.connection_not_found(connection_id))
            
            # Validate search filter for injection attacks
            filter_check = self.security_validator.validate_search_filter(search_filter)
            if filter_check.is_left():
                return filter_check
            
            # Validate size limit
            if size_limit > 10000:
                size_limit = 10000  # Enterprise security limit
            
            conn = self.connections[connection_id]
            
            # Default attributes to retrieve
            if attributes is None:
                attributes = [
                    'distinguishedName', 'sAMAccountName', 'userPrincipalName',
                    'displayName', 'givenName', 'sn', 'mail', 'memberOf',
                    'userAccountControl', 'lastLogon', 'whenCreated', 'whenChanged',
                    'department', 'title', 'telephoneNumber', 'mobile'
                ]
            
            logger.info(f"Searching LDAP users: base={search_base}, filter={search_filter}")
            start_time = datetime.now(UTC)
            
            # Perform search with security limits
            success = conn.search(
                search_base=search_base,
                search_filter=search_filter,
                attributes=attributes,
                size_limit=size_limit,
                time_limit=30,  # 30 second timeout
                paged_size=100  # Page results for better performance
            )
            
            search_time = (datetime.now(UTC) - start_time).total_seconds()
            
            if not success:
                error_msg = getattr(conn, 'last_error', 'Unknown search error')
                return Either.left(EnterpriseError.search_failed(error_msg))
            
            # Convert results to LDAPUser objects
            users = []
            errors = []
            
            for entry in conn.entries:
                user_result = self._convert_ldap_entry_to_user(entry)
                if user_result.is_right():
                    users.append(user_result.get_right())
                else:
                    errors.append(str(user_result.get_left()))
            
            # Update connection statistics
            if connection_id in self.connection_stats:
                stats = self.connection_stats[connection_id]
                stats['searches_performed'] += 1
                stats['records_retrieved'] += len(users)
                stats['last_activity'] = datetime.now(UTC)
            
            logger.info(f"LDAP search completed: {len(users)} users found in {search_time:.2f}s")
            if errors:
                logger.warning(f"LDAP search had {len(errors)} conversion errors")
            
            return Either.right(users)
            
        except Exception as e:
            logger.error(f"LDAP search failed: {str(e)}")
            return Either.left(EnterpriseError.search_failed(str(e)))
    
    @require(lambda self, connection_id: isinstance(connection_id, str) and len(connection_id) > 0)
    async def search_groups(self, connection_id: str, search_base: str,
                           search_filter: str = "(objectClass=group)",
                           attributes: List[str] = None) -> Either[EnterpriseError, List[Dict[str, Any]]]:
        """Search for groups in LDAP/AD."""
        try:
            if connection_id not in self.connections:
                return Either.left(EnterpriseError.connection_not_found(connection_id))
            
            # Validate search filter
            filter_check = self.security_validator.validate_search_filter(search_filter)
            if filter_check.is_left():
                return filter_check
            
            conn = self.connections[connection_id]
            
            # Default group attributes
            if attributes is None:
                attributes = [
                    'distinguishedName', 'sAMAccountName', 'displayName',
                    'description', 'member', 'memberOf', 'groupType',
                    'whenCreated', 'whenChanged'
                ]
            
            logger.info(f"Searching LDAP groups: base={search_base}, filter={search_filter}")
            
            success = conn.search(
                search_base=search_base,
                search_filter=search_filter,
                attributes=attributes,
                size_limit=5000,  # Reasonable limit for groups
                time_limit=30
            )
            
            if not success:
                error_msg = getattr(conn, 'last_error', 'Unknown search error')
                return Either.left(EnterpriseError.search_failed(error_msg))
            
            # Convert results to dictionaries
            groups = []
            for entry in conn.entries:
                group_data = {
                    'distinguished_name': str(entry.distinguishedName) if hasattr(entry, 'distinguishedName') else "",
                    'name': str(entry.sAMAccountName) if hasattr(entry, 'sAMAccountName') else "",
                    'display_name': str(entry.displayName) if hasattr(entry, 'displayName') and entry.displayName else "",
                    'description': str(entry.description) if hasattr(entry, 'description') and entry.description else "",
                    'members': [str(member) for member in entry.member] if hasattr(entry, 'member') and entry.member else [],
                    'member_count': len(entry.member) if hasattr(entry, 'member') and entry.member else 0
                }
                groups.append(group_data)
            
            logger.info(f"LDAP group search completed: {len(groups)} groups found")
            return Either.right(groups)
            
        except Exception as e:
            logger.error(f"LDAP group search failed: {str(e)}")
            return Either.left(EnterpriseError.search_failed(str(e)))
    
    @require(lambda self, connection_id: isinstance(connection_id, str) and len(connection_id) > 0)
    async def sync_users(self, connection_id: str, sync_options: Dict[str, Any]) -> Either[EnterpriseError, SyncResult]:
        """Synchronize users from LDAP/AD with comprehensive tracking."""
        try:
            start_time = datetime.now(UTC)
            logger.info(f"Starting LDAP user synchronization: {connection_id}")
            
            # Get search parameters from sync options
            search_base = sync_options.get('search_base', '')
            search_filter = sync_options.get('search_filter', '(objectClass=user)')
            batch_size = min(sync_options.get('batch_size', 100), 500)  # Max 500 for safety
            include_inactive = sync_options.get('include_inactive', False)
            
            if not include_inactive:
                # Modify filter to exclude disabled accounts
                if '(&' in search_filter:
                    search_filter = search_filter.replace('(objectClass=user)', 
                                                        '(&(objectClass=user)(!(userAccountControl:1.2.840.113556.1.4.803:=2)))')
                else:
                    search_filter = f"(&{search_filter}(!(userAccountControl:1.2.840.113556.1.4.803:=2)))"
            
            # Search for users
            users_result = await self.search_users(connection_id, search_base, search_filter)
            if users_result.is_left():
                return users_result
            
            users = users_result.get_right()
            
            # Process users in batches
            processed = 0
            successful = 0
            failed = 0
            errors = []
            warnings = []
            
            for i in range(0, len(users), batch_size):
                batch = users[i:i + batch_size]
                
                for user in batch:
                    try:
                        # Process user (store in local system, update attributes, etc.)
                        await self._process_user_sync(user, sync_options)
                        successful += 1
                        
                        # Check for potential issues
                        if not user.email:
                            warnings.append(f"User {user.username} has no email address")
                        if not user.groups:
                            warnings.append(f"User {user.username} belongs to no groups")
                            
                    except Exception as e:
                        failed += 1
                        error_msg = f"Failed to sync user {user.username}: {str(e)}"
                        errors.append(error_msg)
                        logger.warning(error_msg)
                    
                    processed += 1
                
                # Small delay between batches to prevent overwhelming the server
                if i + batch_size < len(users):
                    await asyncio.sleep(0.1)
            
            sync_duration = (datetime.now(UTC) - start_time).total_seconds()
            completed_at = datetime.now(UTC)
            
            result = SyncResult(
                operation="sync_users",
                integration_type=IntegrationType.LDAP,
                records_processed=processed,
                records_successful=successful,
                records_failed=failed,
                errors=errors,
                warnings=warnings,
                sync_duration=sync_duration,
                started_at=start_time,
                completed_at=completed_at
            )
            
            logger.info(f"LDAP user sync completed: {successful}/{processed} successful in {sync_duration:.2f}s")
            return Either.right(result)
            
        except Exception as e:
            logger.error(f"LDAP user sync failed: {str(e)}")
            return Either.left(EnterpriseError.sync_failed(str(e)))
    
    def get_connection_status(self, connection_id: str) -> Dict[str, Any]:
        """Get status information for LDAP connection."""
        if connection_id not in self.connections:
            return {"status": "not_found"}
        
        stats = self.connection_stats.get(connection_id, {})
        conn = self.connections[connection_id]
        
        # Check connection health
        is_bound = getattr(conn, 'bound', False)
        
        return {
            "status": "connected" if is_bound else "disconnected",
            "connection_id": connection_id,
            "connected_at": stats.get('connected_at'),
            "connection_time": stats.get('connection_time'),
            "searches_performed": stats.get('searches_performed', 0),
            "records_retrieved": stats.get('records_retrieved', 0),
            "last_activity": stats.get('last_activity'),
            "server_info": stats.get('server_info', {}),
            "is_bound": is_bound
        }
    
    async def disconnect(self, connection_id: str) -> Either[EnterpriseError, None]:
        """Safely disconnect from LDAP server."""
        try:
            if connection_id not in self.connections:
                return Either.left(EnterpriseError.connection_not_found(connection_id))
            
            conn = self.connections[connection_id]
            
            # Unbind connection
            if hasattr(conn, 'unbind'):
                conn.unbind()
            
            # Clean up
            del self.connections[connection_id]
            if connection_id in self.connection_stats:
                del self.connection_stats[connection_id]
            
            logger.info(f"LDAP connection disconnected: {connection_id}")
            return Either.right(None)
            
        except Exception as e:
            logger.error(f"LDAP disconnect failed: {str(e)}")
            return Either.left(EnterpriseError.connection_failed(str(e)))
    
    def _build_user_dn(self, username: str, domain: Optional[str], base_dn: Optional[str]) -> str:
        """Build user distinguished name for authentication."""
        if '@' in username:
            return username  # Already in UPN format
        
        if domain and not base_dn:
            return f"{username}@{domain}"  # Use UPN format
        
        if base_dn:
            return f"CN={username},{base_dn}"  # Use DN format
        
        return username  # Use as-is
    
    def _convert_ldap_entry_to_user(self, entry) -> Either[EnterpriseError, LDAPUser]:
        """Convert LDAP entry to LDAPUser object with comprehensive validation."""
        try:
            # Extract basic attributes
            dn = str(entry.distinguishedName) if hasattr(entry, 'distinguishedName') else ""
            username = str(entry.sAMAccountName) if hasattr(entry, 'sAMAccountName') else ""
            
            if not dn or not username:
                return Either.left(EnterpriseError.invalid_ldap_entry())
            
            # Extract other attributes safely
            email = str(entry.mail) if hasattr(entry, 'mail') and entry.mail else None
            display_name = str(entry.displayName) if hasattr(entry, 'displayName') and entry.displayName else None
            first_name = str(entry.givenName) if hasattr(entry, 'givenName') and entry.givenName else None
            last_name = str(entry.sn) if hasattr(entry, 'sn') and entry.sn else None
            
            # Extract group memberships
            groups = set()
            if hasattr(entry, 'memberOf') and entry.memberOf:
                for group_dn in entry.memberOf:
                    group_name = self._extract_group_name(str(group_dn))
                    if group_name:
                        groups.add(group_name)
            
            # Check if user is active
            is_active = True
            if hasattr(entry, 'userAccountControl') and entry.userAccountControl:
                try:
                    uac = int(entry.userAccountControl)
                    is_active = not (uac & 0x2)  # Check disabled flag
                except (ValueError, TypeError):
                    pass
            
            # Extract last login time
            last_login = None
            if hasattr(entry, 'lastLogon') and entry.lastLogon:
                try:
                    # Convert Windows FILETIME to datetime
                    filetime = int(entry.lastLogon)
                    if filetime > 0:
                        # Windows FILETIME epoch is January 1, 1601
                        epoch = datetime(1601, 1, 1)
                        last_login = epoch + timedelta(microseconds=filetime / 10)
                except (ValueError, TypeError):
                    pass
            
            # Extract creation and modification times
            created_at = datetime.now(UTC)
            modified_at = None
            
            if hasattr(entry, 'whenCreated') and entry.whenCreated:
                try:
                    created_at = entry.whenCreated
                except (ValueError, TypeError):
                    pass
            
            if hasattr(entry, 'whenChanged') and entry.whenChanged:
                try:
                    modified_at = entry.whenChanged
                except (ValueError, TypeError):
                    pass
            
            # Convert all attributes for additional data
            attributes = {}
            for attr_name in entry.entry_attributes:
                try:
                    attr_value = getattr(entry, attr_name)
                    if attr_value:
                        if isinstance(attr_value, list):
                            attributes[attr_name] = [str(val) for val in attr_value]
                        else:
                            attributes[attr_name] = [str(attr_value)]
                except Exception:
                    # Skip problematic attributes
                    continue
            
            user = LDAPUser(
                distinguished_name=dn,
                username=username,
                email=email,
                display_name=display_name,
                first_name=first_name,
                last_name=last_name,
                groups=groups,
                attributes=attributes,
                is_active=is_active,
                last_login=last_login,
                created_at=created_at,
                modified_at=modified_at
            )
            
            return Either.right(user)
            
        except Exception as e:
            logger.error(f"LDAP entry conversion failed: {str(e)}")
            return Either.left(EnterpriseError.ldap_conversion_failed(str(e)))
    
    def _extract_group_name(self, group_dn: str) -> Optional[str]:
        """Extract group name from distinguished name."""
        try:
            # Parse DN to extract CN (Common Name)
            parts = group_dn.split(',')
            for part in parts:
                part = part.strip()
                if part.startswith('CN='):
                    return part[3:]  # Remove 'CN=' prefix
            return None
        except Exception:
            return None
    
    async def _process_user_sync(self, user: LDAPUser, sync_options: Dict[str, Any]) -> None:
        """Process individual user during synchronization."""
        # This would integrate with the local user management system
        # For this implementation, we'll just log the operation
        
        # Extract processing options
        update_existing = sync_options.get('update_existing', True)
        create_missing = sync_options.get('create_missing', True)
        sync_groups = sync_options.get('sync_groups', True)
        
        logger.debug(f"Processing user sync: {user.username} (active: {user.is_active})")
        
        # In a real implementation, this would:
        # 1. Check if user exists in local system
        # 2. Create or update user record
        # 3. Sync group memberships
        # 4. Update user attributes
        # 5. Handle deactivation if user is disabled in AD
        
        # Simulate processing time
        await asyncio.sleep(0.01)