#!/usr/bin/env python3
"""
Database Migration Scripts for Cloudburst System
Handles schema updates and data migrations
"""

import asyncio
import asyncpg
import logging
from datetime import datetime
from typing import List, Dict, Optional
import os
from pathlib import Path

class DatabaseMigrator:
    """Handles database schema migrations and updates"""
    
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.migrations_dir = Path('migrations')
        self.migrations_dir.mkdir(exist_ok=True)
        self.setup_logging()
    
    def setup_logging(self):
        """Setup migration logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('migration.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    async def initialize_migration_table(self):
        """Create migration tracking table if it doesn't exist"""
        async with asyncpg.create_pool(self.connection_string) as pool:
            async with pool.acquire() as conn:
                await conn.execute('''
                    CREATE TABLE IF NOT EXISTS cloudburst.migrations (
                        migration_id SERIAL PRIMARY KEY,
                        name VARCHAR(255) UNIQUE NOT NULL,
                        applied_at TIMESTAMPTZ DEFAULT NOW(),
                        checksum VARCHAR(64),
                        execution_time INTERVAL
                    )
                ''')
                self.logger.info("Migration table initialized")
    
    async def get_applied_migrations(self) -> List[str]:
        """Get list of already applied migrations"""
        async with asyncpg.create_pool(self.connection_string) as pool:
            async with pool.acquire() as conn:
                rows = await conn.fetch(
                    'SELECT name FROM cloudburst.migrations ORDER BY applied_at'
                )
                return [row['name'] for row in rows]
    
    async def create_migration(self, name: str, description: str = "") -> str:
        """Create a new migration file"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{name}.sql"
        filepath = self.migrations_dir / filename
        
        template = f"""-- Migration: {name}
-- Description: {description}
-- Created: {datetime.now().isoformat()}

-- UP Migration
-- Write your schema changes here

-- DOWN Migration (optional)
-- Write rollback statements here
"""
        
        with open(filepath, 'w') as f:
            f.write(template)
        
        self.logger.info(f"Created migration file: {filename}")
        return filename
    
    async def run_migration(self, filename: str):
        """Run a specific migration file"""
        filepath = self.migrations_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"Migration file not found: {filename}")
        
        # Read migration file
        with open(filepath, 'r') as f:
            content = f.read()
        
        # Split into UP and DOWN sections
        sections = content.split('-- DOWN Migration')
        up_sql = sections[0].replace('-- UP Migration', '').strip()
        
        if len(sections) > 1:
            down_sql = sections[1].strip()
        else:
            down_sql = None
        
        start_time = datetime.now()
        
        try:
            async with asyncpg.create_pool(self.connection_string) as pool:
                async with pool.acquire() as conn:
                    # Execute UP migration within a transaction
                    async with conn.transaction():
                        # Split by semicolons and execute each statement
                        statements = [stmt.strip() for stmt in up_sql.split(';') if stmt.strip()]
                        
                        for statement in statements:
                            if statement:  # Skip empty statements
                                self.logger.info(f"Executing: {statement[:100]}...")
                                await conn.execute(statement)
                        
                        # Record migration
                        execution_time = datetime.now() - start_time
                        await conn.execute('''
                            INSERT INTO cloudburst.migrations (name, applied_at, execution_time)
                            VALUES ($1, $2, $3)
                        ''', filename, datetime.now(), execution_time)
            
            self.logger.info(f"Successfully applied migration: {filename}")
            
        except Exception as e:
            self.logger.error(f"Migration failed: {filename} - {e}")
            raise
    
    async def run_all_pending_migrations(self):
        """Run all pending migrations in order"""
        await self.initialize_migration_table()
        
        applied = await self.get_applied_migrations()
        all_migrations = sorted([f.name for f in self.migrations_dir.glob('*.sql')])
        
        pending = [m for m in all_migrations if m not in applied]
        
        if not pending:
            self.logger.info("No pending migrations")
            return
        
        self.logger.info(f"Found {len(pending)} pending migrations")
        
        for migration in pending:
            try:
                await self.run_migration(migration)
            except Exception as e:
                self.logger.error(f"Failed to apply migration {migration}: {e}")
                break
    
    async def rollback_migration(self, migration_name: str):
        """Rollback a specific migration"""
        filepath = self.migrations_dir / migration_name
        
        if not filepath.exists():
            raise FileNotFoundError(f"Migration file not found: {migration_name}")
        
        # Read migration file to get DOWN section
        with open(filepath, 'r') as f:
            content = f.read()
        
        sections = content.split('-- DOWN Migration')
        if len(sections) < 2 or not sections[1].strip():
            self.logger.warning(f"No DOWN migration defined for {migration_name}")
            return
        
        down_sql = sections[1].strip()
        
        try:
            async with asyncpg.create_pool(self.connection_string) as pool:
                async with pool.acquire() as conn:
                    async with conn.transaction():
                        statements = [stmt.strip() for stmt in down_sql.split(';') if stmt.strip()]
                        
                        for statement in statements:
                            if statement:
                                self.logger.info(f"Rolling back: {statement[:100]}...")
                                await conn.execute(statement)
                        
                        # Remove migration record
                        await conn.execute(
                            'DELETE FROM cloudburst.migrations WHERE name = $1',
                            migration_name
                        )
            
            self.logger.info(f"Successfully rolled back migration: {migration_name}")
            
        except Exception as e:
            self.logger.error(f"Rollback failed: {migration_name} - {e}")
            raise
    
    async def create_initial_schema(self):
        """Create initial database schema"""
        self.logger.info("Creating initial database schema...")
        
        # Read the main SQL file
        sql_file = Path('software/cloud_backend/database/init_timescale.sql')
        with open(sql_file, 'r') as f:
            schema_sql = f.read()
        
        try:
            async with asyncpg.create_pool(self.connection_string) as pool:
                async with pool.acquire() as conn:
                    # Split and execute statements
                    statements = [stmt.strip() for stmt in schema_sql.split(';') if stmt.strip()]
                    
                    for i, statement in enumerate(statements):
                        if statement and not statement.startswith('--'):
                            self.logger.info(f"Executing schema statement {i+1}/{len(statements)}")
                            try:
                                await conn.execute(statement)
                            except Exception as e:
                                self.logger.warning(f"Statement {i+1} failed: {e}")
                                # Continue with next statement for non-critical errors
            
            self.logger.info("Initial schema created successfully")
            
        except Exception as e:
            self.logger.error(f"Schema creation failed: {e}")
            raise
    
    async def backup_database(self, backup_path: str):
        """Create database backup"""
        self.logger.info(f"Creating database backup: {backup_path}")
        
        # This would use pg_dump in production
        # For this example, we'll create a metadata backup
        backup_data = {
            'backup_timestamp': datetime.now().isoformat(),
            'tables': {}
        }
        
        async with asyncpg.create_pool(self.connection_string) as pool:
            async with pool.acquire() as conn:
                # Get table counts
                tables = await conn.fetch('''
                    SELECT table_name 
                    FROM information_schema.tables 
                    WHERE table_schema = 'cloudburst'
                ''')
                
                for table in tables:
                    table_name = table['table_name']
                    count = await conn.fetchval(f'SELECT COUNT(*) FROM cloudburst.{table_name}')
                    backup_data['tables'][table_name] = {'row_count': count}
        
        # Save backup metadata
        import json
        with open(backup_path, 'w') as f:
            json.dump(backup_data, f, indent=2)
        
        self.logger.info(f"Backup completed: {backup_path}")
    
    async def check_database_health(self) -> Dict:
        """Check database health and statistics"""
        health_info = {
            'status': 'healthy',
            'checks': {},
            'statistics': {}
        }
        
        try:
            async with asyncpg.create_pool(self.connection_string) as pool:
                async with pool.acquire() as conn:
                    # Check connection
                    health_info['checks']['connection'] = 'ok'
                    
                    # Check table sizes
                    table_sizes = await conn.fetch('''
                        SELECT 
                            table_name,
                            pg_size_pretty(pg_total_relation_size('cloudburst.' || table_name)) as size,
                            (SELECT COUNT(*) FROM cloudburst.' || table_name || ') as row_count
                        FROM information_schema.tables 
                        WHERE table_schema = 'cloudburst'
                        ORDER BY pg_total_relation_size('cloudburst.' || table_name) DESC
                    ''')
                    
                    health_info['statistics']['table_sizes'] = [
                        dict(table) for table in table_sizes
                    ]
                    
                    # Check chunk status for hypertables
                    chunks = await conn.fetch('''
                        SELECT 
                            hypertable_name,
                            COUNT(*) as chunk_count,
                            MIN(range_start) as oldest_chunk,
                            MAX(range_end) as newest_chunk
                        FROM timescaledb_information.chunks 
                        WHERE hypertable_schema = 'cloudburst'
                        GROUP BY hypertable_name
                    ''')
                    
                    health_info['statistics']['chunks'] = [
                        dict(chunk) for chunk in chunks
                    ]
                    
                    # Check recent data ingestion
                    recent_data = await conn.fetch('''
                        SELECT 
                            node_id,
                            COUNT(*) as readings_count,
                            MAX(timestamp) as latest_reading
                        FROM cloudburst.sensor_data 
                        WHERE timestamp > NOW() - INTERVAL '1 hour'
                        GROUP BY node_id
                    ''')
                    
                    health_info['statistics']['recent_activity'] = [
                        dict(row) for row in recent_data
                    ]
                    
                    # Check for any long-running queries
                    long_queries = await conn.fetch('''
                        SELECT 
                            query,
                            state,
                            EXTRACT(EPOCH FROM (NOW() - query_start)) as duration_seconds
                        FROM pg_stat_activity 
                        WHERE state = 'active' 
                            AND query_start < NOW() - INTERVAL '30 seconds'
                            AND query != ''
                    ''')
                    
                    if long_queries:
                        health_info['checks']['long_queries'] = 'warning'
                        health_info['long_queries'] = [dict(q) for q in long_queries]
                    else:
                        health_info['checks']['long_queries'] = 'ok'
            
            self.logger.info("Database health check completed")
            
        except Exception as e:
            health_info['status'] = 'unhealthy'
            health_info['error'] = str(e)
            self.logger.error(f"Database health check failed: {e}")
        
        return health_info

# Pre-defined migrations
async def create_standard_migrations(migrator: DatabaseMigrator):
    """Create standard migrations for the Cloudburst system"""
    
    migrations = [
        {
            'name': 'add_weather_radar_integration',
            'description': 'Add tables for weather radar data integration'
        },
        {
            'name': 'add_alert_escalation_rules', 
            'description': 'Add alert escalation and routing rules'
        },
        {
            'name': 'add_performance_optimizations',
            'description': 'Add indexes and performance optimizations'
        },
        {
            'name': 'add_data_retention_policies',
            'description': 'Implement automated data retention policies'
        }
    ]
    
    for migration in migrations:
        await migrator.create_migration(
            migration['name'],
            migration['description']
        )

# Example usage
async def main():
    """Example usage of database migrator"""
    
    # Connection string - replace with your actual database details
    connection_string = "postgresql://user:password@localhost/cloudburst_db"
    
    migrator = DatabaseMigrator(connection_string)
    
    # Create initial schema
    await migrator.create_initial_schema()
    
    # Create standard migrations
    await create_standard_migrations(migrator)
    
    # Run all pending migrations
    await migrator.run_all_pending_migrations()
    
    # Check database health
    health = await migrator.check_database_health()
    print("Database Health:", health)
    
    # Create backup
    await migrator.backup_database('database_backup.json')

if __name__ == '__main__':
    asyncio.run(main())
