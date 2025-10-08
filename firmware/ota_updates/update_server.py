#!/usr/bin/env python3
"""
OTA (Over-the-Air) Update Server for Cloudburst System
Manages firmware updates for edge nodes and gateways
"""

import asyncio
import aiohttp
import aiofiles
import hashlib
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from aiohttp import web

class OTAServer:
    def __init__(self, host: str = '0.0.0.0', port: int = 8080):
        self.host = host
        self.port = port
        self.firmware_dir = Path('firmware_binaries')
        self.firmware_dir.mkdir(exist_ok=True)
        
        # Node registry
        self.nodes: Dict[str, Dict] = {}
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Create web application
        self.app = web.Application()
        self.setup_routes()
    
    def setup_routes(self):
        """Setup HTTP routes for OTA server"""
        self.app.router.add_get('/', self.handle_root)
        self.app.router.add_get('/firmware/check/{node_id}', self.handle_firmware_check)
        self.app.router.add_get('/firmware/download/{filename}', self.handle_firmware_download)
        self.app.router.add_post('/firmware/upload', self.handle_firmware_upload)
        self.app.router.add_post('/node/register', self.handle_node_register)
        self.app.router.add_post('/node/status', self.handle_node_status)
    
    async def handle_root(self, request):
        """Root endpoint - server status"""
        return web.json_response({
            'status': 'running',
            'timestamp': datetime.now().isoformat(),
            'registered_nodes': len(self.nodes),
            'available_firmware': len(list(self.firmware_dir.glob('*.bin')))
        })
    
    async def handle_firmware_check(self, request):
        """Check if firmware update is available for a node"""
        node_id = request.match_info['node_id']
        current_version = request.query.get('version', '1.0.0')
        
        self.logger.info(f"Firmware check for node {node_id}, version {current_version}")
        
        # Get latest firmware info
        latest_firmware = await self.get_latest_firmware()
        
        if not latest_firmware:
            return web.json_response({
                'update_available': False,
                'message': 'No firmware available'
            })
        
        # Check if update needed
        if self.compare_versions(current_version, latest_firmware['version']) < 0:
            return web.json_response({
                'update_available': True,
                'version': latest_firmware['version'],
                'size': latest_firmware['size'],
                'hash': latest_firmware['hash'],
                'url': f"/firmware/download/{latest_firmware['filename']}",
                'description': latest_firmware.get('description', ''),
                'critical': latest_firmware.get('critical', False)
            })
        else:
            return web.json_response({
                'update_available': False,
                'message': 'Node is running latest version'
            })
    
    async def handle_firmware_download(self, request):
        """Download firmware binary"""
        filename = request.match_info['filename']
        file_path = self.firmware_dir / filename
        
        if not file_path.exists():
            return web.json_response({'error': 'Firmware not found'}, status=404)
        
        # Log download
        node_id = request.query.get('node_id', 'unknown')
        self.logger.info(f"Firmware download: {filename} by node {node_id}")
        
        return web.FileResponse(file_path)
    
    async def handle_firmware_upload(self, request):
        """Upload new firmware (admin only)"""
        # Check authentication
        auth_header = request.headers.get('Authorization')
        if not await self.authenticate_admin(auth_header):
            return web.json_response({'error': 'Unauthorized'}, status=401)
        
        data = await request.post()
        firmware_file = data.get('firmware')
        version_info = data.get('version_info')
        
        if not firmware_file or not version_info:
            return web.json_response({'error': 'Missing firmware or version info'}, status=400)
        
        try:
            version_data = json.loads(version_info)
        except json.JSONDecodeError:
            return web.json_response({'error': 'Invalid version info JSON'}, status=400)
        
        # Validate firmware
        firmware_data = firmware_file.file.read()
        file_hash = hashlib.sha256(firmware_data).hexdigest()
        file_size = len(firmware_data)
        
        # Save firmware
        filename = f"firmware_v{version_data['version']}_{datetime.now().strftime('%Y%m%d')}.bin"
        file_path = self.firmware_dir / filename
        
        async with aiofiles.open(file_path, 'wb') as f:
            await f.write(firmware_data)
        
        # Update firmware metadata
        firmware_meta = {
            'filename': filename,
            'version': version_data['version'],
            'size': file_size,
            'hash': file_hash,
            'upload_time': datetime.now().isoformat(),
            'description': version_data.get('description', ''),
            'critical': version_data.get('critical', False),
            'compatible_hardware': version_data.get('compatible_hardware', [])
        }
        
        await self.save_firmware_metadata(firmware_meta)
        
        self.logger.info(f"New firmware uploaded: {filename} (v{version_data['version']})")
        
        return web.json_response({
            'status': 'success',
            'filename': filename,
            'hash': file_hash,
            'size': file_size
        })
    
    async def handle_node_register(self, request):
        """Register a new edge node"""
        try:
            data = await request.json()
            node_id = data['node_id']
            hardware_info = data['hardware_info']
            
            self.nodes[node_id] = {
                'hardware': hardware_info,
                'firmware_version': data.get('firmware_version', '1.0.0'),
                'last_seen': datetime.now().isoformat(),
                'ip_address': request.remote
            }
            
            self.logger.info(f"Node registered: {node_id}")
            
            return web.json_response({'status': 'registered'})
            
        except (KeyError, json.JSONDecodeError) as e:
            return web.json_response({'error': f'Invalid data: {e}'}, status=400)
    
    async def handle_node_status(self, request):
        """Update node status and check for updates"""
        try:
            data = await request.json()
            node_id = data['node_id']
            status = data['status']
            
            if node_id in self.nodes:
                self.nodes[node_id].update({
                    'last_seen': datetime.now().isoformat(),
                    'status': status
                })
            
            # Check for updates
            current_version = data.get('firmware_version', '1.0.0')
            latest_firmware = await self.get_latest_firmware()
            
            update_available = (
                latest_firmware and 
                self.compare_versions(current_version, latest_firmware['version']) < 0
            )
            
            return web.json_response({
                'update_available': update_available,
                'latest_version': latest_firmware['version'] if latest_firmware else current_version
            })
            
        except (KeyError, json.JSONDecodeError) as e:
            return web.json_response({'error': f'Invalid data: {e}'}, status=400)
    
    async def get_latest_firmware(self) -> Optional[Dict]:
        """Get the latest firmware metadata"""
        meta_file = self.firmware_dir / 'firmware_metadata.json'
        
        if not meta_file.exists():
            return None
        
        async with aiofiles.open(meta_file, 'r') as f:
            content = await f.read()
            try:
                metadata = json.loads(content)
                return metadata.get('latest')
            except json.JSONDecodeError:
                return None
    
    async def save_firmware_metadata(self, firmware_meta: Dict):
        """Save firmware metadata"""
        meta_file = self.firmware_dir / 'firmware_metadata.json'
        
        # Load existing metadata
        metadata = {'versions': []}
        if meta_file.exists():
            async with aiofiles.open(meta_file, 'r') as f:
                content = await f.read()
                if content:
                    metadata = json.loads(content)
        
        # Add new version
        metadata['versions'].append(firmware_meta)
        
        # Set as latest if it's newer
        current_latest = metadata.get('latest')
        if (not current_latest or 
            self.compare_versions(current_latest['version'], firmware_meta['version']) < 0):
            metadata['latest'] = firmware_meta
        
        # Save updated metadata
        async with aiofiles.open(meta_file, 'w') as f:
            await f.write(json.dumps(metadata, indent=2))
    
    def compare_versions(self, v1: str, v2: str) -> int:
        """Compare two version strings"""
        v1_parts = [int(x) for x in v1.split('.')]
        v2_parts = [int(x) for x in v2.split('.')]
        
        for i in range(max(len(v1_parts), len(v2_parts))):
            v1_part = v1_parts[i] if i < len(v1_parts) else 0
            v2_part = v2_parts[i] if i < len(v2_parts) else 0
            
            if v1_part < v2_part:
                return -1
            elif v1_part > v2_part:
                return 1
        
        return 0
    
    async def authenticate_admin(self, auth_header: str) -> bool:
        """Authenticate admin user"""
        # In production, this would verify JWT tokens or API keys
        if not auth_header:
            return False
        
        # Simple token-based authentication for demo
        expected_token = "Bearer admin_token_123"
        return auth_header == expected_token
    
    async def start(self):
        """Start the OTA server"""
        runner = web.AppRunner(self.app)
        await runner.setup()
        
        site = web.TCPSite(runner, self.host, self.port)
        await site.start()
        
        self.logger.info(f"OTA Server running on http://{self.host}:{self.port}")
        
        # Keep server running
        await asyncio.Event().wait()
    
    async def stop(self):
        """Stop the OTA server"""
        await self.app.shutdown()
        await self.app.cleanup()

async def main():
    """Main function to start OTA server"""
    server = OTAServer()
    try:
        await server.start()
    except KeyboardInterrupt:
        await server.stop()

if __name__ == '__main__':
    asyncio.run(main())
