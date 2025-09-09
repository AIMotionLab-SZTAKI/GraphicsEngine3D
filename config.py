import yaml
import copy
from pathlib import Path

class Config():
    """
    Configuration class with YAML support, nested dict access, and comment preservation.
    
    Features:
    - Nested dict-like access (config['key'] or config['section']['key'] or config['section.subsection.key'])
    - YAML file loading/saving with comments
    - Config merging with precedence
    """
    
    def __init__(self, folder=None):
        self._config = {}
        self._comments = {}
        self._folder = Path(folder)
        self._load_defaults()

        if folder:
            self._load_from_yaml(folder)
    
    def _load_defaults(self):
        """Load default configuration with comments"""
        self._config = {
            'folder': self._folder,
            'app': {
                'window_size': [1920, 1080],
                'background_color': [0.08, 0.16, 0.18]
            },
            'scene': ['all'],
            'clock': {
                'FPS': 30,
                'time_animation_multiplier': 15,
                'paused': False
            },
        }
        
        # Comments for each configuration entry
        self._comments = {
            'folder': 'Project folder path',

            'app': 'Application settings',
            'app.window_size': 'Window size [width, height]',
            'app.background_color': 'Background color [r, g, b] (0-1 range)',

            'scene': 'List of scene objects to be loaded ["all", "grid", "plans", "terrain"]',

            'clock': 'Clock and timing settings',
            'clock.FPS': 'Target frames per second',
            'clock.time_animation_multiplier': 'Animation time speed multiplier',
            'clock.paused': 'Start with animation paused (true/false)',
        }
    
    def _load_from_yaml(self, folder):
        """Load configuration from YAML file in the specified folder"""
        yaml_path = Path(folder) / 'config.yaml'
        if yaml_path.exists():
            try:
                with open(yaml_path, 'r') as f:
                    yaml_config = yaml.safe_load(f) or {}
                
                # Merge with defaults (YAML takes precedence)
                self._config = self._deep_merge(self._config, yaml_config)

                if 'folder' in self._config: # Ensure folder is a Path object
                    self._config['folder'] = Path(self._config['folder'])

                print(f"Configuration loaded from: {yaml_path}")
            except Exception as e:
                print(f"Error loading config from {yaml_path}: {e}")
                print("Using default configuration")
        else:
            print(f"No config.yaml found in {folder}, using defaults")
    
    def _deep_merge(self, default, override):
        """Deep merge two dictionaries, with override taking precedence"""
        result = copy.deepcopy(default)
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = copy.deepcopy(value)
        
        return result
    
    def _get_nested(self, key):
        """Get nested value using dot notation (e.g., 'camera.position')"""
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                raise KeyError(f"Key '{key}' not found in configuration")
        
        return value
    
    def _set_nested(self, key, value):
        """Set nested value using dot notation"""
        keys = key.split('.')
        config = self._config
        
        # Navigate to the parent of the target key
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        # Set the final value
        config[keys[-1]] = value
    
    def export_yaml(self, filepath, include_comments=True):
        """Export current configuration to YAML file with optional comments"""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Ensure 'scene' is set to ['all'] if it was modified
        if 'scene' in self._config:
            sceneobjectssave = self._config['scene']
            self._config['scene'] = ['all']

        if include_comments:
            self._export_yaml_with_comments(filepath)
        else:
            with open(filepath, 'w') as f:
                yaml.dump(self._config, f, default_flow_style=False, indent=2, sort_keys=False)

        # Restore original 'scene' value
        if 'scene' in self._config:
            self._config['scene'] = sceneobjectssave

    def _export_yaml_with_comments(self, filepath):
        """Export YAML with comments preserved"""
        lines = []
        
        def add_section(data, prefix='', indent=0):
            indent_str = '  ' * indent
            
            for key, value in data.items():
                full_key = f"{prefix}.{key}" if prefix else key
                
                if isinstance(value, dict):
                    # Add comment before section if available
                    if full_key in self._comments:
                        lines.append(f"{indent_str}# {self._comments[full_key]}")
                    lines.append(f"{indent_str}{key}:")
                    add_section(value, full_key, indent + 1)
                    # Add empty line after sections (but not at the very end)
                    if indent == 0:
                        lines.append("")
                else:
                    # Handle different value types properly
                    if isinstance(value, str):
                        yaml_value = f"'{value}'" if ' ' in value or value == '' else value
                    elif isinstance(value, list):
                        if len(value) == 0:
                            yaml_value = "[]"
                        elif all(isinstance(item, (int, float)) for item in value):
                            yaml_value = str(value)  # Keep numeric lists compact
                        else:
                            yaml_value = str(value)  # For mixed or string lists
                    elif isinstance(value, (int, float, bool)):
                        yaml_value = str(value).lower() if isinstance(value, bool) else str(value)
                    elif value is None:
                        yaml_value = "null"
                    else:
                        yaml_value = str(value)
                    
                    line = f"{indent_str}{key}: {yaml_value}"
                    
                    # Add comment after value if available
                    if full_key in self._comments:
                        line += f"  # {self._comments[full_key]}"
                    lines.append(line)
        
        add_section(self._config)
        
        # Remove trailing empty lines
        while lines and lines[-1] == "":
            lines.pop()
        
        with open(filepath, 'w') as f:
            f.write('\n'.join(lines))
            f.write('\n')  # Add final newline

        print(f"Configuration exported to: {filepath}. Scene set to ['all'] for export.")

    # Dict-like interface methods
    def __getitem__(self, key):
        if '.' in key:
            return self._get_nested(key)
        return self._config[key]
    
    def __setitem__(self, key, value):
        if '.' in key:
            self._set_nested(key, value)
        else:
            self._config[key] = value

    def __delitem__(self, key):
        if '.' in key:
            keys = key.split('.')
            config = self._config
            for k in keys[:-1]:
                config = config[k]
            del config[keys[-1]]
        else:
            del self._config[key]
        
    def __contains__(self, key):
        try:
            self[key]
            return True
        except KeyError:
            return False
    
    def get(self, key, default=None):
        """Get configuration value with default fallback"""
        try:
            return self[key]
        except KeyError:
            return default
    
    def items(self):
        return self._config.items()
