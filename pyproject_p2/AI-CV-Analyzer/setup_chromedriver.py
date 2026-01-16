#!/usr/bin/env python3
"""
Setup script to download and install the correct chromedriver version
This can be run during deployment to ensure compatibility
"""

import os
import sys
import subprocess
import platform
import tempfile
import zipfile
import shutil
from urllib.request import urlretrieve

def get_chrome_version():
    """Get the installed Chrome version on Windows"""
    # Windows-specific Chrome detection
    try:
        # Try to find Chrome in Program Files
        chrome_paths = [
            r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
            r"C:\Users\%USERNAME%\AppData\Local\Google\Chrome\Application\chrome.exe"
        ]
        
        for path in chrome_paths:
            expanded_path = os.path.expandvars(path)
            if os.path.exists(expanded_path):
                # Use WMIC to get version info on Windows
                try:
                    output = subprocess.check_output(
                        ['wmic', 'datafile', 'where', f'name="{expanded_path.replace("\\", "\\\\")}"', 'get', 'Version', '/value'],
                        stderr=subprocess.STDOUT
                    )
                    version_str = output.decode('utf-8').strip()
                    if "Version=" in version_str:
                        version = version_str.split('=')[1].split('.')[0]
                        return version
                except:
                    # Try alternative method
                    try:
                        output = subprocess.check_output([expanded_path, '--version'], stderr=subprocess.STDOUT)
                        version = output.decode('utf-8').strip().split()[-1].split('.')[0]
                        return version
                    except:
                        pass
    except Exception as e:
        print(f"Could not determine Chrome version on Windows: {str(e)}")
    
    # If we can't detect automatically, ask the user
    print("Could not automatically detect Chrome version.")
    try:
        user_version = input("Please enter your Chrome version (e.g., 120): ")
        if user_version.isdigit():
            return user_version
    except:
        pass
    
    # Default to latest if all else fails
    print("Using default Chrome version 120.")
    return "120"

def download_chromedriver(version):
    """Download the appropriate chromedriver for Windows"""
    platform_name = "win32"
    
    # Map Chrome version to compatible chromedriver version
    # Source: https://chromedriver.chromium.org/downloads
    version_map = {
        "120": "120.0.6099.109",
        "119": "119.0.6045.105",
        "118": "118.0.5993.70",
        "117": "117.0.5938.149",
        "116": "116.0.5845.96",
        "115": "115.0.5790.170",
        "114": "114.0.5735.90",
        "113": "113.0.5672.63",
        "112": "112.0.5615.49",
        "111": "111.0.5563.64",
        "110": "110.0.5481.77",
        "109": "109.0.5414.74",
    }
    
    if version in version_map:
        driver_version = version_map[version]
    else:
        # Default to latest if version not found
        print(f"Chrome version {version} not found in mapping, using latest chromedriver")
        driver_version = "120.0.6099.109"  # Default to latest
    
    # Create URL for download
    download_url = f"https://chromedriver.storage.googleapis.com/{driver_version}/chromedriver_{platform_name}.zip"
    
    # Create temp directory for download
    temp_dir = tempfile.mkdtemp()
    zip_path = os.path.join(temp_dir, "chromedriver.zip")
    
    try:
        print(f"Downloading chromedriver version {driver_version} from {download_url}")
        urlretrieve(download_url, zip_path)
        
        # Extract the zip file
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        
        # Find the chromedriver executable
        chromedriver_path = os.path.join(temp_dir, "chromedriver.exe")
        
        # Create directory if it doesn't exist
        install_dir = os.path.join(os.environ.get('LOCALAPPDATA', os.path.expanduser("~")), "ChromeDriver")
            
        os.makedirs(install_dir, exist_ok=True)
        
        # Copy to install directory
        install_path = os.path.join(install_dir, os.path.basename(chromedriver_path))
        shutil.copy2(chromedriver_path, install_path)
        
        print(f"Installed chromedriver to {install_path}")
        return install_path
    
    except Exception as e:
        print(f"Error downloading/installing chromedriver: {str(e)}")
        return None
    finally:
        # Clean up temp directory
        shutil.rmtree(temp_dir, ignore_errors=True)

def main():
    """Main function to detect Chrome version and install matching chromedriver"""
    print("Setting up chromedriver...")
    
    # Get Chrome version
    chrome_version = get_chrome_version()
    if not chrome_version:
        print("Could not detect Chrome version. Please install Chrome or Chromium.")
        sys.exit(1)
    
    print(f"Detected Chrome version: {chrome_version}")
    
    # Download and install matching chromedriver
    chromedriver_path = download_chromedriver(chrome_version)
    if not chromedriver_path:
        print("Failed to install chromedriver.")
        sys.exit(1)
    
    print(f"Successfully installed chromedriver for Chrome version {chrome_version}")
    print(f"Chromedriver path: {chromedriver_path}")
    
    # Add to PATH environment variable for this session
    os.environ["PATH"] = os.path.dirname(chromedriver_path) + os.pathsep + os.environ["PATH"]
    
    # Test chromedriver
    try:
        version_output = subprocess.check_output([chromedriver_path, "--version"], stderr=subprocess.STDOUT)
        print(f"Chromedriver version: {version_output.decode('utf-8').strip()}")
        print("Chromedriver setup completed successfully!")
    except Exception as e:
        print(f"Error testing chromedriver: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 