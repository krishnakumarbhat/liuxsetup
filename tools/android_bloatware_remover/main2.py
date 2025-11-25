import subprocess
import re
import datetime
import sys

# --- Configuration ---
OUTPUT_FILENAME = "android_package_details_v4.txt"
# ---------------------

def run_adb_command(command):
    """Executes an ADB shell command and returns the output."""
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8'
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        if "device unauthorized" in e.stderr:
            sys.exit("\n🚨 ERROR: Device Unauthorized. Please check your mobile screen and tap 'Allow' for USB debugging.")
        else:
            print(f"\n🚨 ADB Command Failed: {command}. Error: {e.stderr.strip()}")
        return None
    except FileNotFoundError:
        sys.exit("\n🚨 ERROR: 'adb' command not found. Ensure ADB is installed and configured in your system's PATH.")
def parse_dumpsys_output(output, pkg_name):
    """Parses the raw dumpsys output for the required fields, including permissions."""
    details = {
        "pkg": pkg_name,
        "userId": "N/A",
        "dataDir": "N/A",
        "lastUpdateTime": "N/A",
        "pkgFlags": "N/A",
        "firstInstallTime": "N/A",
        "Authority": "N/A",
        "Path": "N/A",
        "queriesPackages": "N/A",
        "grantedPermissions": "None Listed" 
    }
    
    # 1. Basic properties & Time stamps (Your original regex here is usually fine)
    details["userId"] = re.search(r'userId=(\d+)', output)
    details["userId"] = details["userId"].group(1) if details["userId"] else "N/A"
    
    details["dataDir"] = re.search(r'dataDir=([/\w\.]*)', output)
    details["dataDir"] = details["dataDir"].group(1) if details["dataDir"] else "N/A"
    
    details["lastUpdateTime"] = re.search(r'lastUpdateTime=(\S+)', output)
    details["lastUpdateTime"] = details["lastUpdateTime"].group(1) if details["lastUpdateTime"] else "N/A"

    details["firstInstallTime"] = re.search(r'firstInstallTime=(\S+)', output)
    details["firstInstallTime"] = details["firstInstallTime"].group(1) if details["firstInstallTime"] else "N/A"

    # 2. Package Flags (Your original regex here is good)
    flags_hex_match = re.search(r'pkgFlags=(0x[0-9a-fA-F]+)', output)
    flags_list_match = re.search(r'pkgFlags=\[(.*?)\]', output, re.DOTALL)
    
    if flags_hex_match:
        details["pkgFlags"] = flags_hex_match.group(1)
    elif flags_list_match:
        details["pkgFlags"] = flags_list_match.group(1).strip().replace('\n', ' ').replace('  ', ' ')
    
    # 3. Provider Authority and Path (Using a more robust search)
    # <--- MODIFIED: Looks for "Providers:" OR "ContentProviders:" and has more terminators
    providers_match = re.search(r'(?:Providers|ContentProviders):([\s\S]*?)(?:Receivers:|Activities:|Services:|Permissions:)', output)
    
    if providers_match:
        provider_details = providers_match.group(1)
        
        # <--- MODIFIED: Case-insensitive and allows for spaces around '='
        details["Authority"] = re.search(r'[Aa]uthority\s*=\s*([\w\.:,]+)', provider_details)
        details["Authority"] = details["Authority"].group(1) if details["Authority"] else "N/A"

        # <--- MODIFIED: Case-insensitive and allows for spaces around '='
        details["Path"] = re.search(r'[Pp]ath\s*=\s*([/\w\.:]+)', provider_details)
        details["Path"] = details["Path"].group(1) if details["Path"] else "N/A"

    # 4. Queries Packages (Robust multi-line capture)
    # <--- MODIFIED: Allows for different spacing and uses [\s\S]
    queries_match = re.search(r'queriesPackages:\s*\[([\s\S]*?)\]', output)
    if queries_match:
        details["queriesPackages"] = queries_match.group(1).strip().replace('\n', ' ').replace('  ', ' ')

    # 5. GRANTED PERMISSIONS (New Section)
    # <--- MODIFIED: This is the most likely fix. 
    # It looks for "runtimePermissions:" OR "grantedPermissions:"
    # It also has a more robust "stop" condition (blank line, next section, or end of file)
    permissions_match = re.search(r'(?:grantedPermissions|runtimePermissions):\n([\s\S]*?)(?:\n\n|installPermissions:|uid:|\Z)', output)
    
    if permissions_match:
        perms = permissions_match.group(1).strip()
        
        # <--- MODIFIED: A slightly more robust parser for different permission formats
        perms_list = []
        if perms:
            for line in perms.split('\n'):
                line = line.strip()
                if not line:
                    continue
                
                # Format 1: "android.permission.INTERNET: granted=true..."
                if ':' in line:
                    permission_name = line.split(':', 1)[0]
                    # Filter out non-permission lines that might have a colon
                    if 'permission' in permission_name or '.' in permission_name:
                         perms_list.append(permission_name)
                # Format 2: "name:android.permission.INTERNET" (from your original)
                elif line.startswith('name:'):
                    perms_list.append(line[5:].strip())
        
        # Clean up duplicates
        perms_list = sorted(list(set(filter(None, perms_list))))
        details["grantedPermissions"] = ", ".join(perms_list) if perms_list else "None Listed"
    
    return details

def main():
    print(f"--- Starting Android Package Dump on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---")
    
    # 1. Get list of all package names
    print("1. Retrieving all package names from device...")
    packages_output = run_adb_command("adb shell cmd package list packages")
    if packages_output is None:
        return

    package_list = [line.replace("package:", "").strip() for line in packages_output.split('\n') if line.strip()]
    print(f"Found {len(package_list)} packages.")

    # Prepare output file
    with open(OUTPUT_FILENAME, "w", encoding='utf-8') as f:
        # Write Header
        f.write("Package Details Dump\n")
        f.write(f"Generated: {datetime.datetime.now()}\n")
        f.write("="*80 + "\n")
        
        # 2. Loop through each package and dump details
        for i, pkg_name in enumerate(package_list):
            if not pkg_name:
                continue

            print(f"2. Dumping details for package {i+1}/{len(package_list)}: {pkg_name}...")
            
            # Execute dumpsys command
            dumpsys_command = f"adb shell dumpsys package {pkg_name}"
            dumpsys_output = run_adb_command(dumpsys_command)
            
            if dumpsys_output is None:
                continue

            # 3. Parse and write to file
            app_data = parse_dumpsys_output(dumpsys_output, pkg_name)
            
            # Write structured output to file
            f.write(f"\n--- Package: {app_data['pkg']} ---\n")
            f.write(f"User ID: {app_data['userId']}\n")
            f.write(f"Data Directory: {app_data['dataDir']}\n")
            f.write(f"First Install Time: {app_data['firstInstallTime']}\n")
            f.write(f"Last Update Time: {app_data['lastUpdateTime']}\n")
            f.write(f"Package Flags: {app_data['pkgFlags']}\n")
            f.write(f"Provider Authority: {app_data['Authority']}\n")
            f.write(f"Provider Path: {app_data['Path']}\n")
            f.write(f"Queries Packages: {app_data['queriesPackages']}\n")
            f.write(f"GRANTED PERMISSIONS: {app_data['grantedPermissions']}\n")
            f.write("-" * 30 + "\n") # Separator for clarity in the output file

    print(f"\n✅ Success! Complete package details saved to {OUTPUT_FILENAME}")

if __name__ == "__main__":
    main()