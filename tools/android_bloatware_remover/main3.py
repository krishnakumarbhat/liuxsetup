import subprocess
import re
import datetime
import sys

# --- Configuration ---
OUTPUT_FILENAME = "android_package_details_v6.txt"
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
        elif "No package found" in e.stderr:
            print(f"   ... Package not found or dump failed, skipping.")
            return None
        else:
            print(f"\n🚨 ADB Command Failed: {command}. Error: {e.stderr.strip()}")
        return None
    except FileNotFoundError:
        sys.exit("\n🚨 ERROR: 'adb' command not found. Ensure ADB is installed and configured in your system's PATH.")

def parse_dumpsys_output(output, pkg_name):
    """
    Parses the raw dumpsys output, customized for the iQOO/Vivo format.
    """
    details = {
        "pkg": pkg_name,
        "userId": "N/A",
        "dataDir": "N/A",
        "lastUpdateTime": "N/A",
        "pkgFlags": "N/A",
        "firstInstallTime": "N/A",
        "Authority": "N/A",
        "Path": "N/A",  # Path is not clearly defined in this dump format
        "queriesPackages": "N/A",
        "grantedPermissions": "None Listed" 
    }

    # --- 1. Find the Main Package Block ---
    # We first find the main "Package [com.whatsapp] (...):" block
    package_block_match = re.search(
        r'Package\s*\[{}\]\s*\(.*?\):\s*([\s\S]*?)(?:(?:Packages:)|(?:Queries:))'.format(re.escape(pkg_name)),
        output,
        re.MULTILINE
    )
    
    # If we can't find the main block, we can't parse anything.
    if not package_block_match:
        # Fallback for basic info if main block regex fails
        details["userId"] = re.search(r'userId=(\d+)', output)
        details["userId"] = details["userId"].group(1) if details["userId"] else "N/A"
        return details

    pkg_data = package_block_match.group(1)

    # 2. Basic properties & Time stamps (from within the package block)
    details["userId"] = re.search(r'^\s*userId=(\d+)', pkg_data, re.MULTILINE)
    details["userId"] = details["userId"].group(1) if details["userId"] else "N/A"
    
    details["dataDir"] = re.search(r'^\s*dataDir=([/\w\.]*)', pkg_data, re.MULTILINE)
    details["dataDir"] = details["dataDir"].group(1) if details["dataDir"] else "N/A"
    
    details["lastUpdateTime"] = re.search(r'^\s*lastUpdateTime=(\S+)', pkg_data, re.MULTILINE)
    details["lastUpdateTime"] = details["lastUpdateTime"].group(1) if details["lastUpdateTime"] else "N/A"

    # firstInstallTime is under the 'User 0:' sub-block
    user_0_block = re.search(r'User 0:([\s\S]*?)(?:User \d+:|\Z)', pkg_data)
    if user_0_block:
        user_data = user_0_block.group(1)
        fit_match = re.search(r'^\s*firstInstallTime=([\w\s:-]+)', user_data, re.MULTILINE)
        if fit_match:
            details["firstInstallTime"] = fit_match.group(1).strip()

    # 3. Package Flags
    flags_list_match = re.search(r'^\s*pkgFlags=\[([\s\S]*?)\]', pkg_data, re.MULTILINE)
    if flags_list_match:
        details["pkgFlags"] = flags_list_match.group(1).strip().replace('\n', ' ').replace('  ', '')

    # 4. Provider Authority (NEW REGEX)
    # This is in a totally different section: "ContentProvider Authorities:"
    auth_block_match = re.search(
        r'ContentProvider Authorities:\s*([\s\S]*?)(?:Key Set Manager:|Packages:|\Z)',
        output,
        re.MULTILINE
    )
    if auth_block_match:
        auth_data = auth_block_match.group(1)
        # Find all authorities that belong to our package
        # e.g., Provider{... com.whatsapp/...}
        # This is complex, so we'll just find all authorities listed in the block
        # for this package.
        
        # Simpler approach: Find all [authority.name]: lines
        authorities = re.findall(r'^\s*\[(.+?)\]:', auth_data, re.MULTILINE)
        if authorities:
            details["Authority"] = ", ".join(authorities)

    # 5. Queries Packages (NEW REGEX)
    # Format is: queriesPackages=[...]
    queries_match = re.search(r'^\s*queriesPackages=\[(.*?)\]', pkg_data, re.DOTALL | re.MULTILINE)
    if queries_match:
        queries = queries_match.group(1).strip().replace('\n', ' ').replace('  ', ' ')
        details["queriesPackages"] = queries if queries else "None Listed"

    # 6. GRANTED PERMISSIONS (NEW REGEX)
    # Format is under "User 0:" -> "runtime permissions:"
    if user_0_block:
        user_data = user_0_block.group(1)
        perms_block_match = re.search(
            r'runtime permissions:\s*([\s\S]*?)(?:^\s*disabledComponents:|^\s*enabledComponents:|\Z)',
            user_data,
            re.MULTILINE
        )
        
        if perms_block_match:
            perms_block = perms_block_match.group(1)
            perms_list = []
            for line in perms_block.split('\n'):
                # Look for lines like:
                # android.permission.POST_NOTIFICATIONS: granted=true, ...
                match = re.search(r'^\s*([\w\._]+):\s*granted=true', line.strip())
                if match:
                    perms_list.append(match.group(1))
            
            if perms_list:
                details["grantedPermissions"] = ", ".join(sorted(perms_list))

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