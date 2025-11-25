import subprocess
import sys
import os
import re

# --- Configuration ---
OUTPUT_FILE = "adb_app_details_metadata_final.txt"

# Command to list all installed 3rd-party package names
LIST_PACKAGES_CMD = ["adb", "shell", "pm", "list", "packages", "-3"]

# Regex pattern to identify the start of key package data sections
# This ensures we grab all the raw metadata you wanted (pkg=, dataDir=, queriesPackages=, etc.)
METADATA_START_PATTERN = re.compile(r"Package \[\w+\] \([\w]+\)|userId=\d+")

def run_adb_command(command_parts):
    """Runs a single ADB command and returns the output (stdout)."""
    try:
        result = subprocess.run(
            command_parts,
            capture_output=True,
            text=True,
            check=True,
            encoding="utf-8"
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Command failed for package list: {e.stderr.strip()}", file=sys.stderr)
        return None
    except FileNotFoundError:
        print("\n[CRITICAL ERROR] ADB not found. Ensure it's in your system's PATH.", file=sys.stderr)
        sys.exit(1)

def run_full_scan_with_metadata():
    print(f"--- Starting ADB scan for Full Details (Metadata, Dates, Permissions) ---")
    print("This may take a few minutes. Do not disconnect your device.")

    package_list_output = run_adb_command(LIST_PACKAGES_CMD)
    if package_list_output is None:
        return

    packages = [line.strip().replace("package:", "") for line in package_list_output.splitlines() if line.strip()]

    if not packages:
        print("\n[INFO] No third-party packages found on device.")
        return

    print(f"\nFound {len(packages)} third-party apps to scan...")

    # 2. Iterate through packages and dump details
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for i, pkg in enumerate(packages):
            sys.stdout.write(f"\r[PROGRESS] Scanning {i + 1}/{len(packages)}: {pkg}")
            sys.stdout.flush()

            DUMPSYS_CMD = ["adb", "shell", "dumpsys", "package", pkg]
            raw_output = run_adb_command(DUMPSYS_CMD)

            if raw_output is not None:
                f.write(f"\n=======================================================\n")
                f.write(f"--- PACKAGE: {pkg} ---\n")
                f.write(f"=======================================================\n")

                in_granted_permissions_block = False
                in_metadata_block = False

                for line in raw_output.splitlines():
                    stripped_line = line.strip()
                    
                    # --- 1. Filter for Dates and Metadata ---
                    
                    # Start capturing the main metadata block from the top of the dumpsys output
                    if METADATA_START_PATTERN.search(stripped_line):
                        in_metadata_block = True
                    
                    if in_metadata_block:
                        # Capture key metadata lines
                        if stripped_line.startswith(('pkg=', 'dataDir=', 'queriesPackages=', 'pkgFlags=', 'userId=')):
                             f.write(" " * 4 + stripped_line + "\n")
                        
                        # Capture dates (and skip the invalid 1970 date)
                        if ("firstInstallTime=" in stripped_line or "lastUpdateTime=" in stripped_line) and "1970-01-01" not in stripped_line:
                            f.write(stripped_line + "\n")
                            
                        # Stop metadata block just before the permission definitions start
                        if stripped_line.startswith("requestedPermissions:") or stripped_line.startswith("permissions:"):
                            in_metadata_block = False
                            
                    # --- 2. Filter for Granted Permissions Block ---

                    # Start of Granted Permissions Block
                    if stripped_line == "grantedPermissions:":
                        f.write("\n" + stripped_line + "\n")
                        in_granted_permissions_block = True
                        continue

                    if in_granted_permissions_block:
                        # Stop capturing when the next major section starts
                        if stripped_line.startswith("requestedPermissions:") or stripped_line.startswith("runtimePermissions:"):
                            in_granted_permissions_block = False
                            continue

                        # Capture the permission string itself
                        if stripped_line.startswith("android.permission.") or stripped_line.startswith("com."):
                            f.write(stripped_line + "\n")

    # 3. Completion Message
    print(f"\n\n--- Scan Complete ---")
    print(f"✅ Successfully scanned {len(packages)} apps.")
    print(f"The detailed data has been saved to: {os.path.abspath(OUTPUT_FILE)}")

if __name__ == "__main__":
    run_full_scan_with_metadata()