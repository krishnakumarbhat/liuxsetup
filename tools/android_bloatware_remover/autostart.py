import subprocess
import sys

# List of package names provided by the user
raw_packages_string = """
com.adobe.psmobile (Photoshop Express)
com.adobe.scan.android (Adobe Scan)
com.aimchess (AimChess)
com.azure.authenticator (Microsoft Authenticator)
com.bigbasket.mobileapp (BigBasket)
com.camerasideas.instashot (InShot)
com.chess (Chess.com)
ch.protonmail.android (Proton Mail)
com.deniscerri.ytdl (YTDL)
com.discord (Discord)
com.flipkart.android (Flipkart)
com.google.android.apps.authenticator2 (Google Authenticator)
com.google.android.apps.bard (Gemini)
com.google.android.apps.docs (Google Docs)
com.google.android.apps.docs.editors.docs (Docs)
com.google.android.apps.docs.editors.sheets (Sheets)
com.google.android.apps.docs.editors.slides (Slides)
com.google.android.apps.labs.language.tailwind (Tailwind)
com.google.android.apps.tasks (Google Tasks)
com.grofers.customerapp (Blinkit)
com.idea.videocompress (Video Compress)
in.pricehistory.app (Price History)
in.stablemoney.app (Stable Money)
io.metamask (MetaMask)
com.jio.myjio (MyJio)
com.kadambasociety (Kadamba Society)
com.medium.reader (Medium)
com.microsoft.office.officelens (Microsoft Lens)
com.microsoft.office.onenote (OneNote)
org.mozilla.fenix (Firefox)
music.ytstream.youtify (Youtify)
net.skyscanner.android.main (Skyscanner)
com.nobroker.app (NoBroker)
notion.id (Notion)
org.cris.aikyam (Aikyam)
org.kde.kdeconnect_tp (KDE Connect)
com.popoko.weiqi (Go/Weiqi)
com.rapido.passenger (Rapido)
com.reddit.frontpage (Reddit)
sixpack.sixpackabs.absworkout (Abs Workout)
com.snapwork.hdfc (HDFC Bank)
com.soartech.shopswiftly.shopswiftly (ShopSwiftly)
com.Splitwise.SplitwiseMobile (Splitwise)
com.techmash.playo (Playo)
com.ttxapps.drivesync (DriveSync)
com.twitter.android (X/Twitter)
com.vgfit.yoga (Yoga)
com.whatsapp (WhatsApp)
"""

def parse_package_names(raw_string):
    """Parses the raw string into a clean list of package names."""
    package_names = []
    for line in raw_string.strip().split('\n'):
        line = line.strip()
        if not line:  # Skip empty lines
            continue
        
        # Take only the first part of the line (the package name)
        package_name = line.split()[0]
        package_names.append(package_name)
    return package_names

def set_background_permission(package_name):
    """Runs the ADB command to restrict background activity."""
    command = [
        "adb",
        "shell",
        "appops",
        "set",
        package_name,
        "RUN_ANY_IN_BACKGROUND",  # <-- This is the updated command
        "ignore"
    ]
    
    print(f"Attempting to set 'RUN_ANY_IN_BACKGROUND ignore' for: {package_name}")
    
    try:
        # Execute the command
        result = subprocess.run(
            command, 
            check=True,  # Raise an error if the command fails
            capture_output=True,  # Capture stdout/stderr
            text=True,  # Decode as text
            timeout=10  # Set a timeout of 10 seconds
        )
        
        # appops doesn't print to stdout on success, but check stderr
        if result.stderr:
            print(f"  -> Warning/Info: {result.stderr.strip()}")
        else:
            print(f"  -> Success.")

    except FileNotFoundError:
        print("\nError: 'adb' command not found.", file=sys.stderr)
        print("Please ensure ADB is installed and in your system's PATH.", file=sys.stderr)
        sys.exit(1)  # Exit the script
        
    except subprocess.CalledProcessError as e:
        # This catches errors from the adb command itself
        print(f"  -> Failed for {package_name}.")
        print(f"  -> Error: {e.stderr.strip()}", file=sys.stderr)
        
    except subprocess.TimeoutExpired:
        print(f"  -> Timeout expired for {package_name}. Is the device responsive?", file=sys.stderr)
        
    print("-" * 20)

# --- Main execution ---
if __name__ == "__main__":
    packages = parse_package_names(raw_packages_string)
    
    print(f"Found {len(packages)} packages to process.")
    print("Ensure your Android device is connected and USB debugging is enabled.")
    print("---")
    
    for pkg in packages:
        set_background_permission(pkg)
        
    print("\nScript finished.")