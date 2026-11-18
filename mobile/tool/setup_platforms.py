"""Add the permissions the app needs after `flutter create` made the platform folders.

    cd mobile
    flutter create . --project-name lift_judge --org com.ze0o --platforms android,ios
    python tool/setup_platforms.py

Android: internet permission + plain-HTTP to your PC (the dev server has no HTTPS).
iOS: camera / microphone / photo library descriptions + local-network HTTP.
Safe to run more than once.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

IOS_KEYS = {
    "NSCameraUsageDescription": "Record your lift so it can be judged.",
    "NSMicrophoneUsageDescription": "Video recording includes audio.",
    "NSPhotoLibraryUsageDescription": "Choose a lift video from your library.",
    "NSLocalNetworkUsageDescription": "Send your video to the Lift Judge server on your computer.",
}


def patch_android(manifest: Path) -> list[str]:
    s = manifest.read_text(encoding="utf-8")
    done = []
    if "android.permission.INTERNET" not in s:
        s = re.sub(r"(<manifest[^>]*>)", r'\1\n    <uses-permission android:name="android.permission.INTERNET"/>', s, count=1)
        done.append("INTERNET permission")
    if "usesCleartextTraffic" not in s:
        s = re.sub(r"<application\b", '<application\n        android:usesCleartextTraffic="true"', s, count=1)
        done.append("plain HTTP to the dev server")
    manifest.write_text(s, encoding="utf-8")
    return done


def patch_ios(plist: Path) -> list[str]:
    s = plist.read_text(encoding="utf-8")
    add, done = [], []
    for k, v in IOS_KEYS.items():
        if f"<key>{k}</key>" not in s:
            add.append(f"\t<key>{k}</key>\n\t<string>{v}</string>")
            done.append(k)
    if "<key>NSAppTransportSecurity</key>" not in s:
        add.append("\t<key>NSAppTransportSecurity</key>\n\t<dict>\n\t\t<key>NSAllowsLocalNetworking</key>\n"
                   "\t\t<true/>\n\t\t<key>NSAllowsArbitraryLoads</key>\n\t\t<true/>\n\t</dict>")
        done.append("local-network HTTP")
    if add:
        i = s.rindex("</dict>")
        s = s[:i] + "\n".join(add) + "\n" + s[i:]
    plist.write_text(s, encoding="utf-8")
    return done


def main(root: Path = ROOT) -> int:
    manifest = root / "android" / "app" / "src" / "main" / "AndroidManifest.xml"
    plist = root / "ios" / "Runner" / "Info.plist"
    if not manifest.exists() and not plist.exists():
        print("No android/ or ios/ folder yet. Run first:\n"
              "  flutter create . --project-name lift_judge --org com.ze0o --platforms android,ios")
        return 1
    if manifest.exists():
        print("Android:", ", ".join(patch_android(manifest)) or "already set up")
    if plist.exists():
        print("iOS:", ", ".join(patch_ios(plist)) or "already set up")
    return 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT))
