import pyautogui
import time
import os
import subprocess
import argparse
from datetime import datetime

# For Windows specific functionality
try:
    import win32gui
except ImportError:
    pass


def focus_application(app_name):
    """
    Focus on the specified application by searching window titles
    """
    print(f"Attempting to focus on window containing: {app_name}")

    # Platform-specific application focus
    if os.name == "posix":  # macOS or Linux
        if "darwin" in os.sys.platform:  # macOS
            try:
                # For macOS, use System Events to focus on process by name
                # This is more reliable than using "tell application" which requires the exact app name
                script = f'''
                tell application "System Events"
                    set appProcesses to processes whose name contains "{app_name}"
                    if (count of appProcesses) > 0 then
                        set frontmost of first item of appProcesses to true
                        return true
                    else
                        return false
                    end if
                end tell
                '''
                result = subprocess.run(
                    ["osascript", "-e", script],
                    capture_output=True,
                    text=True,
                    check=False,
                )

                if "true" in result.stdout:
                    print(f"Successfully focused on {app_name}")
                else:
                    print(f"Could not find window containing '{app_name}'")
                    print("Please focus the window manually before continuing.")
                    # Give user a moment to focus the window manually
                    time.sleep(2)
            except Exception as e:
                print(f"Error when trying to focus window: {e}")
                print("Please focus the window manually before continuing.")
                # Give user a moment to focus the window manually
                time.sleep(2)
        else:  # Linux
            try:
                # Try with partial match
                subprocess.run(["wmctrl", "-a", app_name], check=False)
            except Exception:
                print("Could not focus window automatically. Please focus it manually.")
                time.sleep(2)
    elif os.name == "nt":  # Windows
        try:
            # Try to find windows with the given title
            import win32gui

            def windowEnumerationHandler(hwnd, windows):
                if win32gui.IsWindowVisible(hwnd):
                    window_title = win32gui.GetWindowText(hwnd)
                    if app_name.lower() in window_title.lower():
                        windows.append((hwnd, window_title))

            windows = []
            win32gui.EnumWindows(windowEnumerationHandler, windows)

            if windows:
                print(f"Found matching windows: {[title for _, title in windows]}")
                win32gui.SetForegroundWindow(windows[0][0])
            else:
                # Fallback to AppActivate method
                os.system(
                    f"powershell -c \"(New-Object -ComObject WScript.Shell).AppActivate('{app_name}')\""
                )
        except Exception as e:
            print(f"Could not focus window: {e}")
            print("Please focus the window manually before continuing.")
            time.sleep(2)

    print("Proceeding with screenshot sequence...")

    # Give the application a moment to come into focus
    time.sleep(1)


def take_screenshots(num_slides, delay, output_folder):
    """
    Take screenshots of just the active window, pressing right arrow between each one
    """
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    print(f"Starting screenshot sequence. Will capture {num_slides} slides.")
    print("Press Ctrl+C at any time to stop the process.")

    try:
        # For macOS, we'll use a different approach with screencapture command
        is_macos = os.name == "posix" and "darwin" in os.sys.platform

        # Get current mouse position to restore it later
        original_mouse_pos = pyautogui.position()
        print(f"Current mouse position: {original_mouse_pos.x}, {original_mouse_pos.y}")
        print("Please keep the mouse over the window you want to capture.")
        time.sleep(1)  # Short pause for user to position mouse if needed

        for i in range(num_slides):
            # Generate a filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{output_folder}/slide_{i + 1:03d}_{timestamp}.png"

            print(f"Taking screenshot {i + 1}/{num_slides}...")

            if is_macos:
                # On macOS, use the native screencapture command with -w flag for window selection
                print("Using macOS native window capture...")

                # Start the screencapture process (this will display crosshair/camera cursor)
                proc = subprocess.Popen(["screencapture", "-w", filename])

                # Wait a moment for the crosshair/camera to appear
                time.sleep(0.5)

                # Click at the current mouse position (don't move the mouse)
                pyautogui.click()

                # Wait for the screencapture process to complete
                proc.wait()

                # Check if file was created successfully
                if os.path.exists(filename) and os.path.getsize(filename) > 0:
                    print(f"Saved as {filename}")
                else:
                    print("Screenshot failed, trying fullscreen fallback...")
                    screenshot = pyautogui.screenshot()
                    screenshot.save(filename)
            else:
                # For Windows and other platforms, try to get window bounds
                try:
                    if os.name == "nt":  # Windows
                        import win32gui

                        hwnd = win32gui.GetForegroundWindow()
                        rect = win32gui.GetWindowRect(hwnd)
                        x, y, right, bottom = rect
                        width, height = right - x, bottom - y

                        # Capture just the specified region
                        screenshot = pyautogui.screenshot(region=(x, y, width, height))
                        print(
                            f"Capturing window region: x={x}, y={y}, width={width}, height={height}"
                        )
                    else:
                        # Fall back to full screen for other platforms
                        screenshot = pyautogui.screenshot()
                        print(
                            "Window capture not supported on this platform, capturing full screen."
                        )

                    screenshot.save(filename)
                    print(f"Saved as {filename}")

                except Exception as e:
                    print(f"Error capturing window: {e}")
                    print("Falling back to full screen capture")
                    screenshot = pyautogui.screenshot()
                    screenshot.save(filename)
                    print(f"Saved as {filename}")

            # If this isn't the last slide, press right arrow and wait
            if i < num_slides - 1:
                print(f"Pressing right arrow and waiting {delay} seconds...")
                pyautogui.press("right")
                time.sleep(delay)

        print(f"Screenshot sequence complete! {num_slides} slides captured.")

    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Exiting gracefully.")

    # No need to restore mouse position as we never changed it


def list_open_windows():
    """
    List all the currently open windows to help the user identify the correct window title
    """
    print("\nListing open windows to help you identify the correct one:")

    if os.name == "nt":  # Windows
        try:

            def enum_window_callback(hwnd, window_list):
                if win32gui.IsWindowVisible(hwnd):
                    title = win32gui.GetWindowText(hwnd)
                    if title and len(title) > 0:
                        window_list.append(title)

            window_list = []
            win32gui.EnumWindows(enum_window_callback, window_list)

            for i, title in enumerate(window_list):
                if title.strip():  # Only print non-empty titles
                    print(f"{i + 1}. {title}")

            return window_list
        except Exception as e:
            print(f"Could not list windows: {e}")
            return []

    elif "darwin" in os.sys.platform:  # macOS
        try:
            # This will get all running applications with their visible process names
            script = """
            tell application "System Events"
                set appList to name of every process where background only is false
                return appList
            end tell
            """
            result = subprocess.run(
                ["osascript", "-e", script], capture_output=True, text=True, check=False
            )
            apps = result.stdout.strip().split(", ")

            # Also get window titles for more context
            window_script = """
            tell application "System Events"
                set windowList to {}
                repeat with proc in (every process where background only is false)
                    set procName to name of proc
                    try
                        repeat with w in (every window of proc)
                            try
                                set end of windowList to (procName & ": " & name of w)
                            on error
                                set end of windowList to procName
                            end try
                        end repeat
                    on error
                        set end of windowList to procName
                    end try
                end repeat
                return windowList
            end tell
            """
            window_result = subprocess.run(
                ["osascript", "-e", window_script],
                capture_output=True,
                text=True,
                check=False,
            )

            if window_result.stdout.strip():
                all_windows = window_result.stdout.strip().split(", ")
                for i, window in enumerate(all_windows):
                    if window.strip():
                        print(f"{i + 1}. {window}")
                return all_windows
            else:
                # Fallback to just app names if window titles aren't available
                for i, app in enumerate(apps):
                    if app.strip():
                        print(f"{i + 1}. {app}")
                return apps
        except Exception as e:
            print(f"Could not list applications: {e}")
            return []

    elif os.name == "posix":  # Linux
        try:
            result = subprocess.check_output(["wmctrl", "-l"], text=True)
            windows = result.strip().split("\n")
            window_titles = [
                line.split(None, 3)[3] if len(line.split(None, 3)) > 3 else line
                for line in windows
            ]
            for i, title in enumerate(window_titles):
                print(f"{i + 1}. {title}")
            return window_titles
        except Exception as e:
            print(f"Could not list windows: {e}")
            return []

    return []


def main():
    parser = argparse.ArgumentParser(
        description="Automate screenshots of slides with arrow key navigation."
    )
    parser.add_argument(
        "--app", default="", help="Part of the application window title to focus on"
    )
    parser.add_argument(
        "--slides",
        type=int,
        default=10,
        help="Number of slides to capture (default: 10)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Delay between slides in seconds (default: 1.0)",
    )
    parser.add_argument(
        "--output",
        default="slides_output",
        help="Output folder for screenshots (default: slides_output)",
    )
    parser.add_argument(
        "--list-windows", action="store_true", help="List all open windows and exit"
    )
    parser.add_argument(
        "--start-delay",
        type=int,
        default=5,
        help="Seconds to wait before starting (default: 5)",
    )

    args = parser.parse_args()

    print("=== Slide Screenshot Automation ===")

    # If list-windows flag is set, just list windows and exit
    if args.list_windows:
        list_open_windows()
        print(
            "\nRun the script again with --app parameter including part of the window title."
        )
        return

    # List windows to help user
    open_windows = list_open_windows()

    # Ask for app name if not provided
    app_name = args.app
    if not app_name:
        print("\nPlease enter the number of the window or part of the window title:")
        user_input = input("> ")

        # Check if user entered a number
        try:
            index = int(user_input) - 1
            if 0 <= index < len(open_windows):
                app_name = open_windows[index]
                print(f"Selected: {app_name}")
            else:
                app_name = user_input
        except ValueError:
            app_name = user_input

    print(f"\nTarget window containing: '{app_name}'")
    print("\nIMPORTANT: You have {args.start_delay} seconds to:")
    print(f"1. Switch to the {app_name} window")
    print("2. Position your mouse over the window you want to capture")
    print(f"Screenshots will start automatically in {args.start_delay} seconds...")

    # Countdown timer
    for i in range(args.start_delay, 0, -1):
        print(f"{i}...", end="", flush=True)
        time.sleep(1)
    print("Starting!")

    # Focus the application (though the user should already have done this manually)
    try:
        focus_application(app_name)
    except:
        print(
            "Focus attempt failed, but continuing anyway as you should have focused manually."
        )

    # Take the screenshots
    take_screenshots(args.slides, args.delay, args.output)


if __name__ == "__main__":
    main()
