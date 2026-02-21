"""
Tests for cross-platform camera support in STREAM.

This test module validates that camera capture is correctly configured to work
across ALL operating systems (macOS, Windows, Linux) and ALL STREAM modes
(desktop/PyWebView, server/browser, future mobile app).

WHAT WE'RE TESTING:
  1. macOS .app bundle has NSCameraUsageDescription in Info.plist
     (without this, macOS silently denies camera access)
  2. Desktop main.py sets QTWEBENGINE_CHROMIUM_FLAGS for Qt renderer
     (without this, Qt WebEngine blocks getUserMedia)
  3. The three-tier camera detection strategy is documented correctly
  4. PyWebView configuration includes camera-compatible settings

WHY THESE TESTS MATTER:
  Camera access is a platform-specific capability that can silently break:
    - macOS: Missing Info.plist key → getUserMedia returns NotAllowedError
    - Qt renderer: Missing env var → getUserMedia blocked without dialog
    - PyWebView: Missing media support → getUserMedia undefined
  These tests act as guardrails to prevent regressions.

WHAT WE'RE NOT TESTING:
  - getUserMedia itself (requires a real camera + browser environment)
  - The React CameraModal component (requires a DOM/browser test runner)
  - <input capture="environment"> behavior (requires a mobile device)
  These are tested manually during development. The detection logic and
  configuration are tested here because they're the most fragile parts.

Run with:
    cd /Users/nassar/Documents/CODES/STREAM
    .venv/bin/python -m pytest tests/test_camera_cross_platform.py -v
"""

import os

import pytest

# =============================================================================
# FIXTURES: Shared test data and file paths
# =============================================================================


@pytest.fixture
def project_root():
    """Root directory of the STREAM project."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture
def spec_file_path(project_root):
    """Path to the PyInstaller spec file."""
    return os.path.join(project_root, "stream.spec")


@pytest.fixture
def main_py_path(project_root):
    """Path to the desktop main.py entry point."""
    return os.path.join(project_root, "stream", "desktop", "main.py")


@pytest.fixture
def image_upload_path(project_root):
    """Path to the ImageUpload.tsx component."""
    return os.path.join(
        project_root,
        "frontends",
        "react",
        "src",
        "components",
        "input",
        "ImageUpload.tsx",
    )


@pytest.fixture
def spec_file_content(spec_file_path):
    """Read the spec file content."""
    with open(spec_file_path) as f:
        return f.read()


@pytest.fixture
def main_py_content(main_py_path):
    """Read the desktop main.py content."""
    with open(main_py_path) as f:
        return f.read()


@pytest.fixture
def image_upload_content(image_upload_path):
    """Read the ImageUpload.tsx content."""
    with open(image_upload_path) as f:
        return f.read()


# =============================================================================
# TESTS: macOS Info.plist camera permission (stream.spec)
# =============================================================================
# On macOS, every app that accesses the camera MUST declare a usage description
# in its Info.plist. This string is shown to the user in the system permission
# dialog. Without it, the OS silently denies camera access or crashes the app.
#
# Apple's documentation:
# https://developer.apple.com/documentation/bundleresources/information_property_list/nscamerausagedescription


class TestMacOSCameraPermission:
    """Tests that the macOS .app bundle is configured for camera access."""

    def test_spec_file_exists(self, spec_file_path):
        """The PyInstaller spec file must exist."""
        assert os.path.exists(spec_file_path), (
            f"stream.spec not found at {spec_file_path}. "
            "This file is required to build the macOS .app bundle."
        )

    def test_nscamerausagedescription_present(self, spec_file_content):
        """Info.plist must include NSCameraUsageDescription.

        Without this key, macOS will deny camera access when the CameraModal
        calls navigator.mediaDevices.getUserMedia(). The user would see
        "Camera access was denied" with no system prompt to grant permission.
        """
        assert "NSCameraUsageDescription" in spec_file_content, (
            "stream.spec is missing NSCameraUsageDescription in info_plist. "
            "The macOS .app bundle needs this key for camera access. "
            "Add it to the info_plist dict in the BUNDLE section."
        )

    def test_camera_description_is_meaningful(self, spec_file_content):
        """The camera usage description should explain WHY the camera is needed.

        Apple's App Review guidelines require that privacy usage descriptions
        clearly explain the purpose. Generic descriptions like "needs camera"
        get rejected during App Store review (relevant for future distribution).
        """
        # Check that the description mentions AI/analysis (our actual purpose)
        lower_content = spec_file_content.lower()
        assert "multimodal" in lower_content or "ai" in lower_content, (
            "NSCameraUsageDescription should mention the AI/multimodal purpose. "
            "Apple requires meaningful privacy descriptions."
        )

    def test_info_plist_has_required_keys(self, spec_file_content):
        """Info.plist should have all essential keys for a well-configured app."""
        required_keys = [
            "CFBundleName",
            "CFBundleDisplayName",
            "CFBundleShortVersionString",
            "CFBundleVersion",
            "NSHighResolutionCapable",
            "CFBundleIconFile",
            "NSCameraUsageDescription",
        ]
        for key in required_keys:
            assert key in spec_file_content, (
                f"stream.spec is missing {key} in info_plist. "
                f"This key is required for a properly configured macOS .app."
            )


# =============================================================================
# TESTS: Qt WebEngine camera support (main.py)
# =============================================================================
# When PyWebView uses the Qt/WebEngine renderer (common on Linux, optional on
# macOS/Windows), getUserMedia is blocked by default because Qt WebEngine
# doesn't auto-grant media permissions. The QTWEBENGINE_CHROMIUM_FLAGS env
# var with --use-fake-ui-for-media-stream bypasses this.
#
# Qt documentation on Chromium flags:
# https://doc.qt.io/qt-6/qtwebengine-debugging.html


class TestQtWebEngineCameraSupport:
    """Tests that desktop main.py configures Qt for camera access."""

    def test_main_py_exists(self, main_py_path):
        """The desktop main.py entry point must exist."""
        assert os.path.exists(main_py_path), (
            f"main.py not found at {main_py_path}. " "This file is the desktop app's entry point."
        )

    def test_qt_chromium_flags_set(self, main_py_content):
        """main.py must set QTWEBENGINE_CHROMIUM_FLAGS for camera support.

        Without this, Qt WebEngine's getUserMedia call will be blocked
        (no permission dialog appears, the promise just rejects). The
        CameraModal would show "Could not access camera" on Linux systems
        using the Qt renderer.
        """
        assert "QTWEBENGINE_CHROMIUM_FLAGS" in main_py_content, (
            "main.py must set QTWEBENGINE_CHROMIUM_FLAGS environment variable. "
            "This enables getUserMedia in Qt WebEngine renderer. "
            "Add: os.environ.setdefault('QTWEBENGINE_CHROMIUM_FLAGS', "
            "'--use-fake-ui-for-media-stream')"
        )

    def test_fake_ui_for_media_stream_flag(self, main_py_content):
        """The Qt flag must include --use-fake-ui-for-media-stream.

        This specific Chromium flag tells the browser engine to auto-grant
        media stream permissions without showing a UI dialog. Other flags
        exist (like --use-fake-device-for-media-stream which uses synthetic
        video) but we need the REAL camera, just without the dialog.
        """
        assert "--use-fake-ui-for-media-stream" in main_py_content, (
            "QTWEBENGINE_CHROMIUM_FLAGS must include --use-fake-ui-for-media-stream. "
            "This flag auto-grants camera permission in Qt WebEngine."
        )

    def test_uses_setdefault_not_direct_assignment(self, main_py_content):
        """Should use os.environ.setdefault() to avoid overwriting user flags.

        If the user has already set QTWEBENGINE_CHROMIUM_FLAGS for other
        purposes (e.g., remote debugging), we should NOT overwrite their
        value. setdefault() only sets the value if the key doesn't exist.
        """
        assert "setdefault" in main_py_content, (
            "Use os.environ.setdefault() instead of os.environ['...'] = '...' "
            "to avoid overwriting user-configured Qt flags."
        )

    def test_qt_flags_set_before_webview_start(self, main_py_content):
        """Qt flags must be set BEFORE webview.start() is called.

        Qt reads QTWEBENGINE_CHROMIUM_FLAGS at engine initialization time.
        If set after webview.start(), the flag has no effect. The env var
        must appear in the code BEFORE the start() call.

        We search line-by-line and skip comment lines (starting with #)
        to compare the ACTUAL code positions, not mentions in comments.
        """
        flags_line = None
        start_line = None

        for i, line in enumerate(main_py_content.splitlines()):
            stripped = line.strip()
            # Skip comments and blank lines
            if stripped.startswith("#") or not stripped:
                continue
            if "QTWEBENGINE_CHROMIUM_FLAGS" in stripped and flags_line is None:
                flags_line = i
            if "webview.start()" in stripped and start_line is None:
                start_line = i

        assert (
            flags_line is not None
        ), "QTWEBENGINE_CHROMIUM_FLAGS not found in main.py (non-comment code)"
        assert start_line is not None, "webview.start() not found in main.py (non-comment code)"
        assert flags_line < start_line, (
            f"QTWEBENGINE_CHROMIUM_FLAGS (line {flags_line}) must appear BEFORE "
            f"webview.start() (line {start_line}) in the code. "
            "Qt reads environment variables at engine initialization time."
        )


# =============================================================================
# TESTS: Frontend camera detection strategy (ImageUpload.tsx)
# =============================================================================
# The frontend uses a three-tier feature detection strategy instead of
# user-agent sniffing. These tests validate that the component implements
# the correct pattern and doesn't regress to UA sniffing.


class TestFrontendCameraStrategy:
    """Tests that ImageUpload.tsx uses feature detection, not UA sniffing."""

    def test_image_upload_exists(self, image_upload_path):
        """ImageUpload.tsx must exist."""
        assert os.path.exists(
            image_upload_path
        ), f"ImageUpload.tsx not found at {image_upload_path}"

    def test_no_user_agent_sniffing_for_camera(self, image_upload_content):
        """Camera detection must NOT use user-agent string matching.

        User-agent sniffing is fragile and breaks with:
          - iPads reporting desktop Safari UAs (since iPadOS 13)
          - Foldable phones switching between mobile/desktop UAs
          - Capacitor/Tauri apps with custom UAs
          - Future device categories

        We use maxTouchPoints instead (hardware capability check).
        """
        # Check that the camera click handler doesn't use navigator.userAgent
        # for mobile detection. We allow navigator.userAgent elsewhere in the
        # file (e.g., for logging), but not in the camera strategy logic.
        camera_section = _extract_between(
            image_upload_content,
            "handleCameraClick",
            "handleCameraCapture",
        )
        assert camera_section is not None, "Could not find handleCameraClick in ImageUpload.tsx"
        assert "navigator.userAgent" not in camera_section, (
            "handleCameraClick must not use navigator.userAgent for device detection. "
            "Use navigator.maxTouchPoints or the isTouchDevice() function instead."
        )

    def test_uses_max_touch_points(self, image_upload_content):
        """Camera detection should use maxTouchPoints for device detection.

        navigator.maxTouchPoints is a hardware capability check that returns:
          - 0 on desktops without touchscreens
          - 1+ on phones, tablets, foldables, touch-enabled laptops
        Unlike user-agent sniffing, it works reliably across all devices.
        """
        assert "maxTouchPoints" in image_upload_content, (
            "ImageUpload.tsx should use navigator.maxTouchPoints to detect "
            "touch devices for camera strategy selection."
        )

    def test_uses_get_user_media_check(self, image_upload_content):
        """Camera detection should check for getUserMedia availability.

        Some environments (PyWebView on macOS/Linux) don't expose the
        getUserMedia API. The code must check for its existence before
        attempting to open the webcam modal.
        """
        assert (
            "getUserMedia" in image_upload_content
        ), "ImageUpload.tsx must check for getUserMedia availability."

    def test_has_three_camera_strategies(self, image_upload_content):
        """Must support three camera strategies: native-camera, webcam-modal, file-picker.

        The three-tier strategy ensures the camera button works everywhere:
          1. native-camera: Touch devices use the native camera app
          2. webcam-modal: Desktop browsers use getUserMedia webcam preview
          3. file-picker: Fallback for environments without camera API
        """
        assert (
            "native-camera" in image_upload_content
        ), "Missing 'native-camera' strategy for touch devices."
        assert (
            "webcam-modal" in image_upload_content
        ), "Missing 'webcam-modal' strategy for desktop browsers."
        assert "file-picker" in image_upload_content, "Missing 'file-picker' fallback strategy."

    def test_exports_detection_functions(self, image_upload_content):
        """Detection utility functions should be exported for testing.

        The isTouchDevice(), hasGetUserMedia(), and getCameraStrategy()
        functions should be exported so they can be unit-tested independently
        and potentially reused by other components.
        """
        assert (
            "export function isTouchDevice" in image_upload_content
        ), "isTouchDevice() should be exported from ImageUpload.tsx"
        assert (
            "export function hasGetUserMedia" in image_upload_content
        ), "hasGetUserMedia() should be exported from ImageUpload.tsx"
        assert (
            "export function getCameraStrategy" in image_upload_content
        ), "getCameraStrategy() should be exported from ImageUpload.tsx"

    def test_input_capture_environment(self, image_upload_content):
        """Must have an <input capture="environment"> for native camera.

        The `capture` attribute on a file input tells mobile browsers to
        open the camera instead of the file picker:
          - "environment" = rear camera (for documents, whiteboards)
          - "user" = front camera (for selfies)
        We use "environment" because STREAM's primary use case is capturing
        documents, diagrams, and physical objects for AI analysis.
        """
        assert (
            'capture="environment"' in image_upload_content
        ), 'Must have <input capture="environment"> for native camera on mobile.'

    def test_accepts_all_image_types(self, image_upload_content):
        """File inputs must accept all image types (image/*)."""
        assert (
            'accept="image/*"' in image_upload_content
        ), 'File inputs must use accept="image/*" to support all image formats.'


# =============================================================================
# TESTS: CameraModal error handling
# =============================================================================
# The CameraModal must handle every getUserMedia error type gracefully.
# Different errors need different messages because the user action to fix
# them is different (grant permission vs. connect camera vs. close Zoom).


class TestCameraModalErrorHandling:
    """Tests that CameraModal handles all getUserMedia error types."""

    def test_handles_not_allowed_error(self, image_upload_content):
        """Must handle NotAllowedError (user denied camera permission).

        This is the most common error. The user clicked "Block" in the
        browser's permission dialog, or the OS denied access.
        The message should tell them HOW to fix it.
        """
        assert (
            "NotAllowedError" in image_upload_content
        ), "CameraModal must handle NotAllowedError (camera permission denied)."

    def test_handles_not_found_error(self, image_upload_content):
        """Must handle NotFoundError (no camera hardware).

        Happens on desktops without a webcam, or when the camera is
        disconnected. The message should suggest using Upload instead.
        """
        assert (
            "NotFoundError" in image_upload_content
        ), "CameraModal must handle NotFoundError (no camera detected)."

    def test_handles_not_readable_error(self, image_upload_content):
        """Must handle NotReadableError (camera busy / in use).

        Happens when another app (Zoom, Teams, FaceTime) has exclusive
        access to the camera. The message should name common culprits.
        """
        assert (
            "NotReadableError" in image_upload_content
        ), "CameraModal must handle NotReadableError (camera in use by another app)."

    def test_handles_overconstrained_error(self, image_upload_content):
        """Must handle OverconstrainedError (unsupported camera constraints).

        Happens when we request facingMode: 'environment' on a desktop
        webcam that only supports 'user' mode. The code should retry
        without the facingMode constraint.
        """
        assert "OverconstrainedError" in image_upload_content, (
            "CameraModal must handle OverconstrainedError and retry "
            "without facingMode constraint."
        )

    def test_guards_against_missing_getusermedia(self, image_upload_content):
        """Must check if getUserMedia exists before calling it.

        In PyWebView on some platforms, navigator.mediaDevices may be
        undefined. Calling .getUserMedia() on undefined would crash
        with "Cannot read properties of undefined". The modal must
        check first and show a helpful error message.
        """
        assert "navigator.mediaDevices?.getUserMedia" in image_upload_content, (
            "CameraModal must use optional chaining (?.getUserMedia) to avoid "
            "crashing when navigator.mediaDevices is undefined."
        )

    def test_suggests_upload_as_fallback(self, image_upload_content):
        """Error messages should suggest the Upload button as a fallback.

        When the camera doesn't work (for any reason), the user should
        know they can still add images via the Upload button.
        """
        upload_mentions = image_upload_content.count("Upload button")
        assert upload_mentions >= 2, (
            "Camera error messages should mention the 'Upload button' as "
            f"a fallback. Found {upload_mentions} mentions, expected >= 2."
        )

    def test_suggests_browser_for_desktop_app_users(self, image_upload_content):
        """Error messages should suggest opening STREAM in browser.

        When getUserMedia fails in PyWebView (desktop app), the most
        reliable fix is to open STREAM in a regular browser instead.
        """
        browser_mentions = image_upload_content.lower().count("browser")
        assert browser_mentions >= 2, (
            "Camera error messages should suggest opening STREAM in the "
            f"browser as a fallback. Found {browser_mentions} mentions."
        )


# =============================================================================
# TESTS: Camera stream cleanup
# =============================================================================
# Failing to stop camera tracks causes the camera indicator to stay on,
# prevents other apps from using the camera, and wastes battery on laptops.


class TestCameraStreamCleanup:
    """Tests that camera streams are properly cleaned up."""

    def test_stops_tracks_on_close(self, image_upload_content):
        """Must stop all media tracks when the modal closes.

        Without stopping tracks:
          - The camera LED stays on (confusing and privacy-concerning)
          - Other apps can't access the camera
          - Battery drain on laptops
        """
        # Count occurrences of track stopping pattern
        stop_count = image_upload_content.count(".getTracks().forEach")
        assert stop_count >= 3, (
            f"Found {stop_count} track-stopping patterns, expected >= 3. "
            "Camera tracks must be stopped in: (1) cleanup effect, "
            "(2) handleClose, (3) cancelled check."
        )

    def test_cleanup_on_unmount(self, image_upload_content):
        """Must clean up camera in the useEffect cleanup function.

        React's useEffect cleanup runs when the component unmounts.
        If the parent component conditionally removes the CameraModal
        (e.g., navigating away), the cleanup function must stop the
        camera stream to release the hardware.
        """
        # The cleanup function should set cancelled = true and stop tracks
        assert "cancelled = true" in image_upload_content, (
            "useEffect cleanup must set cancelled = true to prevent " "state updates after unmount."
        )


# =============================================================================
# TESTS: Cross-platform documentation
# =============================================================================
# The camera strategy and platform-specific behavior must be documented
# so future developers understand WHY specific patterns are used.


class TestCameraDocumentation:
    """Tests that camera cross-platform logic is properly documented."""

    def test_documents_three_tier_strategy(self, image_upload_content):
        """Must document the three-tier camera detection strategy."""
        assert (
            "Tier 1" in image_upload_content or "tier" in image_upload_content.lower()
        ), "ImageUpload.tsx should document the three-tier camera strategy."

    def test_documents_why_not_user_agent(self, image_upload_content):
        """Must explain why user-agent sniffing was rejected."""
        ua_doc = (
            "userAgent" in image_upload_content.lower()
            or "user-agent" in image_upload_content.lower()
            or "user agent" in image_upload_content.lower()
        )
        assert ua_doc, "Should document why user-agent sniffing is not used for detection."

    def test_documents_secure_context_requirement(self, image_upload_content):
        """Must document that getUserMedia requires a secure context."""
        assert (
            "secure context" in image_upload_content.lower()
            or "localhost" in image_upload_content.lower()
        ), "Should document that getUserMedia requires HTTPS or localhost."

    def test_documents_pywebview_compatibility(self, image_upload_content):
        """Must document PyWebView compatibility considerations."""
        assert (
            "PyWebView" in image_upload_content or "pywebview" in image_upload_content.lower()
        ), "Should document PyWebView compatibility for desktop mode."


# =============================================================================
# TESTS: Desktop mode environment setup
# =============================================================================
# These tests verify the runtime behavior of the desktop mode setup
# by checking that the environment variables and configuration are
# correctly applied.


class TestDesktopModeSetup:
    """Tests for desktop mode camera configuration at runtime."""

    def test_qt_flags_env_var_setdefault_behavior(self):
        """os.environ.setdefault should not overwrite existing values.

        If a user has already set QTWEBENGINE_CHROMIUM_FLAGS for debugging
        or other purposes, our setdefault call should NOT overwrite it.
        This test verifies Python's setdefault semantics.
        """
        key = "_STREAM_TEST_QT_FLAGS"
        try:
            # Case 1: Key doesn't exist → setdefault sets it
            os.environ.pop(key, None)
            os.environ.setdefault(key, "--use-fake-ui-for-media-stream")
            assert os.environ[key] == "--use-fake-ui-for-media-stream"

            # Case 2: Key already exists → setdefault does NOT overwrite
            os.environ[key] = "--remote-debugging-port=9222"
            os.environ.setdefault(key, "--use-fake-ui-for-media-stream")
            assert (
                os.environ[key] == "--remote-debugging-port=9222"
            ), "setdefault must not overwrite existing env var values"
        finally:
            os.environ.pop(key, None)

    def test_os_import_available_in_main(self, main_py_content):
        """main.py must import os for setting environment variables."""
        assert "import os" in main_py_content, "main.py must import the os module to set env vars."


# =============================================================================
# TESTS: Future mobile app readiness
# =============================================================================
# These tests verify that the frontend code is structured to work with
# future mobile app wrappers (Capacitor, PWA, React Native).


class TestMobileAppReadiness:
    """Tests that the camera code is ready for mobile app deployment."""

    def test_touch_detection_covers_multi_touch(self, image_upload_content):
        """Touch detection must use maxTouchPoints, not just ontouchstart.

        Modern smartphones report maxTouchPoints >= 5 (multi-touch).
        Using maxTouchPoints > 0 catches:
          - Phones (maxTouchPoints = 5-10)
          - Tablets (maxTouchPoints = 5-10)
          - Touch-enabled laptops (maxTouchPoints = 1-10)
          - Foldables (maxTouchPoints = 5-10)
        """
        assert (
            "maxTouchPoints" in image_upload_content
        ), "Must use maxTouchPoints for reliable touch device detection."

    def test_ontouchstart_as_secondary_signal(self, image_upload_content):
        """Should also check 'ontouchstart' for older browsers.

        Some older mobile browsers don't report maxTouchPoints correctly.
        Checking 'ontouchstart' in window is a secondary signal that
        catches these edge cases.
        """
        assert (
            "ontouchstart" in image_upload_content
        ), "Should check 'ontouchstart' as a secondary touch detection signal."

    def test_capture_environment_for_rear_camera(self, image_upload_content):
        """Must use capture="environment" for the rear camera.

        STREAM's primary use case for camera capture is photographing
        documents, diagrams, whiteboards, and physical objects — tasks
        that require the rear camera, not the selfie camera.

        The capture attribute values are:
          - "environment" = rear camera
          - "user" = front camera (selfie)
        """
        assert (
            'capture="environment"' in image_upload_content
        ), 'Must use capture="environment" to open the rear camera on mobile.'

    def test_video_element_has_playsinline(self, image_upload_content):
        """Video element must have playsInline attribute.

        On iOS Safari, videos without playsInline open in fullscreen mode
        instead of playing inline. This would break the webcam preview
        in the CameraModal. The playsInline attribute prevents this.

        Apple documentation:
        https://webkit.org/blog/6784/new-video-policies-for-ios/
        """
        assert (
            "playsInline" in image_upload_content
        ), "Video element must have playsInline for iOS compatibility."

    def test_video_element_is_muted(self, image_upload_content):
        """Video element must be muted.

        On mobile browsers, autoplay is only allowed for muted videos.
        Even though we don't capture audio, the video element needs the
        muted attribute to autoplay the webcam preview without user
        interaction.
        """
        # Check for the muted attribute in the video element context
        video_section = _extract_between(
            image_upload_content,
            "<video",
            "/>",
        )
        assert (
            video_section is not None and "muted" in video_section
        ), "Video element must be muted for mobile autoplay compatibility."


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def _extract_between(text: str, start_marker: str, end_marker: str) -> str | None:
    """Extract text between two markers (non-greedy).

    Returns the text between the first occurrence of start_marker and
    the next occurrence of end_marker, or None if not found.
    """
    start_idx = text.find(start_marker)
    if start_idx == -1:
        return None
    end_idx = text.find(end_marker, start_idx + len(start_marker))
    if end_idx == -1:
        return None
    return text[start_idx:end_idx]
