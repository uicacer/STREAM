/**
 * ImageUpload.tsx - Image Upload, Camera Capture, and Preview
 * ============================================================
 *
 * This component handles adding images to chat messages. It supports:
 * - Click to open file picker (upload existing images)
 * - Camera capture via webcam (desktop) or native camera (mobile)
 * - Paste images from clipboard (Ctrl+V / Cmd+V)
 * - Automatic compression (max 1024px, JPEG 85%)
 * - Thumbnail previews with remove buttons
 *
 * WHY COMPRESS IMAGES?
 * --------------------
 * Raw images from cameras/screenshots can be 5-15 MB. When base64-encoded
 * (as required by the OpenAI vision format), they grow ~33% larger. This
 * causes two problems:
 *   1. Globus Compute has a 10 MB task payload limit
 *      (https://globus-compute.readthedocs.io/en/stable/limits.html)
 *   2. Large payloads slow down the request and waste bandwidth
 *
 * By compressing to max 1024px and JPEG 85% quality, most images become
 * 100-500 KB — small enough for any tier, yet detailed enough for vision
 * models to analyze accurately.
 *
 * SIZE ENFORCEMENT:
 *   Images are NOT blocked on the frontend. Instead, when the user sends
 *   a message to the Lakeshore tier with images exceeding 6 MB total, a
 *   warning is shown suggesting Local or Cloud tiers. This keeps the UX
 *   simple: attach freely, get feedback only when it matters.
 *
 * CAMERA CAPTURE — CROSS-PLATFORM STRATEGY:
 * ------------------------------------------
 * The camera must work across ALL environments STREAM runs in:
 *   - Desktop mode (PyWebView): macOS WKWebView, Windows WebView2, Linux WebKitGTK
 *   - Server mode (browser):    Chrome, Firefox, Safari, Edge on any OS
 *   - Future mobile app:        Capacitor, PWA, or React Native wrapper
 *
 * We use a THREE-TIER detection strategy based on FEATURE DETECTION
 * (not user-agent sniffing, which is fragile and breaks with new devices):
 *
 *   Tier 1 — Touch device (phone, tablet, foldable):
 *     Uses <input capture="environment"> to open the native camera app.
 *     Native camera is ALWAYS better UX on touch devices — it provides
 *     flash, zoom, HDR, focus controls, tap-to-focus, etc.
 *     Detection: `navigator.maxTouchPoints > 0` (works for iPads with
 *     desktop-class Safari, foldables, and future form factors).
 *
 *   Tier 2 — Desktop with getUserMedia support (most browsers):
 *     Opens a CameraModal with live webcam preview via WebRTC
 *     navigator.mediaDevices.getUserMedia(). User clicks "Take Photo"
 *     to capture a frame from the video stream.
 *     Detection: `navigator.mediaDevices?.getUserMedia` exists.
 *
 *   Tier 3 — Fallback (PyWebView without media support, old browsers):
 *     Opens the regular file picker. The user can select an existing
 *     photo. This always works, everywhere.
 *     Detection: Neither touch nor getUserMedia available.
 *
 * WHY maxTouchPoints INSTEAD OF USER-AGENT:
 *   User-agent strings are unreliable and increasingly spoofed:
 *   - iPads report desktop Safari user agents since iPadOS 13
 *   - Foldable phones (Galaxy Fold) can switch between mobile/desktop
 *   - Capacitor/WebView apps may have custom user agents
 *   - navigator.maxTouchPoints is a hardware capability check that
 *     accurately reflects whether the device has a touchscreen,
 *     regardless of what the user agent says.
 *
 * getUserMedia COMPATIBILITY NOTES:
 *   - Requires a "secure context" (HTTPS or localhost). STREAM serves
 *     on localhost in both modes, so this is satisfied.
 *   - PyWebView (desktop mode) may not support getUserMedia depending
 *     on the renderer (WKWebView, WebView2, WebKitGTK). The code
 *     checks for its existence before attempting to use it.
 *   - macOS .app bundles need NSCameraUsageDescription in Info.plist
 *     (configured in stream.spec).
 *   - Qt-based renderers need QTWEBENGINE_CHROMIUM_FLAGS env var
 *     (configured in stream/desktop/main.py).
 */

import { useRef, useState, useCallback, useEffect } from 'react'
import { ImagePlus, Camera, X, CameraOff } from 'lucide-react'
import { cn } from '../../lib/utils'

const MAX_IMAGE_DIMENSION = 1024
const JPEG_QUALITY = 0.85

/**
 * Maximum total image data per message for Lakeshore tier (in bytes).
 * Globus Compute enforces a 10 MB task payload limit. STREAM uses an
 * 8 MB safety limit, leaving ~2 MB for text/serialization overhead.
 * Reference: https://globus-compute.readthedocs.io/en/stable/limits.html
 */
export const LAKESHORE_MAX_IMAGE_BYTES = 6 * 1024 * 1024  // 6 MB

async function compressImage(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const img = new Image()
    img.onload = () => {
      let { width, height } = img
      if (width > MAX_IMAGE_DIMENSION || height > MAX_IMAGE_DIMENSION) {
        if (width > height) {
          height = Math.round(height * (MAX_IMAGE_DIMENSION / width))
          width = MAX_IMAGE_DIMENSION
        } else {
          width = Math.round(width * (MAX_IMAGE_DIMENSION / height))
          height = MAX_IMAGE_DIMENSION
        }
      }

      const canvas = document.createElement('canvas')
      canvas.width = width
      canvas.height = height
      const ctx = canvas.getContext('2d')!
      ctx.drawImage(img, 0, 0, width, height)

      const dataUrl = canvas.toDataURL('image/jpeg', JPEG_QUALITY)
      resolve(dataUrl)
    }
    img.onerror = () => reject(new Error('Failed to load image'))

    const reader = new FileReader()
    reader.onload = (e) => { img.src = e.target?.result as string }
    reader.onerror = () => reject(new Error('Failed to read file'))
    reader.readAsDataURL(file)
  })
}

/**
 * Capture a frame from a video element as a compressed JPEG data URL.
 */
function captureFrameFromVideo(video: HTMLVideoElement): string {
  const canvas = document.createElement('canvas')
  let { videoWidth: w, videoHeight: h } = video

  if (w > MAX_IMAGE_DIMENSION || h > MAX_IMAGE_DIMENSION) {
    if (w > h) {
      h = Math.round(h * (MAX_IMAGE_DIMENSION / w))
      w = MAX_IMAGE_DIMENSION
    } else {
      w = Math.round(w * (MAX_IMAGE_DIMENSION / h))
      h = MAX_IMAGE_DIMENSION
    }
  }

  canvas.width = w
  canvas.height = h
  const ctx = canvas.getContext('2d')!
  ctx.drawImage(video, 0, 0, w, h)
  return canvas.toDataURL('image/jpeg', JPEG_QUALITY)
}

/**
 * Get total payload size of all attached images (in bytes).
 * The data URL string length closely approximates its byte contribution
 * in the JSON payload since base64 is ASCII.
 */
export function getTotalImageBytes(images: string[]): number {
  return images.reduce((sum, img) => sum + img.length, 0)
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

/**
 * ImagePreviewStrip - Thumbnails shown ABOVE the text input.
 */
export function ImagePreviewStrip({
  images,
  onRemove,
}: {
  images: string[]
  onRemove: (index: number) => void
}) {
  if (images.length === 0) return null

  return (
    <div className="pb-2">
      <div className="flex items-center gap-2 px-1 overflow-x-auto">
        {images.map((dataUrl, index) => (
          <div
            key={index}
            className="relative group flex-shrink-0 w-16 h-16 rounded-lg overflow-hidden
                       border-2 border-muted-foreground/20 hover:border-primary/50 transition-colors"
          >
            <img
              src={dataUrl}
              alt={`Attached image ${index + 1}`}
              className="w-full h-full object-cover"
            />
            <button
              onClick={() => onRemove(index)}
              className="absolute top-0 right-0 p-0.5 bg-red-500 text-white rounded-bl-lg
                         opacity-0 group-hover:opacity-100 transition-opacity"
              aria-label={`Remove image ${index + 1}`}
            >
              <X className="w-3 h-3" />
            </button>
          </div>
        ))}

        <span className="text-xs text-muted-foreground whitespace-nowrap ml-1">
          {images.length} image{images.length !== 1 ? 's'  : ''} ({formatBytes(getTotalImageBytes(images))})
        </span>
      </div>
    </div>
  )
}

// ============================================================================
// Camera detection utilities — exported for testing
// ============================================================================

/**
 * Detect whether the current device has a touchscreen.
 *
 * WHY THIS MATTERS:
 * Touch devices (phones, tablets, foldables) should use the native camera
 * app via <input capture="environment"> because it provides a far better
 * experience: flash, zoom, HDR, tap-to-focus, exposure controls, etc.
 * A getUserMedia webcam preview can't compete with a native camera app.
 *
 * WHY NOT USER-AGENT SNIFFING:
 * User-agent strings are unreliable for device type detection:
 *   - iPads report desktop Safari UAs since iPadOS 13
 *   - Galaxy Fold switches between mobile/desktop UAs
 *   - Capacitor/Tauri apps may have custom UAs
 *   - New device categories appear every year
 *
 * Instead, we check `navigator.maxTouchPoints` — a hardware capability
 * that directly answers "does this device have a touchscreen?" regardless
 * of the user agent string. It returns:
 *   - 0 on desktops without touchscreens
 *   - 1+ on phones, tablets, foldables, touch-enabled laptops
 *   - 5+ on most modern smartphones (multi-touch)
 *
 * We also check `ontouchstart` as a secondary signal for older browsers
 * that might not report maxTouchPoints correctly.
 */
export function isTouchDevice(): boolean {
  return (
    navigator.maxTouchPoints > 0 ||
    'ontouchstart' in window
  )
}

/**
 * Check if the WebRTC getUserMedia API is available.
 *
 * getUserMedia requires:
 *   1. A "secure context" — the page must be served over HTTPS or localhost.
 *      STREAM serves on localhost in both desktop and server modes.
 *   2. Browser/WebView support — all modern browsers support it, but some
 *      embedded WebViews (PyWebView on macOS/Linux) may not expose it.
 *   3. An actual camera device — even if the API exists, calling it will
 *      fail with NotFoundError if no camera hardware is present.
 *
 * This function only checks condition (1) and (2). Condition (3) is
 * handled by the CameraModal's error handling when getUserMedia is called.
 */
export function hasGetUserMedia(): boolean {
  return !!(navigator.mediaDevices?.getUserMedia)
}

/**
 * Determine which camera strategy to use, based on progressive feature
 * detection. This is the core cross-platform logic.
 *
 * Returns one of three strategies:
 *   'native-camera' — Touch device: use <input capture> for native camera app
 *   'webcam-modal'  — Desktop with getUserMedia: open CameraModal
 *   'file-picker'   — Fallback: open regular file picker
 */
export function getCameraStrategy(): 'native-camera' | 'webcam-modal' | 'file-picker' {
  if (isTouchDevice()) return 'native-camera'
  if (hasGetUserMedia()) return 'webcam-modal'
  return 'file-picker'
}

// ============================================================================
// CameraModal - Live webcam preview with capture (desktop browsers)
// ============================================================================
//
// This modal opens the user's webcam via getUserMedia and shows a live video
// preview. The user clicks "Take Photo" to capture a single frame.
//
// CROSS-PLATFORM NOTES:
//   - Desktop browsers (Chrome, Firefox, Safari, Edge): Works on all OSes.
//   - PyWebView (macOS WKWebView): May work if NSCameraUsageDescription is
//     in Info.plist and the OS grants permission. Falls back gracefully.
//   - PyWebView (Windows WebView2): Usually works; WebView2 shows a system
//     permission prompt the first time.
//   - PyWebView (Linux WebKitGTK): Depends on GStreamer plugins; may fail.
//     The error handler catches this and shows a helpful message.
//
// ERROR HANDLING:
//   The modal handles every getUserMedia error type defined in the WebRTC
//   spec (https://developer.mozilla.org/en-US/docs/Web/API/MediaDevices/getUserMedia):
//     - NotAllowedError:      User denied camera permission
//     - NotFoundError:        No camera hardware detected
//     - NotReadableError:     Camera is busy (Zoom, Teams, etc.)
//     - OverconstrainedError: Requested constraints can't be satisfied
//                             (e.g., 'environment' facing mode on a desktop
//                             webcam that only has a 'user' facing camera).
//                             We automatically retry without facingMode.
//     - AbortError / other:   Generic fallback message
//
// CLEANUP:
//   The video stream MUST be stopped when the modal closes, otherwise the
//   camera remains active (the "recording" indicator stays on, and other
//   apps can't use the camera). We stop all tracks in both the cleanup
//   effect and the handleClose function.

function CameraModal({
  onCapture,
  onClose,
}: {
  onCapture: (dataUrl: string) => void
  onClose: () => void
}) {
  const videoRef = useRef<HTMLVideoElement>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [ready, setReady] = useState(false)

  useEffect(() => {
    let cancelled = false

    /**
     * Attempt to start the camera with the given video constraints.
     * Returns the MediaStream on success, or null on failure (after
     * setting the error state).
     */
    async function tryGetUserMedia(
      constraints: MediaStreamConstraints
    ): Promise<MediaStream | null> {
      try {
        return await navigator.mediaDevices.getUserMedia(constraints)
      } catch (err) {
        if (cancelled) return null

        console.error('[Camera] getUserMedia failed:', err)

        if (err instanceof DOMException) {
          switch (err.name) {
            case 'NotAllowedError':
              setError(
                'Camera access was denied. Please allow camera access in your ' +
                'browser or system settings, then try again.'
              )
              return null

            case 'NotFoundError':
              setError(
                'No camera found. Make sure a camera is connected to your ' +
                'computer. You can use the Upload button to add existing images.'
              )
              return null

            case 'NotReadableError':
              setError(
                'Camera is in use by another application (e.g., Zoom, Teams, ' +
                'FaceTime). Close the other app and try again.'
              )
              return null

            case 'OverconstrainedError':
              // This happens when the requested facingMode ('environment')
              // isn't available. Desktop webcams typically only have 'user'
              // facing mode (front-facing). We return null here and the
              // caller will retry without the facingMode constraint.
              return null

            default:
              setError(
                'Could not access camera. If you\'re using the desktop app, ' +
                'try opening STREAM in your browser instead. You can also ' +
                'use the Upload button to add existing images.'
              )
              return null
          }
        }

        setError(
          'Unexpected camera error. Use the Upload button to add images, ' +
          'or try opening STREAM in your browser.'
        )
        return null
      }
    }

    async function startCamera() {
      // Guard: if getUserMedia isn't available at all (e.g., PyWebView
      // on some platforms), show a helpful error immediately rather than
      // crashing with "Cannot read properties of undefined".
      if (!navigator.mediaDevices?.getUserMedia) {
        setError(
          'Camera is not supported in this environment. This can happen in ' +
          'the desktop app on some operating systems. Try opening STREAM in ' +
          'your browser instead, or use the Upload button to add images.'
        )
        return
      }

      // First attempt: request the rear-facing camera ('environment').
      // This is the preferred camera for capturing documents, whiteboards,
      // diagrams, etc. — the primary use case for multimodal AI queries.
      // On desktop webcams, this constraint is often ignored (most only
      // have front-facing cameras), but on mobile/tablets with multiple
      // cameras, it selects the right one.
      let stream = await tryGetUserMedia({
        video: {
          facingMode: 'environment',
          width: { ideal: 1280 },
          height: { ideal: 720 },
        },
        audio: false,
      })

      // If the 'environment' facingMode failed with OverconstrainedError,
      // retry without it. This handles desktop webcams that only support
      // 'user' facing mode, and embedded cameras in laptops.
      if (!stream && !error) {
        stream = await tryGetUserMedia({
          video: {
            width: { ideal: 1280 },
            height: { ideal: 720 },
          },
          audio: false,
        })
      }

      // If we still don't have a stream, either an error was already set
      // by tryGetUserMedia, or the component was cancelled.
      if (!stream || cancelled) {
        if (stream) stream.getTracks().forEach(t => t.stop())
        return
      }

      streamRef.current = stream
      if (videoRef.current) {
        videoRef.current.srcObject = stream
        videoRef.current.play()
        setReady(true)
      }
    }

    startCamera()

    // Cleanup: stop all media tracks when the modal unmounts.
    // This releases the camera hardware so other apps can use it,
    // and turns off the "recording" indicator in the OS.
    return () => {
      cancelled = true
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(t => t.stop())
        streamRef.current = null
      }
    }
  }, [])  // eslint-disable-line react-hooks/exhaustive-deps -- runs once on mount

  const handleCapture = () => {
    if (!videoRef.current || !ready) return
    const dataUrl = captureFrameFromVideo(videoRef.current)
    onCapture(dataUrl)
  }

  const handleClose = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(t => t.stop())
      streamRef.current = null
    }
    onClose()
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
      <div className="bg-background rounded-2xl shadow-2xl border border-border overflow-hidden max-w-lg w-full mx-4">
        {/* Header */}
        <div className="flex items-center justify-between px-4 py-3 border-b border-border">
          <h3 className="font-medium text-sm">Take a Photo</h3>
          <button
            onClick={handleClose}
            className="p-1 rounded-lg hover:bg-muted transition-colors"
            aria-label="Close camera"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Video preview or error */}
        <div className="relative aspect-video bg-black flex items-center justify-center">
          {error ? (
            <div className="text-center p-6">
              <CameraOff className="w-10 h-10 text-muted-foreground mx-auto mb-3" />
              <p className="text-sm text-muted-foreground max-w-xs mx-auto">{error}</p>
            </div>
          ) : (
            <>
              <video
                ref={videoRef}
                autoPlay
                playsInline
                muted
                className="w-full h-full object-cover"
              />
              {!ready && (
                <div className="absolute inset-0 flex items-center justify-center bg-black/50">
                  <p className="text-sm text-white/70">Starting camera...</p>
                </div>
              )}
            </>
          )}
        </div>

        {/* Actions */}
        <div className="flex items-center justify-center gap-3 px-4 py-3 border-t border-border">
          {!error && (
            <button
              onClick={handleCapture}
              disabled={!ready}
              className={cn(
                "flex items-center gap-2 px-5 py-2 rounded-xl text-sm font-medium transition-all",
                ready
                  ? "bg-primary text-primary-foreground hover:bg-primary/90"
                  : "bg-muted text-muted-foreground cursor-not-allowed"
              )}
            >
              <Camera className="w-4 h-4" />
              Take Photo
            </button>
          )}
          <button
            onClick={handleClose}
            className="px-5 py-2 rounded-xl text-sm hover:bg-muted transition-colors"
          >
            {error ? 'Close' : 'Cancel'}
          </button>
        </div>
      </div>
    </div>
  )
}

// ============================================================================
// ImageUpload - Upload button + Camera button
// ============================================================================
//
// Two buttons are always visible:
//   1. Upload (ImagePlus icon) — opens file picker to select existing images
//   2. Camera (Camera icon) — behavior depends on the device:
//
//   ┌──────────────────┬─────────────────────────────────────────────┐
//   │ Device Type      │ What happens when Camera is clicked         │
//   ├──────────────────┼─────────────────────────────────────────────┤
//   │ Touch device     │ Opens native camera app via <input capture> │
//   │ (phone, tablet)  │ for best UX (flash, zoom, HDR, etc.)       │
//   ├──────────────────┼─────────────────────────────────────────────┤
//   │ Desktop with     │ Opens CameraModal with live webcam preview  │
//   │ getUserMedia     │ (Chrome, Firefox, Safari, Edge, WebView2)   │
//   ├──────────────────┼─────────────────────────────────────────────┤
//   │ Desktop without  │ Opens file picker (PyWebView on some OSes   │
//   │ getUserMedia     │ where camera API is not available)           │
//   └──────────────────┴─────────────────────────────────────────────┘
//
// The <input capture="environment"> element uses the REAR camera by default.
// This is intentional: STREAM's primary use case for camera capture is
// photographing documents, diagrams, whiteboards, and physical objects
// for AI analysis — not selfies.

interface ImageUploadProps {
  images: string[]
  onImagesChange: (images: string[]) => void
  disabled?: boolean
}

export function ImageUpload({ images, onImagesChange, disabled }: ImageUploadProps) {
  const fileInputRef = useRef<HTMLInputElement>(null)
  const cameraInputRef = useRef<HTMLInputElement>(null)
  const [showCamera, setShowCamera] = useState(false)

  const processFiles = useCallback(async (files: FileList | File[]) => {
    const fileArray = Array.from(files)
    const imageFiles = fileArray.filter(f => f.type.startsWith('image/'))
    if (imageFiles.length === 0) return

    const newImages: string[] = []

    for (const file of imageFiles) {
      try {
        const compressed = await compressImage(file)
        newImages.push(compressed)
      } catch (err) {
        console.error('[ImageUpload] Failed to compress image:', err)
      }
    }

    if (newImages.length > 0) {
      onImagesChange([...images, ...newImages])
    }
  }, [images, onImagesChange])

  const handleFileChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      processFiles(e.target.files)
      e.target.value = ''
    }
  }, [processFiles])

  /**
   * Camera button click handler — uses feature detection to pick the
   * best camera strategy for the current device/environment.
   *
   * See getCameraStrategy() for the three-tier detection logic.
   */
  const handleCameraClick = useCallback(() => {
    const strategy = getCameraStrategy()

    switch (strategy) {
      case 'native-camera':
        // Touch device: open native camera app for best UX.
        // The hidden <input capture="environment"> element triggers the
        // OS camera app (iOS Camera, Android Camera, etc.).
        cameraInputRef.current?.click()
        break

      case 'webcam-modal':
        // Desktop browser with getUserMedia: open webcam preview modal.
        // The CameraModal handles all error cases internally.
        setShowCamera(true)
        break

      case 'file-picker':
        // No camera API available (e.g., PyWebView without media support).
        // Fall back to the regular file picker so the user can still
        // select an existing photo from their filesystem.
        fileInputRef.current?.click()
        break
    }
  }, [])

  const handleCameraCapture = useCallback((dataUrl: string) => {
    onImagesChange([...images, dataUrl])
    setShowCamera(false)
  }, [images, onImagesChange])

  return (
    <>
      <div className="flex items-center">
        {/* Upload existing image */}
        <button
          onClick={() => fileInputRef.current?.click()}
          disabled={disabled}
          className={cn(
            "p-3 rounded-xl transition-all",
            "hover:bg-muted text-muted-foreground",
            "disabled:opacity-50 disabled:cursor-not-allowed"
          )}
          aria-label="Upload image"
          title="Upload image"
        >
          <ImagePlus className="w-5 h-5" />
        </button>

        {/* Take a photo */}
        <button
          onClick={handleCameraClick}
          disabled={disabled}
          className={cn(
            "p-2 -ml-1 rounded-xl transition-all",
            "hover:bg-muted text-muted-foreground",
            "disabled:opacity-50 disabled:cursor-not-allowed"
          )}
          aria-label="Take a photo"
          title="Take a photo"
        >
          <Camera className="w-4 h-4" />
        </button>

        {/* Hidden file input for uploading existing images */}
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          multiple
          onChange={handleFileChange}
          className="hidden"
          aria-hidden="true"
        />

        {/*
          Hidden file input for native camera capture on touch devices.

          The `capture` attribute tells mobile browsers to open the camera
          instead of the file picker:
            - "environment" = rear camera (for documents, whiteboards, objects)
            - "user" = front camera (for selfies)

          On desktop browsers, `capture` is ignored — the regular file picker
          opens instead. This is fine because desktop users get the CameraModal.

          WHY A SEPARATE INPUT:
          We can't reuse the fileInputRef because it has `multiple` set (for
          selecting multiple existing images). The camera input should only
          capture one photo at a time.
        */}
        <input
          ref={cameraInputRef}
          type="file"
          accept="image/*"
          capture="environment"
          onChange={handleFileChange}
          className="hidden"
          aria-hidden="true"
        />
      </div>

      {/* Webcam modal for desktop camera capture */}
      {showCamera && (
        <CameraModal
          onCapture={handleCameraCapture}
          onClose={() => setShowCamera(false)}
        />
      )}
    </>
  )
}

export { compressImage }
