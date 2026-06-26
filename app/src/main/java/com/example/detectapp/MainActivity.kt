package com.example.detectapp

import android.Manifest
import android.content.ClipData
import android.content.ClipboardManager
import android.content.ContentValues
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.RectF
import android.graphics.Typeface
import android.net.Uri
import android.os.Bundle
import android.provider.CalendarContract
import android.provider.MediaStore
import android.util.Base64
import android.util.Log
import android.util.Size
import android.view.MotionEvent
import android.view.View
import android.view.ViewGroup
import android.widget.AdapterView
import android.widget.ArrayAdapter
import android.widget.Button
import android.widget.FrameLayout
import android.widget.HorizontalScrollView
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.Spinner
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.ImageProxy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import androidx.core.view.updateLayoutParams
import androidx.core.view.updatePadding
import com.example.detectapp.databinding.ActivityMainBinding
import com.google.android.gms.tasks.Tasks
import com.google.mlkit.nl.translate.TranslateLanguage
import com.google.mlkit.nl.translate.Translation
import com.google.mlkit.nl.translate.Translator
import com.google.mlkit.nl.translate.TranslatorOptions
import com.google.mlkit.common.model.DownloadConditions
import com.google.mlkit.nl.entityextraction.Entity
import com.google.mlkit.nl.entityextraction.EntityExtraction
import com.google.mlkit.nl.entityextraction.EntityExtractionParams
import com.google.mlkit.nl.entityextraction.EntityExtractorOptions
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.text.TextRecognition
import com.google.mlkit.vision.text.korean.KoreanTextRecognizerOptions
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.json.JSONArray
import org.json.JSONObject
import org.opencv.core.CvType
import org.opencv.core.Mat
import java.io.ByteArrayOutputStream
import java.io.OutputStream
import java.net.HttpURLConnection
import java.net.URL
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private lateinit var cameraExecutor: ExecutorService
    private var cameraProvider: ProcessCameraProvider? = null
    private var imageCapture: ImageCapture? = null

    // Live-tuned state (read from the analysis thread).
    @Volatile private var thresholdLow = 50.0
    @Volatile private var thresholdHigh = 150.0
    @Volatile private var mode = 4               // default: 엣지 오버레이

    private var resolutionIndex = 0
    private var usingFrontCamera = false

    // Reused across frames to avoid per-frame allocation.
    private var outMat: Mat? = null
    private var outBitmap: Bitmap? = null

    private var lastSavedUri: Uri? = null

    private val recognizer by lazy {
        TextRecognition.getClient(KoreanTextRecognizerOptions.Builder().build())
    }

    private val entityExtractor by lazy {
        EntityExtraction.getClient(
            EntityExtractorOptions.Builder(EntityExtractorOptions.KOREAN).build()
        )
    }
    @Volatile private var entityModelReady = false

    // --- Live AR translation state ---
    private val liveExecutor = Executors.newSingleThreadExecutor()
    private val liveKoEn by lazy {
        Translation.getClient(
            TranslatorOptions.Builder()
                .setSourceLanguage(TranslateLanguage.KOREAN)
                .setTargetLanguage(TranslateLanguage.ENGLISH).build()
        )
    }
    private val liveEnKo by lazy {
        Translation.getClient(
            TranslatorOptions.Builder()
                .setSourceLanguage(TranslateLanguage.ENGLISH)
                .setTargetLanguage(TranslateLanguage.KOREAN).build()
        )
    }
    @Volatile private var liveModelsReady = false
    @Volatile private var liveDirection = DIR_BOTH
    @Volatile private var liveOcrBusy = false
    private var lastFrameSig: IntArray? = null   // 32x32 luma of previous frame (motion)
    private var stillSince = 0L                  // when the camera last settled
    @Volatile private var liveOverlay: List<OverlayItem> = emptyList()
    private val livePaintBg = Paint().apply { color = Color.WHITE }
    private val livePaintText = Paint().apply {
        color = Color.BLACK; isAntiAlias = true; isFakeBoldText = true
    }

    // FPS / timing bookkeeping.
    private var frameCount = 0
    private var lastFpsTimestamp = System.currentTimeMillis()
    private var procMicrosAccum = 0L

    private val targetSizes = arrayOf(Size(640, 480), Size(1280, 720), Size(1920, 1080))

    private val requestPermission =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
            if (granted) bindCamera()
            else Toast.makeText(this, R.string.camera_permission_denied, Toast.LENGTH_LONG).show()
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)
        supportActionBar?.hide()  // fullscreen camera feel

        if (!OpenCVLoader.initLocal()) {
            Log.e(TAG, "OpenCV initialization failed")
            Toast.makeText(this, "OpenCV 초기화 실패", Toast.LENGTH_LONG).show()
        }

        applyInsets()
        setupControls()
        cameraExecutor = Executors.newSingleThreadExecutor()

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            == PackageManager.PERMISSION_GRANTED
        ) {
            startCamera()
        } else {
            requestPermission.launch(Manifest.permission.CAMERA)
        }
    }

    // targetSdk 35+ forces edge-to-edge: keep overlays out from under the system bars.
    private fun applyInsets() {
        val basePanelPad = binding.controlPanel.paddingBottom
        ViewCompat.setOnApplyWindowInsetsListener(binding.root) { _, insets ->
            val bars = insets.getInsets(WindowInsetsCompat.Type.systemBars())
            binding.fpsText.updateLayoutParams<FrameLayout.LayoutParams> {
                topMargin = bars.top + dp(12)
            }
            binding.btnFlip.updateLayoutParams<FrameLayout.LayoutParams> {
                topMargin = bars.top + dp(12)
            }
            binding.btnDir.updateLayoutParams<FrameLayout.LayoutParams> {
                topMargin = bars.top + dp(12)
            }
            binding.controlPanel.updatePadding(bottom = basePanelPad + bars.bottom)
            insets
        }
    }

    private fun dp(value: Int): Int = (value * resources.displayMetrics.density).toInt()

    private fun setupControls() {
        updateLowLabel(thresholdLow.toInt())
        updateHighLabel(thresholdHigh.toInt())

        binding.sliderLow.addOnChangeListener { _, value, _ ->
            thresholdLow = value.toDouble()
            updateLowLabel(value.toInt())
        }
        binding.sliderHigh.addOnChangeListener { _, value, _ ->
            thresholdHigh = value.toDouble()
            updateHighLabel(value.toInt())
        }

        binding.spinnerMode.setSelection(mode)
        binding.spinnerMode.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(p: AdapterView<*>?, v: View?, pos: Int, id: Long) {
                mode = pos
                binding.btnDir.visibility = if (pos == LIVE_TRANSLATE) View.VISIBLE else View.GONE
                if (pos == LIVE_TRANSLATE) {
                    liveOverlay = emptyList()
                    ensureLiveModels()
                }
            }
            override fun onNothingSelected(p: AdapterView<*>?) {}
        }

        updateDirLabel()
        binding.btnDir.setOnClickListener {
            liveDirection = (liveDirection + 1) % 3   // 자동 → 한→영 → 영→한
            liveOverlay = emptyList()                 // clear so it re-translates
            updateDirLabel()
        }

        binding.spinnerRes.setSelection(resolutionIndex)
        binding.spinnerRes.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(p: AdapterView<*>?, v: View?, pos: Int, id: Long) {
                if (pos != resolutionIndex) {
                    resolutionIndex = pos
                    bindCamera()  // rebind with new resolution
                }
            }
            override fun onNothingSelected(p: AdapterView<*>?) {}
        }

        binding.btnCapture.setOnClickListener {
            // Always grab a full-res still: document mode flattens + OCRs it,
            // other modes OCR the photo directly (translate any scene's text).
            captureAndRecognize()
        }
        binding.btnShare.setOnClickListener { shareLastCapture() }
        binding.btnFlip.setOnClickListener {
            usingFrontCamera = !usingFrontCamera
            // Selfie convention: mirror the front-camera preview.
            binding.preview.scaleX = if (usingFrontCamera) -1f else 1f
            bindCamera()
        }
    }

    private fun updateLowLabel(v: Int) { binding.labelLow.text = getString(R.string.label_low, v) }
    private fun updateHighLabel(v: Int) { binding.labelHigh.text = getString(R.string.label_high, v) }

    private fun startCamera() {
        val providerFuture = ProcessCameraProvider.getInstance(this)
        providerFuture.addListener({
            cameraProvider = providerFuture.get()
            bindCamera()
        }, ContextCompat.getMainExecutor(this))
    }

    private fun bindCamera() {
        val provider = cameraProvider ?: return

        val selector = ResolutionSelector.Builder()
            .setResolutionStrategy(
                ResolutionStrategy(
                    targetSizes[resolutionIndex],
                    ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER
                )
            )
            .build()

        val analysis = ImageAnalysis.Builder()
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
            .setOutputImageRotationEnabled(true)
            .setResolutionSelector(selector)
            .build()

        analysis.setAnalyzer(cameraExecutor) { proxy -> processFrame(proxy) }

        // Full-resolution still capture for document scanning / OCR.
        val capture = ImageCapture.Builder()
            .setCaptureMode(ImageCapture.CAPTURE_MODE_MAXIMIZE_QUALITY)
            .build()
        imageCapture = capture

        val lens = if (usingFrontCamera) CameraSelector.DEFAULT_FRONT_CAMERA
        else CameraSelector.DEFAULT_BACK_CAMERA

        try {
            provider.unbindAll()
            provider.bindToLifecycle(this, lens, analysis, capture)
        } catch (e: Exception) {
            Log.e(TAG, "Camera binding failed", e)
        }
    }

    private fun processFrame(image: ImageProxy) {
        try {
            val width = image.width
            val height = image.height
            val plane = image.planes[0]
            val rowStride = plane.rowStride

            // Wrap the RGBA plane (with row padding) into a Mat, then crop to width.
            val full = Mat(height, rowStride / 4, CvType.CV_8UC4)
            val buffer = plane.buffer
            val bytes = ByteArray(buffer.remaining())
            buffer.get(bytes)
            full.put(0, 0, bytes)
            val rgba = full.submat(0, height, 0, width)

            var bmp = outBitmap
            if (bmp == null || bmp.width != width || bmp.height != height) {
                bmp = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
                outBitmap = bmp
            }

            val micros: Long
            if (mode == LIVE_TRANSLATE) {
                Utils.matToBitmap(rgba, bmp)        // clean camera frame
                renderLiveTranslate(bmp)            // overlay translations, schedule OCR
                micros = 0
            } else {
                val out = outMat ?: Mat().also { outMat = it }
                micros = nativeProcess(
                    rgba.nativeObjAddr, out.nativeObjAddr, mode, thresholdLow, thresholdHigh
                )
                Utils.matToBitmap(out, bmp)
            }

            full.release()
            rgba.release()

            val display = bmp
            runOnUiThread {
                binding.preview.setImageBitmap(display)
                tickFps(micros)
            }
        } catch (e: Exception) {
            Log.e(TAG, "Frame processing failed", e)
        } finally {
            image.close()
        }
    }

    // ---- Live AR translation -----------------------------------------------

    private class OverlayItem(val rect: RectF, val text: String, val source: String)

    /** Pre-downloads ko↔en models so live translation can run without stalls. */
    private fun ensureLiveModels() {
        if (liveModelsReady) return
        toast(getString(R.string.live_preparing))
        liveExecutor.execute {
            try {
                val c = DownloadConditions.Builder().build()
                Tasks.await(liveKoEn.downloadModelIfNeeded(c))
                Tasks.await(liveEnKo.downloadModelIfNeeded(c))
                liveModelsReady = true
                runOnUiThread { toast(getString(R.string.live_ready)) }
            } catch (e: Exception) {
                Log.e(TAG, "live model prep failed", e)
                runOnUiThread { toast(getString(R.string.model_download_failed)) }
            }
        }
    }

    /**
     * Draws the cached translations, and only re-scans when the camera has actually
     * moved to a new scene — holding still "locks" the translation so it stops flickering.
     */
    private fun renderLiveTranslate(frame: Bitmap) {
        val now = System.currentTimeMillis()
        val sig = frameSignature(frame)
        val prev = lastFrameSig
        lastFrameSig = sig
        val moving = prev != null && sigDiff(sig, prev) > MOTION_THRESHOLD

        if (moving) {
            // While the camera moves, hide stale boxes (they'd drift / overlap).
            if (liveOverlay.isNotEmpty()) liveOverlay = emptyList()
            stillSince = now
        } else if (liveModelsReady && !liveOcrBusy && liveOverlay.isEmpty() &&
            now - stillSince > 400
        ) {
            // Settled on a new view → scan once; result stays put until next move.
            liveOcrBusy = true
            val copy = frame.copy(Bitmap.Config.ARGB_8888, false)
            liveExecutor.execute { runLiveOcr(copy) }
        }

        val canvas = Canvas(frame)
        for (item in liveOverlay) drawTranslatedBox(canvas, item)
    }

    /** Coarse 32x32 luma fingerprint of a frame, for cheap motion detection. */
    private fun frameSignature(bmp: Bitmap): IntArray {
        val small = Bitmap.createScaledBitmap(bmp, 32, 32, true)
        val px = IntArray(32 * 32)
        small.getPixels(px, 0, 32, 0, 0, 32, 32)
        small.recycle()
        val lum = IntArray(px.size)
        for (i in px.indices) {
            val c = px[i]
            lum[i] = ((c ushr 16 and 0xFF) * 77 + (c ushr 8 and 0xFF) * 150 + (c and 0xFF) * 29) shr 8
        }
        return lum
    }

    private fun sigDiff(a: IntArray, b: IntArray): Int {
        if (a.size != b.size) return Int.MAX_VALUE
        var sum = 0
        for (i in a.indices) sum += kotlin.math.abs(a[i] - b[i])
        return sum / a.size
    }

    private fun runLiveOcr(bmp: Bitmap) {
        try {
            val result = Tasks.await(recognizer.process(InputImage.fromBitmap(bmp, 0)))
            val items = ArrayList<OverlayItem>()
            // Per-line so each box stays line-sized (a whole-paragraph block would draw
            // one giant white box with tiny, unreadable text). Computed on the current
            // still frame, so boxes line up exactly with the text.
            for (block in result.textBlocks) {
                for (line in block.lines) {
                    val box = line.boundingBox ?: continue
                    val source = line.text.trim()
                    if (source.length < 2) continue
                    val translated = translateLiveOrNull(source) ?: continue
                    items.add(OverlayItem(RectF(box), translated, source))
                }
            }
            liveOverlay = items
        } catch (e: Exception) {
            Log.e(TAG, "live OCR failed", e)
        } finally {
            liveOcrBusy = false
            bmp.recycle()
        }
    }

    /** Translates per the chosen direction; null = leave this text alone (don't overlay). */
    private fun translateLiveOrNull(text: String): String? {
        val hangul = hasHangul(text)
        val latin = text.any { it in 'a'..'z' || it in 'A'..'Z' }
        val translator = when (liveDirection) {
            DIR_KO_EN -> if (hangul) liveKoEn else null
            DIR_EN_KO -> if (latin && !hangul) liveEnKo else null
            else -> if (hangul) liveKoEn else if (latin) liveEnKo else null
        } ?: return null
        return try {
            Tasks.await(translator.translate(text))
        } catch (e: Exception) {
            null
        }
    }

    private fun updateDirLabel() {
        binding.btnDir.text = when (liveDirection) {
            DIR_KO_EN -> getString(R.string.dir_ko_en)
            DIR_EN_KO -> getString(R.string.dir_en_ko)
            else -> getString(R.string.dir_auto)
        }
    }

    /** Covers the original text region with a box and the translated text, sized to fit. */
    private fun drawTranslatedBox(canvas: Canvas, item: OverlayItem) {
        val r = item.rect
        canvas.drawRect(r, livePaintBg)

        var size = r.height() * 0.78f
        if (size < 1f) return
        livePaintText.textSize = size
        val maxW = r.width() * 0.96f
        val w = livePaintText.measureText(item.text)
        if (w > maxW && w > 0f) {
            size *= maxW / w
            livePaintText.textSize = size
        }
        val fm = livePaintText.fontMetrics
        val baseline = r.centerY() - (fm.ascent + fm.descent) / 2f
        canvas.drawText(item.text, r.left + r.width() * 0.02f, baseline, livePaintText)
    }

    /** Takes a full-resolution still, then OCRs it (warping first in document mode). */
    private fun captureAndRecognize() {
        val ic = imageCapture ?: return
        val isDoc = mode == MODE_DOC
        toast(getString(R.string.capturing))
        ic.takePicture(cameraExecutor, object : ImageCapture.OnImageCapturedCallback() {
            override fun onCaptureSuccess(image: ImageProxy) {
                try {
                    val bitmap = jpegProxyToBitmap(image)
                    if (isDoc) processDocCapture(bitmap) else processPhotoCapture(bitmap)
                } catch (e: Exception) {
                    Log.e(TAG, "Capture failed", e)
                    runOnUiThread { toast(getString(R.string.save_failed)) }
                } finally {
                    image.close()
                }
            }

            override fun onError(exc: ImageCaptureException) {
                Log.e(TAG, "takePicture error", exc)
                runOnUiThread { toast(getString(R.string.save_failed)) }
            }
        })
    }

    /** Saves the photo, then OCRs an OCR-enhanced version (better full-scene accuracy). */
    private fun processPhotoCapture(bitmap: Bitmap) {
        val uri = saveToGallery(bitmap)  // keep the original color photo

        // Contrast/sharpen the photo so full-scene text reads more reliably.
        val src = Mat()
        Utils.bitmapToMat(bitmap, src)
        val enhanced = Mat()
        nativeEnhance(src.nativeObjAddr, enhanced.nativeObjAddr)
        val enhancedBmp = Bitmap.createBitmap(
            enhanced.cols(), enhanced.rows(), Bitmap.Config.ARGB_8888
        )
        Utils.matToBitmap(enhanced, enhancedBmp)
        src.release()
        enhanced.release()

        runOnUiThread {
            if (uri != null) {
                lastSavedUri = uri
                toast(getString(R.string.saved))
            } else {
                toast(getString(R.string.save_failed))
            }
            runOcr(enhancedBmp)
        }
    }

    private fun processDocCapture(srcBitmap: Bitmap) {
        val mat = Mat()
        Utils.bitmapToMat(srcBitmap, mat)  // CV_8UC4 RGBA
        val warped = Mat()
        val ok = nativeWarpDocument(
            mat.nativeObjAddr, warped.nativeObjAddr, thresholdLow, thresholdHigh
        )
        mat.release()
        if (!ok) {
            warped.release()
            runOnUiThread { toast(getString(R.string.no_document)) }
            return
        }
        val b = Bitmap.createBitmap(warped.cols(), warped.rows(), Bitmap.Config.ARGB_8888)
        Utils.matToBitmap(warped, b)
        warped.release()

        val uri = saveToGallery(b)
        runOnUiThread {
            if (uri != null) {
                lastSavedUri = uri
                toast(getString(R.string.saved))
            } else {
                toast(getString(R.string.save_failed))
            }
            runOcr(b)
        }
    }

    private fun jpegProxyToBitmap(image: ImageProxy): Bitmap {
        val buffer = image.planes[0].buffer
        val bytes = ByteArray(buffer.remaining())
        buffer.get(bytes)
        val decoded = BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
        val rot = image.imageInfo.rotationDegrees
        if (rot == 0) return decoded
        val m = Matrix().apply { postRotate(rot.toFloat()) }
        return Bitmap.createBitmap(decoded, 0, 0, decoded.width, decoded.height, m, true)
    }

    /** On-device OCR of the scanned document, then a copy/share dialog. */
    private fun runOcr(bitmap: Bitmap) {
        toast(getString(R.string.ocr_running))
        val input = InputImage.fromBitmap(bitmap, 0)
        recognizer.process(input)
            .addOnSuccessListener { showBlockPicker(bitmap, it) }
            .addOnFailureListener { e ->
                Log.e(TAG, "OCR failed", e)
                toast(getString(R.string.ocr_failed))
            }
    }

    /**
     * Groups OCR lines into rows by vertical position (close centers = same row),
     * each sorted left-to-right, so the grid survives per-cell translation.
     */
    private fun rowsFromLines(
        lines: List<com.google.mlkit.vision.text.Text.Line>
    ): List<List<String>> {
        data class Item(val text: String, val box: android.graphics.Rect)

        val items = lines.mapNotNull { ln -> ln.boundingBox?.let { Item(ln.text, it) } }
        if (items.isEmpty()) return emptyList()

        val sorted = items.sortedBy { it.box.top }
        val rows = ArrayList<ArrayList<Item>>()
        for (item in sorted) {
            val row = rows.lastOrNull()
            if (row != null) {
                val avgCenter = row.map { it.box.exactCenterY() }.average().toFloat()
                val threshold = item.box.height() * 0.6f
                if (kotlin.math.abs(item.box.exactCenterY() - avgCenter) <= threshold) {
                    row.add(item)
                    continue
                }
            }
            rows.add(arrayListOf(item))
        }
        return rows.map { r -> r.sortedBy { it.box.left }.map { it.text } }
    }

    /**
     * Converts one block to rows. Vertically-stacked text (세로쓰기) — a tall, narrow
     * block whose lines are mostly single characters — is joined into one horizontal
     * word so it isn't split top-to-bottom (best-effort; per-char accuracy is ML Kit's).
     */
    private fun blockToRows(
        block: com.google.mlkit.vision.text.Text.TextBlock
    ): List<List<String>> {
        val lines = block.lines
        val box = block.boundingBox
        val shortLines = lines.count { it.text.trim().length <= 2 }
        val isVertical = box != null && box.height() > box.width() &&
                lines.size >= 3 && shortLines >= (lines.size * 0.6).toInt().coerceAtLeast(2)

        if (isVertical) {
            val joined = lines
                .sortedBy { it.boundingBox?.top ?: 0 }
                .joinToString("") { it.text.trim() }
            return listOf(listOf(joined))
        }
        return rowsFromLines(lines)
    }

    private fun buildLayoutRows(result: com.google.mlkit.vision.text.Text): List<List<String>> =
        result.textBlocks.flatMap { blockToRows(it) }

    private fun renderRows(rows: List<List<String>>): String =
        rows.joinToString("\n") { it.joinToString("    ") }

    /**
     * Shows the captured image with each detected text block outlined. Tapping a
     * block opens just that paragraph (with translation); a button shows all text.
     */
    private fun showBlockPicker(bitmap: Bitmap, result: com.google.mlkit.vision.text.Text) {
        val blocks = result.textBlocks.mapNotNull { b ->
            val box = b.boundingBox ?: return@mapNotNull null
            OcrOverlayView.Block(RectF(box), blockToRows(b))
        }
        if (blocks.isEmpty()) {
            toast(getString(R.string.ocr_empty))
            return
        }

        val overlay = OcrOverlayView(this, bitmap, blocks) { block ->
            showOcrDialog(block.rows)
        }

        AlertDialog.Builder(this)
            .setTitle(R.string.tap_paragraph)
            .setView(overlay)
            .setPositiveButton(R.string.cloud_ocr) { _, _ -> cloudRecognize(bitmap) }
            .setNeutralButton(R.string.show_all_text) { _, _ ->
                showOcrDialog(buildLayoutRows(result))
            }
            .setNegativeButton(R.string.close, null)
            .show()
    }

    /** High-accuracy OCR via Google Cloud Vision (handles vertical/complex text). */
    private fun cloudRecognize(bitmap: Bitmap) {
        if (BuildConfig.CLOUD_VISION_API_KEY.isBlank()) {
            toast(getString(R.string.cloud_key_missing))
            return
        }
        val progress = AlertDialog.Builder(this)
            .setMessage(getString(R.string.cloud_running))
            .setCancelable(false)
            .create()
        progress.show()

        Thread {
            try {
                val text = callCloudVision(bitmap)
                Log.i(TAG, "cloud OCR ok, len=${text.length}, head=${text.take(60)}")
                val rows = text.lines().filter { it.isNotBlank() }.map { listOf(it) }
                runOnUiThread {
                    progress.dismiss()
                    if (rows.isEmpty()) toast(getString(R.string.ocr_empty))
                    else showOcrDialog(rows)
                }
            } catch (e: Exception) {
                Log.e(TAG, "cloud OCR failed", e)
                runOnUiThread {
                    progress.dismiss()
                    toast(getString(R.string.cloud_failed) + ": " + (e.message ?: ""))
                }
            }
        }.start()
    }

    /** Posts the image to Cloud Vision DOCUMENT_TEXT_DETECTION; returns full text. */
    private fun callCloudVision(bitmap: Bitmap): String {
        // Downscale large images to keep the request small (detail is still ample).
        val maxSide = 2400
        val longest = maxOf(bitmap.width, bitmap.height)
        val bmp = if (longest > maxSide) {
            val s = maxSide.toFloat() / longest
            Bitmap.createScaledBitmap(
                bitmap, (bitmap.width * s).toInt(), (bitmap.height * s).toInt(), true
            )
        } else bitmap

        val baos = ByteArrayOutputStream()
        bmp.compress(Bitmap.CompressFormat.JPEG, 90, baos)
        val b64 = Base64.encodeToString(baos.toByteArray(), Base64.NO_WRAP)

        val body = JSONObject().put(
            "requests", JSONArray().put(
                JSONObject()
                    .put("image", JSONObject().put("content", b64))
                    .put(
                        "features", JSONArray().put(
                            JSONObject().put("type", "DOCUMENT_TEXT_DETECTION")
                        )
                    )
                    .put(
                        "imageContext",
                        JSONObject().put("languageHints", JSONArray().put("ko").put("en"))
                    )
            )
        )

        val url = URL(
            "https://vision.googleapis.com/v1/images:annotate?key=" +
                    BuildConfig.CLOUD_VISION_API_KEY
        )
        val conn = (url.openConnection() as HttpURLConnection).apply {
            requestMethod = "POST"
            doOutput = true
            connectTimeout = 15000
            readTimeout = 20000
            setRequestProperty("Content-Type", "application/json; charset=utf-8")
        }
        conn.outputStream.use { it.write(body.toString().toByteArray(Charsets.UTF_8)) }

        val code = conn.responseCode
        val stream = if (code in 200..299) conn.inputStream else conn.errorStream
        val response = stream.bufferedReader(Charsets.UTF_8).use { it.readText() }
        if (code !in 200..299) throw RuntimeException("HTTP $code: $response")

        val responses = JSONObject(response).optJSONArray("responses") ?: return ""
        val first = responses.optJSONObject(0) ?: return ""
        first.optJSONObject("error")?.let {
            throw RuntimeException(it.optString("message"))
        }
        return first.optJSONObject("fullTextAnnotation")?.optString("text").orEmpty()
    }

    private fun showOcrDialog(rows: List<List<String>>) {
        val originalText = renderRows(rows).ifBlank { getString(R.string.ocr_empty) }
        var current = originalText  // updated when translated; copy/share use this

        val pad = dp(16)
        val tv = TextView(this).apply {
            this.text = originalText
            typeface = Typeface.MONOSPACE
            textSize = 13f
            setTextIsSelectable(true)
            setPadding(pad, pad, pad, pad)
        }

        // Translation-direction picker + translate button.
        val spinner = Spinner(this).apply {
            adapter = ArrayAdapter(
                this@MainActivity, android.R.layout.simple_spinner_dropdown_item,
                TRANSLATE_DIRECTIONS
            )
        }
        val translateBtn = Button(this).apply {
            this.text = getString(R.string.translate)
            setOnClickListener {
                translateRows(rows, spinner.selectedItemPosition, tv) { current = it }
            }
        }
        val controls = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            setPadding(pad, pad, pad, 0)
            addView(spinner, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f))
            addView(translateBtn)
        }

        // Smart-action chips (phone/address/date/…) — populated async if any found.
        val actionsRow = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL }
        val actionsScroller = HorizontalScrollView(this).apply {
            addView(actionsRow)
            visibility = View.GONE
        }

        // Monospace + both-axis scrolling so reconstructed columns stay aligned.
        val textScroller = ScrollView(this).apply {
            addView(HorizontalScrollView(this@MainActivity).apply { addView(tv) })
        }
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            addView(controls)
            addView(actionsScroller)
            addView(textScroller, LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, dp(320)))
        }

        populateSmartActions(originalText, actionsRow, actionsScroller)

        AlertDialog.Builder(this)
            .setTitle(R.string.ocr_title)
            .setView(root)
            .setPositiveButton(R.string.copy) { _, _ ->
                getSystemService(ClipboardManager::class.java)
                    .setPrimaryClip(ClipData.newPlainText("OCR", current))
                toast(getString(R.string.copied))
            }
            .setNeutralButton(R.string.share) { _, _ -> shareText(current) }
            .setNegativeButton(R.string.close, null)
            .show()
    }

    /**
     * Translates each cell by its own language so the grid is preserved and mixed
     * Korean/English content is handled. Direction: 0=ko→en, 1=en→ko, 2=both.
     */
    private fun translateRows(
        rows: List<List<String>>, direction: Int, tv: TextView, onResult: (String) -> Unit
    ) {
        if (rows.isEmpty()) return
        val progress = AlertDialog.Builder(this)
            .setMessage(getString(R.string.translating))
            .setCancelable(false)
            .create()
        progress.show()

        Thread {
            var koEn: Translator? = null
            var enKo: Translator? = null
            try {
                val cond = DownloadConditions.Builder().build()
                if (direction == DIR_KO_EN || direction == DIR_BOTH) {
                    koEn = Translation.getClient(
                        TranslatorOptions.Builder()
                            .setSourceLanguage(TranslateLanguage.KOREAN)
                            .setTargetLanguage(TranslateLanguage.ENGLISH).build()
                    )
                    Tasks.await(koEn.downloadModelIfNeeded(cond))
                }
                if (direction == DIR_EN_KO || direction == DIR_BOTH) {
                    enKo = Translation.getClient(
                        TranslatorOptions.Builder()
                            .setSourceLanguage(TranslateLanguage.ENGLISH)
                            .setTargetLanguage(TranslateLanguage.KOREAN).build()
                    )
                    Tasks.await(enKo.downloadModelIfNeeded(cond))
                }

                val out = rows.map { row ->
                    row.map { cell -> translateCell(cell, direction, koEn, enKo) }
                }
                val text = renderRows(out)
                runOnUiThread {
                    progress.dismiss()
                    tv.text = text
                    onResult(text)
                }
            } catch (e: Exception) {
                Log.e(TAG, "translate failed", e)
                runOnUiThread {
                    progress.dismiss()
                    toast(getString(R.string.model_download_failed))
                }
            } finally {
                koEn?.close()
                enKo?.close()
            }
        }.start()
    }

    /** Picks the right translator for a cell based on whether it contains Hangul/Latin. */
    private fun translateCell(
        cell: String, direction: Int, koEn: Translator?, enKo: Translator?
    ): String {
        if (cell.isBlank()) return cell
        val cellHasHangul = hasHangul(cell)
        val hasLatin = cell.any { it in 'a'..'z' || it in 'A'..'Z' }

        // translator + whether the target script is Korean
        val (translator, targetKorean) = when (direction) {
            DIR_KO_EN -> if (cellHasHangul) koEn to false else null to false
            DIR_EN_KO -> if (hasLatin && !cellHasHangul) enKo to true else null to false
            else -> when {
                cellHasHangul -> koEn to false
                hasLatin -> enKo to true
                else -> null to false
            }
        }
        if (translator == null) return cell

        val res = try {
            Tasks.await(translator.translate(cell))
        } catch (e: Exception) {
            return cell  // keep original on per-cell failure
        }

        // A proper noun (e.g. "LEESEUNGMIN") isn't really translated — the model just
        // lowercases it. Detect "nothing actually translated" and keep the original.
        if (res.equals(cell, ignoreCase = true)) return cell
        if (targetKorean && !hasHangul(res)) return cell      // no Hangul produced
        if (!targetKorean && hasHangul(res)) return cell      // Korean left untranslated
        return res
    }

    private fun hasHangul(s: String): Boolean =
        s.any { it.code in 0xAC00..0xD7A3 || it.code in 0x3130..0x318F }

    private fun shareText(text: String) {
        val intent = Intent(Intent.ACTION_SEND).apply {
            type = "text/plain"
            putExtra(Intent.EXTRA_TEXT, text)
        }
        startActivity(Intent.createChooser(intent, getString(R.string.share)))
    }

    // ---- Smart actions (entity extraction) ---------------------------------

    /** Detects phone/address/date/email/url in the text and adds tappable chips. */
    private fun populateSmartActions(text: String, row: LinearLayout, scroller: View) {
        ensureEntityModel {
            val params = EntityExtractionParams.Builder(text).build()
            entityExtractor.annotate(params).addOnSuccessListener { annotations ->
                val seen = HashSet<String>()
                for (ann in annotations) {
                    val entity = ann.entities.firstOrNull() ?: continue
                    val label = actionLabel(entity.type, ann.annotatedText) ?: continue
                    if (!seen.add(entity.type.toString() + ann.annotatedText)) continue
                    row.addView(com.google.android.material.button.MaterialButton(
                        this, null,
                        com.google.android.material.R.attr.materialButtonOutlinedStyle
                    ).apply {
                        this.text = label
                        textSize = 12f
                        isAllCaps = false
                        setTextColor(Color.WHITE)
                        val lp = LinearLayout.LayoutParams(
                            LinearLayout.LayoutParams.WRAP_CONTENT,
                            LinearLayout.LayoutParams.WRAP_CONTENT
                        ).apply { marginEnd = dp(6) }
                        layoutParams = lp
                        setOnClickListener { fireEntityAction(entity, ann.annotatedText) }
                    })
                }
                if (row.childCount > 0) scroller.visibility = View.VISIBLE
            }
        }
    }

    private fun ensureEntityModel(onReady: () -> Unit) {
        if (entityModelReady) { onReady(); return }
        entityExtractor.downloadModelIfNeeded()
            .addOnSuccessListener { entityModelReady = true; onReady() }
            .addOnFailureListener { Log.e(TAG, "entity model download failed", it) }
    }

    private fun actionLabel(type: Int, text: String): String? = when (type) {
        Entity.TYPE_PHONE -> "📞 $text"
        Entity.TYPE_EMAIL -> "✉ $text"
        Entity.TYPE_URL -> "🔗 $text"
        Entity.TYPE_ADDRESS -> "📍 지도: ${text.take(12)}…"
        Entity.TYPE_DATE_TIME -> "🗓 일정: $text"
        Entity.TYPE_TRACKING_NUMBER -> "📦 배송조회"
        else -> null
    }

    private fun fireEntityAction(entity: Entity, text: String) {
        val intent = when (entity.type) {
            Entity.TYPE_PHONE ->
                Intent(Intent.ACTION_DIAL, Uri.parse("tel:" + text.filter { it.isDigit() || it == '+' }))
            Entity.TYPE_EMAIL ->
                Intent(Intent.ACTION_SENDTO, Uri.parse("mailto:$text"))
            Entity.TYPE_URL ->
                Intent(Intent.ACTION_VIEW, Uri.parse(if (text.startsWith("http")) text else "http://$text"))
            Entity.TYPE_ADDRESS ->
                Intent(Intent.ACTION_VIEW, Uri.parse("geo:0,0?q=" + Uri.encode(text)))
            Entity.TYPE_DATE_TIME -> {
                val millis = entity.asDateTimeEntity()?.timestampMillis ?: System.currentTimeMillis()
                Intent(Intent.ACTION_INSERT)
                    .setData(CalendarContract.Events.CONTENT_URI)
                    .putExtra(CalendarContract.EXTRA_EVENT_BEGIN_TIME, millis)
                    .putExtra(CalendarContract.Events.TITLE, "스캔한 일정")
            }
            Entity.TYPE_TRACKING_NUMBER ->
                Intent(Intent.ACTION_VIEW, Uri.parse(
                    "https://search.naver.com/search.naver?query=" + Uri.encode("$text 택배조회")))
            else -> return
        }
        try {
            startActivity(intent)
        } catch (e: Exception) {
            toast(getString(R.string.no_app_for_action))
        }
    }

    private fun saveToGallery(bitmap: Bitmap): Uri? {
        val name = "VisionLab_${System.currentTimeMillis()}.jpg"
        val values = ContentValues().apply {
            put(MediaStore.Images.Media.DISPLAY_NAME, name)
            put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg")
            put(MediaStore.Images.Media.RELATIVE_PATH, "Pictures/VisionLab")
        }
        val resolver = contentResolver
        val uri = resolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values)
            ?: return null
        return try {
            resolver.openOutputStream(uri)?.use { out: OutputStream ->
                bitmap.compress(Bitmap.CompressFormat.JPEG, 95, out)
            }
            uri
        } catch (e: Exception) {
            Log.e(TAG, "Save failed", e)
            null
        }
    }

    private fun shareLastCapture() {
        val uri = lastSavedUri
        if (uri == null) {
            toast(getString(R.string.nothing_to_share))
            return
        }
        val intent = Intent(Intent.ACTION_SEND).apply {
            type = "image/jpeg"
            putExtra(Intent.EXTRA_STREAM, uri)
            addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
        }
        startActivity(Intent.createChooser(intent, getString(R.string.share)))
    }

    private fun tickFps(micros: Long) {
        frameCount++
        procMicrosAccum += micros
        val now = System.currentTimeMillis()
        val elapsed = now - lastFpsTimestamp
        if (elapsed >= 1000) {
            val fps = frameCount * 1000 / elapsed
            val avgMs = procMicrosAccum / 1000.0 / frameCount
            binding.fpsText.text = "FPS: $fps  |  native: %.1f ms".format(avgMs)
            frameCount = 0
            procMicrosAccum = 0
            lastFpsTimestamp = now
        }
    }

    private fun toast(msg: String) = Toast.makeText(this, msg, Toast.LENGTH_SHORT).show()

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
        liveExecutor.shutdown()
        outMat?.release()
        recognizer.close()
        runCatching { liveKoEn.close() }
        runCatching { liveEnKo.close() }
        runCatching { entityExtractor.close() }
    }

    /** Processes [inAddr] into [outAddr] for the given [mode]; returns native time in µs. */
    private external fun nativeProcess(
        inAddr: Long, outAddr: Long, mode: Int, threshold1: Double, threshold2: Double
    ): Long

    /** Detects + perspective-warps a document; returns false if none found. */
    private external fun nativeWarpDocument(
        inAddr: Long, outAddr: Long, threshold1: Double, threshold2: Double
    ): Boolean

    /** Grayscale + contrast + sharpen a full photo for better OCR. */
    private external fun nativeEnhance(inAddr: Long, outAddr: Long)

    companion object {
        private const val TAG = "VisionLab"
        private const val MODE_DOC = 7
        private const val LIVE_TRANSLATE = 8
        private const val MOTION_THRESHOLD = 12  // mean luma diff to count as camera movement

        // Per-cell translation directions (index matches translateRows).
        private const val DIR_KO_EN = 0
        private const val DIR_EN_KO = 1
        private const val DIR_BOTH = 2
        private val TRANSLATE_DIRECTIONS = listOf("한국어 → 영어", "영어 → 한국어", "한 ↔ 영 (양방향)")

        init {
            System.loadLibrary("detectapp")
        }
    }
}

/**
 * Draws the captured image (fit-center) with each OCR text block outlined, and
 * reports which block was tapped (mapping touch coords back to bitmap space).
 */
private class OcrOverlayView(
    context: Context,
    private val bitmap: Bitmap,
    private val blocks: List<Block>,
    private val onTap: (Block) -> Unit
) : View(context) {

    class Block(val rect: RectF, val rows: List<List<String>>)

    private val imgPaint = Paint(Paint.FILTER_BITMAP_FLAG)
    private val fillPaint = Paint().apply {
        color = 0x3300E676; style = Paint.Style.FILL
    }
    private val strokePaint = Paint().apply {
        color = 0xFF00E676.toInt(); style = Paint.Style.STROKE
        strokeWidth = 4f; isAntiAlias = true
    }
    private val matrix = Matrix()

    override fun onMeasure(widthSpec: Int, heightSpec: Int) {
        val width = MeasureSpec.getSize(widthSpec)
        val height = (440 * resources.displayMetrics.density).toInt()
        setMeasuredDimension(width, height)
    }

    private fun fitMatrix() {
        val scale = minOf(width.toFloat() / bitmap.width, height.toFloat() / bitmap.height)
        val dx = (width - bitmap.width * scale) / 2f
        val dy = (height - bitmap.height * scale) / 2f
        matrix.reset()
        matrix.setScale(scale, scale)
        matrix.postTranslate(dx, dy)
    }

    override fun onDraw(canvas: Canvas) {
        fitMatrix()
        canvas.drawBitmap(bitmap, matrix, imgPaint)
        for (b in blocks) {
            val r = RectF(b.rect)
            matrix.mapRect(r)
            canvas.drawRect(r, fillPaint)
            canvas.drawRect(r, strokePaint)
        }
    }

    override fun onTouchEvent(event: MotionEvent): Boolean {
        if (event.action == MotionEvent.ACTION_UP) {
            fitMatrix()
            val inverse = Matrix()
            matrix.invert(inverse)
            val pts = floatArrayOf(event.x, event.y)
            inverse.mapPoints(pts)
            val hit = blocks.firstOrNull { it.rect.contains(pts[0], pts[1]) }
            if (hit != null) {
                performClick()
                onTap(hit)
            }
        }
        return true
    }

    override fun performClick(): Boolean {
        super.performClick()
        return true
    }
}
